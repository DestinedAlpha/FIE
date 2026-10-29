"""Temporal, multi-source neural challenger and chronological comparison.

Unavailable sources are masked. Text comes from timestamped official news,
not social posts. No hindsight-optimal team or future manager rank is a label.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.special import expit, softmax

from .decision import PARAMETERS, minute_band, paired_bootstrap, score_metrics
from .fusion_data import tree_matrix
from .storage import Store, atomic_json, digest, now

INPUTS = ("tab", "sequence", "sequence_mask", "text", "news", "market", "workload", "quality")


def architecture():
    import torch
    from torch import nn

    class FusionNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.numeric = nn.Sequential(nn.Linear(26, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 64))
            self.history = nn.Linear(32, 64)
            self.cls = nn.Parameter(torch.zeros(1, 1, 64))
            self.position = nn.Parameter(torch.randn(1, 13, 64) * 0.02)
            self.temporal = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(64, 4, 128, 0.15, batch_first=True, norm_first=True),
                2,
                enable_nested_tensor=False,
            )
            self.words = nn.Embedding(4096, 32, padding_idx=0)
            self.text_conv = nn.Conv1d(32, 64, 3, padding=1)
            self.news = nn.Sequential(nn.Linear(77, 128), nn.GELU(), nn.Linear(128, 64))
            self.market = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, 64))
            self.workload = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 64))
            self.source_type = nn.Parameter(torch.randn(1, 5, 64) * 0.02)
            self.fusion = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(64, 4, 128, 0.15, batch_first=True, norm_first=True),
                2,
                enable_nested_tensor=False,
            )
            self.head = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.15))
            self.band = nn.Linear(128, 4)
            self.points = nn.Linear(128, 3)
            self.start = nn.Linear(128, 1)
            self.auxiliary = nn.Linear(128, 7)

        def forward(self, batch):
            n = len(batch["tab"])
            seq = torch.cat([self.cls.expand(n, -1, -1), self.history(batch["sequence"])], 1) + self.position
            mask = torch.cat(
                [torch.zeros(n, 1, dtype=torch.bool, device=seq.device), ~batch["sequence_mask"]], 1
            )
            history = self.temporal(seq, src_key_padding_mask=mask)[:, 0]
            text = torch.nn.functional.gelu(self.text_conv(self.words(batch["text"]).transpose(1, 2)))
            token_mask = (batch["text"] != 0).unsqueeze(1)
            text = (text * token_mask).sum(2) / token_mask.sum(2).clamp_min(1)
            source = (
                torch.stack(
                    [
                        self.numeric(batch["tab"]),
                        history,
                        self.news(torch.cat([text, batch["news"]], 1)),
                        self.market(batch["market"]),
                        self.workload(batch["workload"]),
                    ],
                    1,
                )
                + self.source_type
            )
            missing = torch.cat(
                [torch.zeros(n, 1, dtype=torch.bool, device=source.device), batch["quality"] <= 0], 1
            )
            fused = self.fusion(source, src_key_padding_mask=missing)[:, 0]
            hidden = self.head(fused)
            return (
                self.band(hidden),
                self.points(hidden),
                self.start(hidden).squeeze(1),
                self.auxiliary(hidden),
            )

    return FusionNetwork()


def standardized(arrays, indices, normalization=None):
    x = {k: arrays[k][indices].copy() for k in INPUTS}
    if normalization is None:
        normalization = {
            k: {"mean": x[k].mean(0).tolist(), "scale": np.maximum(x[k].std(0), 0.1).tolist()}
            for k in ("tab", "news", "market", "workload")
        }
    for k, stats in normalization.items():
        x[k] = np.clip((x[k] - np.array(stats["mean"])) / np.array(stats["scale"]), -10, 10).astype(
            np.float32
        )
    return x, normalization


def tensor_batch(x, indices):
    import torch

    return {
        k: torch.as_tensor(
            v[indices],
            dtype=torch.int64 if k == "text" else torch.bool if k == "sequence_mask" else torch.float32,
        )
        for k, v in x.items()
    }


def neural_predict(model, x):
    import torch

    logits, means, starts = [], [], []
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(x["tab"]), 1024):
            a, b, c, _ = model(tensor_batch(x, slice(first, first + 1024)))
            logits.append(a.numpy())
            means.append(b.numpy())
            starts.append(c.numpy())
    return np.concatenate(logits), np.concatenate(means), np.concatenate(starts)


def fit_neural(arrays, train, calibration, seed=47, epochs=20):
    import torch
    from torch.nn import functional as f

    torch.set_num_threads(3)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = architecture()
    x, normalization = standardized(arrays, train)
    xc, _ = standardized(arrays, calibration, normalization)
    bands = torch.tensor(minute_band(arrays["minutes"][train]))
    target = torch.tensor(arrays["base_target"][train])
    actual = arrays["target"][calibration]
    starts = torch.tensor(arrays["start"][train])
    observed_start = torch.tensor(arrays["start_mask"][train])
    aux = torch.tensor(arrays["aux"][train])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    generator = np.random.default_rng(seed)
    best, state, stale, trace = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        for first in range(0, len(train), 512):
            if first == 0:
                order = generator.permutation(len(train))
            idx = order[first : first + 512]
            logits, conditional, starter, components = model(tensor_batch(x, idx))
            band = bands[idx]
            appeared = band > 0
            loss = f.cross_entropy(logits, band)
            if appeared.any():
                chosen = conditional[appeared, band[appeared] - 1]
                loss = loss + 0.15 * f.mse_loss(chosen, target[idx][appeared])
            valid = observed_start[idx]
            if valid.any():
                loss = loss + 0.2 * f.binary_cross_entropy_with_logits(starter[valid], starts[idx][valid])
            labels = aux[idx]
            known = (labels >= 0) & appeared[:, None]
            if known.any():
                loss = loss + 0.015 * f.poisson_nll_loss(
                    components.clamp(-8, 4)[known], labels[known], log_input=True
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
        logits, conditional, _ = neural_predict(model, xc)
        probability = softmax(logits, 1)
        mean = (probability[:, 1:] * conditional).sum(1) + (
            probability * arrays["dc_prior"][calibration]
        ).sum(1) * 2
        rmse = score_metrics(actual, mean)["rmse"]
        trace.append(rmse)
        if rmse < best - 0.0002:
            best, state, stale = rmse, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(state)
    return (
        model,
        normalization,
        {
            "epochs": len(trace),
            "selected_epoch": int(np.argmin(trace)) + 1,
            "calibration_rmse": best,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    )


def fit_tree(arrays, matrix, train, calibration):
    parameters = PARAMETERS | {"n_estimators": 240, "num_leaves": 24, "reg_lambda": 15}
    band = minute_band(arrays["minutes"][train])
    appeared = band > 0
    minutes = lgb.LGBMClassifier(**parameters).fit(matrix[train], band)
    points = lgb.LGBMRegressor(**parameters).fit(
        np.column_stack([matrix[train][appeared], band[appeared]]), arrays["base_target"][train][appeared]
    )
    observed = arrays["start_mask"][train]
    starter = lgb.LGBMClassifier(**parameters).fit(matrix[train][observed], arrays["start"][train][observed])
    return {"minutes": minutes, "points": points, "starter": starter}


def tree_predict(bundle, matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def predict(model, values):
        return (
            np.asarray(model.predict(values, num_threads=1))
            if isinstance(model, lgb.Booster)
            else np.asarray(model.predict(values))
        )

    model = bundle["minutes"]
    probability = (
        np.asarray(model.predict_proba(matrix)) if hasattr(model, "predict_proba") else predict(model, matrix)
    )
    conditional = np.column_stack(
        [np.zeros(len(matrix))]
        + [
            predict(bundle["points"], np.column_stack([matrix, np.full(len(matrix), band)]))
            for band in range(1, 4)
        ]
    )
    starter = bundle["starter"]
    starts = (
        starter.predict_proba(matrix)[:, 1] if hasattr(starter, "predict_proba") else predict(starter, matrix)
    )
    return probability, conditional, np.asarray(starts)


def calibrate(probability, conditional, starts, arrays, calibration):
    from .models import curve_fit

    band = minute_band(arrays["minutes"][calibration])
    curves = {str(k): curve_fit(probability[:, k], (band == k).astype(int)) for k in range(4)}
    mask = arrays["start_mask"][calibration]
    curves["start"] = curve_fit(starts[mask], arrays["start"][calibration][mask])
    offsets = {}
    for k in range(1, 4):
        selected = band == k
        # Shrink small-band corrections; fit only on the separate calibration slice.
        residual = arrays["base_target"][calibration][selected] - conditional[selected, k]
        offsets[str(k)] = float(residual.sum() / (len(residual) + 100))
    return curves, offsets


def corrected(probability, conditional, starts, curves, offsets):
    from .models import curve_apply

    p = np.column_stack([curve_apply(probability[:, k], curves[str(k)]) for k in range(4)])
    p /= np.maximum(p.sum(1, keepdims=True), 1e-9)
    c = conditional.copy()
    for k in range(1, 4):
        c[:, k] += offsets[str(k)]
    return p, c, curve_apply(starts, curves["start"])


def predict_neural(model, normalization, arrays, indices) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, _ = standardized(arrays, indices, normalization)
    logits, conditional, starts = neural_predict(model, x)
    return softmax(logits, 1), np.column_stack([np.zeros(len(indices)), conditional]), expit(starts)


def load_data(store, feature_id=None):
    spec = store.get("fusion_features", feature_id)
    path = store.artifact_path(spec["inputs_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["hash"]:
        raise ValueError("Fusion feature integrity failed")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    metadata_path = store.artifact_path(spec["metadata_path"])
    if (
        spec.get("metadata_hash")
        and hashlib.sha256(metadata_path.read_bytes()).hexdigest() != spec["metadata_hash"]
    ):
        raise ValueError("Fusion metadata integrity failed")
    meta = pl.read_parquet(metadata_path).to_dicts()
    return arrays, meta, spec


def train_fusion(store: Store, epochs=20):
    """Compare independently fitted trees and temporal fusion in six forward blocks."""
    import torch

    from .decision import build_dataset
    from .fusion_data import build_fusion_data

    build_dataset(store)
    build_fusion_data(store)
    arrays, meta, features = load_data(store)
    matrix = tree_matrix(
        arrays["tab"],
        arrays["news"],
        arrays["market"],
        arrays["workload"],
        arrays["text"],
        arrays["sequence"],
    )
    groups = np.array([r["deadline_group"] for r in meta])
    unique = sorted(set(groups))
    testing = [g for g in unique if g >= "2025/26-GW01"]
    predictions = []
    folds = []
    protocol = {
        "test": "Expanding eight-deadline blocks from 2025/26, never shuffled",
        "calibration": "Four immediately earlier deadlines excluded from fitting; neural early stopping and probability calibration share this earlier slice",
        "neural": "Two-layer temporal transformer, hashed official news CNN, numerical/market/workload branches, two-layer masked source attention; minutes, conditional points, starters and component auxiliary heads",
        "seed": 47,
        "max_epochs": epochs,
        "promotion": "Upper deadline-bootstrap RMSE delta < 0 against existing production on identical rows; regular-starter improvement also required; tree and neural both disclosed",
        "expert": "Only lagged choices from managers qualified on prior seasons; no current rank",
        "social": "No raw Reddit training; specialist statements remain source evidence until prospectively validated",
        "selection": "Choose one architecture by validation; no post-test optimized ensemble",
    }
    # The comparison uses exact old out-of-fold predictions for the same rows.
    old, reference, oldrows = reference_forecasts(store, meta)
    for first in range(0, len(testing), 8):
        block = testing[first : first + 8]
        before = [g for g in unique if g < block[0]]
        cut = before[-4]
        train = np.flatnonzero(groups < cut)
        cal = np.flatnonzero((groups >= cut) & (groups < block[0]))
        test = np.flatnonzero(np.isin(groups, block))
        tree = fit_tree(arrays, matrix, train, cal)
        curves, offsets = calibrate(*tree_predict(tree, matrix[cal]), arrays, cal)
        tp, tc, ts = corrected(*tree_predict(tree, matrix[test]), curves, offsets)
        model, normalization, training = fit_neural(arrays, train, cal, epochs=epochs)
        nc, no = calibrate(*predict_neural(model, normalization, arrays, cal), arrays, cal)
        np_, ncond, ns = corrected(*predict_neural(model, normalization, arrays, test), nc, no)
        dc = arrays["dc_prior"][test]
        tmean = (tp * (tc + 2 * dc)).sum(1)
        nmean = (np_ * (ncond + 2 * dc)).sum(1)
        for j, index in enumerate(test):
            row = meta[index]
            prior = oldrows.get((row["season"], row["player_id"], row["fixture_id"]))
            if prior is None:
                continue
            predictions.append(
                row
                | {
                    "actual": float(arrays["target"][index]),
                    "minutes": float(arrays["minutes"][index]),
                    "baseline": prior["predicted"],
                    "tree": float(tmean[j]),
                    "network": float(nmean[j]),
                    "tree_p_any": float(1 - tp[j, 0]),
                    "network_p_any": float(1 - np_[j, 0]),
                    "tree_p60": float(tp[j, 2:].sum()),
                    "network_p60": float(np_[j, 2:].sum()),
                }
            )
        folds.append(
            {
                "test": block,
                "train_rows": len(train),
                "calibration_rows": len(cal),
                "test_rows": len(test),
                "neural": training,
            }
        )
        print(f"Fusion validated {block[0]}–{block[-1]}: {training}", flush=True)
    y = np.array([r["actual"] for r in predictions])
    baseline = np.array([r["baseline"] for r in predictions])
    regular = np.array([r["prior_start"] >= 0.6 for r in predictions])
    metrics = {}
    eligible = []
    for kind in ("tree", "network"):
        values = np.array([r[kind] for r in predictions])
        ci = paired_bootstrap(predictions, kind)
        passed = (
            ci[1] < 0
            and score_metrics(y[regular], values[regular])["rmse"]
            < score_metrics(y[regular], baseline[regular])["rmse"]
        )
        metrics[kind] = {
            "metrics": score_metrics(y, values),
            "paired_vs_production_95ci": ci,
            "regular_starters": score_metrics(y[regular], values[regular]),
            "passed": bool(passed),
        }
        if passed:
            eligible.append(kind)
    winner = min(eligible, key=lambda kind: metrics[kind]["metrics"]["rmse"]) if eligible else None
    report = {
        "at": now(),
        "version": "fusion-v1",
        "feature_id": features["id"],
        "compared_decision_model_id": old["id"],
        "reference_decision_model_id": reference["id"],
        "protocol": protocol,
        "folds": folds,
        "baseline": score_metrics(y, baseline),
        "regular_starter_baseline": score_metrics(y[regular], baseline[regular]),
        "candidates": metrics,
        "selected": winner,
        "promoted": False,
        "source_coverage": features["coverage"],
        "limits": [
            "Expert inputs cover current season only; no backfilled current-ranked expert cohort",
            "Workload uses retrospective match facts with conservative completion cutoff, not archived publication times",
            "Reddit leads and current specialist opinions have no validated numerical uplift",
            "This is an architecture comparison, not proof of winning a league or absence of bias",
        ],
    }
    # Save the final candidate for reproducible inference, including when it fails promotion.
    cut = unique[-4]
    train = np.flatnonzero(groups < cut)
    cal = np.flatnonzero(groups >= cut)
    directory = (
        store.root
        / "models"
        / ("fusion-" + digest({"features": features["hash"], "protocol": protocol})[:16])
    )
    directory.mkdir(parents=True, exist_ok=True)
    tree = fit_tree(arrays, matrix, train, cal)
    curves, offsets = calibrate(*tree_predict(tree, matrix[cal]), arrays, cal)
    tree_spec = {"curves": curves, "offsets": offsets, "hashes": {}}
    for name, model in tree.items():
        path = directory / (name + ".txt")
        model.booster_.save_model(str(path))
        tree_spec["hashes"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    model, normalization, training = fit_neural(arrays, train, cal, epochs=epochs)
    nc, no = calibrate(*predict_neural(model, normalization, arrays, cal), arrays, cal)
    weights = directory / "network.pt"
    torch.save(model.state_dict(), weights)
    report.update(
        directory=directory.name,
        tree_spec=tree_spec,
        network_spec={
            "normalization": normalization,
            "curves": nc,
            "offsets": no,
            "hash": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "training": training,
        },
        production_fit_before=cut,
        production_training_rows=len(train),
        production_calibration_rows=len(cal),
    )
    prediction_path = directory / "heldout.parquet"
    pl.DataFrame(predictions, infer_schema_length=None).write_parquet(prediction_path)
    report["prediction_path"] = str(prediction_path)
    report["id"] = store.put("fusion_model", report)
    atomic_json(
        Path("artifacts/evaluations/fusion_model.json"),
        {k: v for k, v in report.items() if k not in ("tree_spec", "network_spec")},
    )
    return freeze_fusion_artifacts(store, report)


def candidate_predict(store, spec, arrays, kind):
    """Hash-verified, non-pickle loading for either independently fitted candidate."""
    directory = store.root / "models" / spec["directory"]
    if kind == "tree":
        models = {}
        for key, expected in spec["tree_spec"]["hashes"].items():
            path = directory / (key + ".txt")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Fusion tree integrity verification failed")
            models[key] = lgb.Booster(model_file=str(path))
        matrix = tree_matrix(
            arrays["tab"],
            arrays["news"],
            arrays["market"],
            arrays["workload"],
            arrays["text"],
            arrays["sequence"],
        )
        return corrected(
            *tree_predict(models, matrix), spec["tree_spec"]["curves"], spec["tree_spec"]["offsets"]
        )
    if kind != "network":
        raise ValueError("Unknown fusion candidate")
    import torch

    torch.set_num_threads(3)
    path = directory / "network.pt"
    config = spec["network_spec"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["hash"]:
        raise ValueError("Fusion neural integrity verification failed")
    model = architecture()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    raw = predict_neural(model, config["normalization"], arrays, np.arange(len(arrays["tab"])))
    return corrected(*raw, config["curves"], config["offsets"])


def compare_live(store: Store):
    from .decision import future_inputs, load_bundle, predict_models
    from .fusion_data import live_inputs
    from .ingestion import load_market

    players, market = load_market(store)
    start = next(e["id"] for e in market["events"] if e.get("is_next"))
    rows, _ = future_inputs(store, market, players, start, start)
    inputs = live_inputs(store, rows, market, players)
    spec = store.get("fusion_model")
    old = store.get("decision_model")
    base, p, c = predict_models(
        load_bundle(store, old), np.array([r["x"] for r in rows]), [r["dc_probability"] for r in rows]
    )
    if old.get("fusion_model_id"):
        p, c, _ = production_predictions(store, old, rows, market, players)
    # Match the production baseline's availability adjustment exactly.
    availability = np.array([r["availability"] for r in rows])
    p[:, 1:] *= availability[:, None]
    p[:, 0] = 1 - p[:, 1:].sum(1)
    dc = np.array([r["dc_probability"] for r in rows])
    base = (p * (c + 2 * dc)).sum(1)
    means = {}
    source_sensitivity: dict[str, dict] = {}
    for kind in ("tree", "network"):
        p, c, _ = candidate_predict(store, spec, inputs, kind)
        means[kind] = (p * (c + 2 * dc)).sum(1)
        source_sensitivity[kind] = {}
        for source in ("news", "market", "workload"):
            ablated = {k: v.copy() for k, v in inputs.items()}
            ablated[source][:] = 0
            if source == "news":
                ablated["text"][:] = 0
                ablated["quality"][:, 1] = 0
            elif source == "market":
                ablated["quality"][:, 2] = 0
            else:
                ablated["quality"][:, 3] = 0
            ap, ac, _ = candidate_predict(store, spec, ablated, kind)
            am = (ap * (ac + 2 * dc)).sum(1)
            source_sensitivity[kind][source] = {
                "mean_absolute_prediction_change": float(np.mean(np.abs(am - means[kind]))),
                "interpretation": "Inference masking sensitivity, not an independently retrained source ablation or causal benefit",
            }
    output = []
    for i, row in enumerate(rows):
        player = players[row["player_id"]]
        output.append(
            {
                "player_id": player.id,
                "name": player.web_name,
                "event": start,
                "production": float(base[i]),
                "tree": float(means["tree"][i]),
                "network": float(means["network"][i]),
            }
        )
    report = {
        "at": now(),
        "fusion_model_id": spec["id"],
        "decision_model_id": old["id"],
        "event": start,
        "players": sorted(output, key=lambda r: -r["production"]),
        "source_sensitivity": source_sensitivity,
        "decision": "Candidate predictions do not overwrite the production forecast before promotion",
    }
    report["id"] = store.put("fusion_comparison", report)
    atomic_json(
        Path("artifacts/evaluations/fusion_comparison.json"),
        {k: v for k, v in report.items() if k != "players"},
    )
    return report


def audit_fusion_decisions(store: Store):
    """Compare actual captain outcomes and matched legal account policies."""
    from collections import defaultdict

    from .decision_audit import replay_policy

    spec = store.get("fusion_model")
    rows = pl.read_parquet(store.artifact_path(spec["prediction_path"])).to_dicts()
    groups = defaultdict(list)
    for row in rows:
        groups[row["deadline_group"]].append(row)
    captains = []
    for group, records in sorted(groups.items()):
        players: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for row in records:
            p = players[row["player_id"]]
            for field in ("baseline", "tree", "network", "actual", "minutes"):
                p[field] += row[field]
        outcomes = {}
        for field in ("baseline", "tree", "network"):
            chosen = sorted(players, key=lambda pid: (-players[pid][field], pid))[:2]
            outcomes[field] = next(
                (players[pid]["actual"] for pid in chosen if players[pid]["minutes"] > 0), 0
            )
        captains.append({"deadline_group": group, **outcomes})
    policies = {}
    for kind in ("tree", "network"):
        transformed = [
            row | {"predicted": row[kind], "network": row["network"], "price_observed": True} for row in rows
        ]
        result = replay_policy(
            store, spec={"id": spec["id"], "predictions": transformed}, artifact_kind="fusion_policy_" + kind
        )
        policies[kind] = {
            name: {k: v for k, v in policy.items() if k != "ledger"}
            for name, policy in result["policies"].items()
        }
    report = {
        "at": now(),
        "fusion_model_id": spec["id"],
        "captain_totals": {
            kind: sum(row[kind] for row in captains) for kind in ("baseline", "tree", "network")
        },
        "captain_deadlines": len(captains),
        "policies": policies,
        "limits": [
            "Prior production and candidates share pre-deadline registry prices and the same initial account in each paired replay",
            "No chips in this policy test",
            "No prospective season-rank claim",
        ],
        "promotable": bool(
            spec["selected"]
            and policies[spec["selected"]]["transfer-learned"]["realized_net_points"]
            >= policies[spec["selected"]]["transfer-shrinkage"]["realized_net_points"]
            and sum(r[spec["selected"]] for r in captains) >= sum(r["baseline"] for r in captains)
        ),
    }
    report["id"] = store.put("fusion_decisions", report)
    atomic_json(Path("artifacts/evaluations/fusion_decisions.json"), report)
    return report


def residual_pools(arrays, metadata, indices, conditional):
    bands = minute_band(arrays["minutes"][indices])
    residual = arrays["base_target"][indices] - conditional[np.arange(len(indices)), bands]
    positions = np.array([metadata[i]["position"] for i in indices])
    pools = {}
    for position in range(1, 5):
        for band in range(1, 4):
            values = residual[(positions == position) & (bands == band)]
            if len(values) < 30:
                values = residual[bands == band]
            if not len(values):
                raise ValueError("Insufficient earlier residual calibration")
            pools[f"{position}:{band}"] = np.quantile(values - values.mean(), np.linspace(0, 1, 201)).tolist()
    return pools


def validate_fusion_worlds(store: Store, worlds=512):
    from .decision import sample_point_outcomes

    spec = store.get("fusion_model")
    kind = spec["selected"]
    if kind is None:
        return {"status": "No candidate passed forecast promotion"}
    arrays, meta, _ = load_data(store, spec["feature_id"])
    matrix = tree_matrix(
        arrays["tab"],
        arrays["news"],
        arrays["market"],
        arrays["workload"],
        arrays["text"],
        arrays["sequence"],
    )
    groups = np.array([r["deadline_group"] for r in meta])
    unique = sorted(set(groups))
    testing = [g for g in unique if g >= "2025/26-GW01"]
    records = []
    for first in range(0, len(testing), 8):
        block = testing[first : first + 8]
        cut = [g for g in unique if g < block[0]][-4]
        train = np.flatnonzero(groups < cut)
        cal = np.flatnonzero((groups >= cut) & (groups < block[0]))
        test = np.flatnonzero(np.isin(groups, block))
        if kind == "tree":
            model = fit_tree(arrays, matrix, train, cal)
            rawcal = tree_predict(model, matrix[cal])
            rawtest = tree_predict(model, matrix[test])
        else:
            model, norm, _ = fit_neural(arrays, train, cal, epochs=spec["protocol"]["max_epochs"])
            rawcal = predict_neural(model, norm, arrays, cal)
            rawtest = predict_neural(model, norm, arrays, test)
        curves, offsets = calibrate(*rawcal, arrays, cal)
        _, calconditional, _ = corrected(*rawcal, curves, offsets)
        probability, conditional, _ = corrected(*rawtest, curves, offsets)
        pools = residual_pools(arrays, meta, cal, calconditional)
        rng = np.random.default_rng(100 + first)
        for i, index in enumerate(test):
            points, _, _ = sample_point_outcomes(
                rng,
                probability[i],
                conditional[i],
                arrays["dc_prior"][index],
                pools,
                meta[index]["position"],
                rng.random(worlds),
                rng.random(worlds),
            )
            q = np.quantile(points, [0.05, 0.1, 0.9, 0.95])
            actual = arrays["target"][index]
            ordered = np.sort(points)
            crps = float(
                np.abs(points - actual).mean()
                - np.dot(2 * np.arange(1, worlds + 1) - worlds - 1, ordered) / (worlds**2)
            )
            mean = float(np.dot(probability[i], conditional[i] + 2 * arrays["dc_prior"][index]))
            records.append(
                {
                    "regular": meta[index]["prior_start"] >= 0.6,
                    "covered80": bool(q[1] <= actual <= q[2]),
                    "covered90": bool(q[0] <= actual <= q[3]),
                    "crps": crps,
                    "mean_error": float(points.mean() - mean),
                }
            )
        print("Fusion distribution audit", block[-1], flush=True)

    def aggregate(rows):
        return {
            "n": len(rows),
            "coverage80": float(np.mean([r["covered80"] for r in rows])),
            "coverage90": float(np.mean([r["covered90"] for r in rows])),
            "crps": float(np.mean([r["crps"] for r in rows])),
            "mean_sampling_bias": float(np.mean([r["mean_error"] for r in rows])),
        }

    report = {
        "at": now(),
        "fusion_model_id": spec["id"],
        "selected": kind,
        "worlds_per_row": worlds,
        "all": aggregate(records),
        "regular_starters": aggregate([r for r in records if r["regular"]]),
        "track": "Exact production sampler; each test fold uses strictly earlier calibration residuals",
        "limits": [
            "Marginal intervals do not establish joint rival probability accuracy",
            "Discrete coverage may be conservative",
        ],
    }
    regular = report["regular_starters"]
    report["passed"] = bool(
        regular["coverage80"] >= 0.75
        and regular["coverage90"] >= 0.85
        and abs(regular["mean_sampling_bias"]) < 0.1
    )
    report["id"] = store.put("fusion_uncertainty", report)
    atomic_json(Path("artifacts/evaluations/fusion_uncertainty.json"), report)
    return report


def production_predictions(store, spec, rows, market, players, scenario=None):
    from .fusion_data import live_inputs

    validation = store.get("fusion_model", spec["fusion_model_id"])
    kind = spec["fusion_kind"]
    inputs = live_inputs(store, rows, market, players)
    p, c, s = candidate_predict(store, validation, inputs, kind)
    healthy = {k: v.copy() for k, v in inputs.items()}
    healthy["text"][:] = 0
    healthy["news"][:, :5] = 0
    healthy["news"][:, 0] = 1
    healthy["news"][:, 5:7] = 1
    healthy["news"][:, 11] = 0
    hp, hc, hs = candidate_predict(store, validation, healthy, kind)
    start = min(r["event"] for r in rows)
    weight = np.exp(-np.array([r["event"] - start for r in rows]) / 5)
    joint = weight[:, None] * p * c + (1 - weight[:, None]) * hp * hc
    probabilities = weight[:, None] * p + (1 - weight[:, None]) * hp
    conditional = np.divide(joint, np.maximum(probabilities, 1e-9))
    starts = weight * s + (1 - weight) * hs
    for i, row in enumerate(rows):
        row["availability"] = 1.0 if players[row["player_id"]].selectable else 0.0
        for change in (scenario or {}).get("players", []):
            if (
                change["player_id"] == row["player_id"]
                and change["from_event"] <= row["event"] <= change.get("through_event", 38)
                and "availability" in change
            ):
                # Explicit availability scenarios replace the news state, rather
                # than applying the same injury chance a second time.
                probabilities[i] = hp[i]
                conditional[i] = hc[i]
                starts[i] = hs[i]
                row["availability"] = change["availability"] if players[row["player_id"]].selectable else 0
    return probabilities, conditional, starts


def promote_fusion(store: Store):
    """Promote only after point, decision and shared-sampler checks pass."""
    spec = store.get("fusion_model")
    decisions = store.get("fusion_decisions")
    uncertainty = store.get("fusion_uncertainty")
    if (
        decisions["fusion_model_id"] != spec["id"]
        or uncertainty["fusion_model_id"] != spec["id"]
        or not decisions["promotable"]
        or not uncertainty["passed"]
    ):
        raise ValueError("Fusion forecast, legal policy and distribution promotion gates must all pass")
    kind = spec["selected"]
    if kind is None:
        raise ValueError("No fusion candidate passed forecast promotion")
    old = store.get("decision_model")
    if old.get("fusion_model_id") == spec["id"]:
        return old
    if spec["compared_decision_model_id"] != old["id"]:
        raise ValueError("Re-evaluate fusion against the current production model")
    arrays, metadata, _ = load_data(store, spec["feature_id"])
    groups = np.array([r["deadline_group"] for r in metadata])
    cal = np.flatnonzero(groups >= spec["production_fit_before"])
    subset = {k: v[cal] for k, v in arrays.items() if k in INPUTS}
    _, conditional, _ = candidate_predict(store, spec, subset, kind)
    pools = residual_pools(arrays, metadata, cal, conditional)
    rows = pl.read_parquet(store.artifact_path(spec["prediction_path"])).to_dicts()
    previous = {(r["season"], r["player_id"], r["fixture_id"]): r for r in old["predictions"]}
    original_metadata = {(r["season"], r["player_id"], r["fixture_id"]): r for r in metadata}
    predictions = []
    for row in rows:
        key = (row["season"], row["player_id"], row["fixture_id"])
        prior = previous.get(key, row | {"baseline": original_metadata[key]["baseline"]})
        predictions.append(
            prior
            | {
                "predicted": row[kind],
                "network": row["network"],
                "price_lag": row["price_lag"],
                "price_observed": True,
                "p_any": row[kind + "_p_any"],
                "p60": row.get(kind + "_p60"),
            }
        )
    y = np.array([r["actual"] for r in predictions])
    b = np.array([r["baseline"] for r in predictions])
    p = np.array([r["predicted"] for r in predictions])
    regular = np.array([r["prior_start"] >= 0.6 for r in predictions])
    body = {k: v for k, v in old.items() if k != "id"}
    from .models import reliability

    body.update(
        at=now(),
        protocol_id=store.put("fusion_protocol", spec["protocol"]),
        appearance=reliability(
            np.array([r["minutes"] > 0 for r in predictions]), np.array([r["p_any"] for r in predictions])
        ),
        by_position={
            str(position): {
                "baseline": score_metrics(y[mask], b[mask]),
                "candidate": score_metrics(y[mask], p[mask]),
            }
            for position in range(1, 5)
            if (mask := np.array([r["position"] == position for r in predictions])).any()
        },
        architecture="multi-source-" + kind,
        fusion_model_id=spec["id"],
        fusion_kind=kind,
        previous_decision_model_id=old["id"],
        feature_id=spec["feature_id"],
        predictions=predictions,
        baseline=score_metrics(y, b),
        candidate=score_metrics(y, p),
        regular_starters={
            "baseline": score_metrics(y[regular], b[regular]),
            "candidate": score_metrics(y[regular], p[regular]),
        },
        paired_rmse_difference_95ci=paired_bootstrap(predictions, "predicted"),
        residuals=pools,
        production_training_rows=spec["production_training_rows"],
        production_calibration_rows=spec["production_calibration_rows"],
        production_fit_before=spec["production_fit_before"],
        fusion_decisions_id=decisions["id"],
        fusion_uncertainty_id=uncertainty["id"],
        training_serve_transform="Shared fusion source_features, sequence, tree_matrix and standardized neural tensors",
        neural={
            "metrics": spec["candidates"]["network"]["metrics"],
            "paired_vs_gbm_95ci": paired_bootstrap(rows, "network", "tree"),
        },
    )
    minutes = arrays["minutes"][cal]
    bands = minute_band(minutes)
    body["minute_quantiles"] = {
        str(k): np.quantile(minutes[bands == k], np.linspace(0, 1, 101)).tolist() for k in range(4)
    }
    body["limits"] = (
        old.get("limits", [])
        + spec["limits"]
        + [
            "Current state is carried forward; injury effects mix toward a healthy counterfactual with a five-gameweek decay; future roles, ownership and non-PL workload are not observed",
            "Within-team residual dependence retains the established estimate and remains approximate",
        ]
    )
    body["id"] = store.put("decision_model", body)
    atomic_json(
        Path("artifacts/evaluations/decision_model.json"),
        {k: v for k, v in body.items() if k not in ("predictions", "residuals", "curves")},
    )
    return body


def freeze_fusion_artifacts(store: Store, spec=None):
    """Give saved candidates and input tensors immutable, content-derived paths."""
    import shutil

    spec = spec or store.get("fusion_model")
    body = {k: v for k, v in spec.items() if k != "id"}
    features = store.get("fusion_features", spec["feature_id"])
    feature = {k: v for k, v in features.items() if k != "id"}
    for field, suffix in (("inputs_path", ".npz"), ("metadata_path", ".parquet")):
        old = store.artifact_path(feature[field])
        sha = hashlib.sha256(old.read_bytes()).hexdigest()
        new = store.root / "features" / ("fusion-" + sha[:24] + suffix)
        if old != new and not new.exists():
            shutil.copy2(old, new)
        feature[field] = str(new)
        if field == "metadata_path":
            feature["metadata_hash"] = sha
    feature.setdefault(
        "lineage",
        {
            "context_archive": store.get("context_archive")["id"],
            "elite_cohort": store.get("elite_cohort")["id"],
            "workload": store.get("workload")["id"],
        },
    )
    feature["id"] = store.put("fusion_features", feature)
    body["feature_id"] = feature["id"]
    original = store.root / "models" / spec["directory"]
    fingerprint = digest(
        {
            "tree": spec["tree_spec"],
            "network": spec["network_spec"],
            "feature": feature["hash"],
            "protocol": spec["protocol"],
        }
    )
    frozen = store.root / "models" / ("fusion-" + fingerprint[:24])
    if original != frozen:
        frozen.mkdir(parents=True, exist_ok=True)
        for path in original.iterdir():
            destination = frozen / path.name
            if destination.exists() and destination.read_bytes() != path.read_bytes():
                raise ValueError("Immutable fusion artifact conflict")
            if not destination.exists():
                shutil.copy2(path, destination)
    body["directory"] = frozen.name
    body["prediction_path"] = str(frozen / "heldout.parquet")
    body["id"] = store.put("fusion_model", body)
    atomic_json(
        Path("artifacts/evaluations/fusion_model.json"),
        {k: v for k, v in body.items() if k not in ("tree_spec", "network_spec")},
    )
    atomic_json(Path("artifacts/evaluations/fusion_data_coverage.json"), feature)
    return body


def review_fusion(store: Store):
    spec = store.get("fusion_model")
    decisions = audit_fusion_decisions(store)
    result = {
        "at": now(),
        "fusion_model_id": spec["id"],
        "forecast_selected": spec["selected"],
        "decisions_id": decisions["id"],
        "promoted": False,
    }
    if spec["selected"] and decisions["promotable"]:
        uncertainty = validate_fusion_worlds(store)
        result["uncertainty_id"] = uncertainty["id"]
        if uncertainty["passed"]:
            production = promote_fusion(store)
            result.update(promoted=True, decision_model_id=production["id"])
        else:
            result["reason"] = "Shared-sampler uncertainty gate failed; previous production retained"
    else:
        result["reason"] = "Forecast or matched captain/policy gate failed; previous production retained"
    result["id"] = store.put("fusion_review", result)
    atomic_json(Path("artifacts/evaluations/fusion_review.json"), result)
    return result


def reference_forecasts(store, metadata):
    """Keep a fixed validated baseline architecture when re-fitting an active fusion model."""
    from .decision import NAMES, fit_components, predict_models

    active = store.get("decision_model")
    reference = active
    while reference.get("fusion_model_id"):
        reference = store.get("decision_model", reference["previous_decision_model_id"])
    output = {(r["season"], r["player_id"], r["fixture_id"]): r for r in reference["predictions"]}
    missing = {
        (r["season"], r["player_id"], r["fixture_id"])
        for r in metadata
        if r["deadline_group"] >= "2025/26-GW01"
    } - output.keys()
    if missing:
        features = store.get("decision_features")
        frame = pl.read_parquet(store.artifact_path(features["path"]))
        groups = sorted(frame["deadline_group"].unique().to_list())
        testing = [g for g in groups if g >= "2025/26-GW01"]
        for first in range(0, len(testing), 4):
            block = testing[first : first + 4]
            test = frame.filter(pl.col("deadline_group").is_in(block))
            records = test.to_dicts()
            if not any((r["season"], r["player_id"], r["fixture_id"]) in missing for r in records):
                continue
            cut = [g for g in groups if g < block[0]][-8]
            train = frame.filter(pl.col("deadline_group") < cut)
            cal = frame.filter((pl.col("deadline_group") >= cut) & (pl.col("deadline_group") < block[0]))
            fitted = fit_components(train, cal, False)
            mean, _, _ = predict_models(
                fitted, test.select(NAMES).to_numpy(), test["dc_probability"].to_list()
            )
            for i, row in enumerate(records):
                output[row["season"], row["player_id"], row["fixture_id"]] = row | {
                    "predicted": float(mean[i])
                }
    return active, reference, output


def refit_active_fusion(store: Store):
    """Weekly refit of the validated architecture; avoid rerunning architecture search."""
    import shutil
    import tempfile

    from .decision import build_dataset
    from .fusion_data import build_fusion_data

    active = store.get("decision_model")
    validated = store.get("fusion_model", active["fusion_model_id"])
    kind = active["fusion_kind"]
    build_dataset(store)
    build_fusion_data(store)
    arrays, meta, features = load_data(store)
    if features["hash"] == store.get("fusion_features", validated["feature_id"])["hash"]:
        return active
    groups = np.array([r["deadline_group"] for r in meta])
    cut = sorted(set(groups))[-4]
    train = np.flatnonzero(groups < cut)
    cal = np.flatnonzero(groups >= cut)
    candidate = {k: v for k, v in validated.items() if k != "id"}
    reference_id = validated.get("validation_reference_id", validated["id"])
    reference = store.get("fusion_model", reference_id)
    with tempfile.TemporaryDirectory(dir=store.root / "models", prefix="refit-") as temporary:
        directory = Path(temporary)
        for path in (store.root / "models" / validated["directory"]).iterdir():
            shutil.copy2(path, directory / path.name)
        if kind == "tree":
            matrix = tree_matrix(
                arrays["tab"],
                arrays["news"],
                arrays["market"],
                arrays["workload"],
                arrays["text"],
                arrays["sequence"],
            )
            fitted = fit_tree(arrays, matrix, train, cal)
            curves, offsets = calibrate(*tree_predict(fitted, matrix[cal]), arrays, cal)
            hashes = {}
            for key, model in fitted.items():
                path = directory / (key + ".txt")
                model.booster_.save_model(str(path))
                hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            candidate["tree_spec"] = {"curves": curves, "offsets": offsets, "hashes": hashes}
        else:
            import torch

            model, norm, training = fit_neural(arrays, train, cal, epochs=validated["protocol"]["max_epochs"])
            curves, offsets = calibrate(*predict_neural(model, norm, arrays, cal), arrays, cal)
            path = directory / "network.pt"
            torch.save(model.state_dict(), path)
            candidate["network_spec"] = {
                "curves": curves,
                "offsets": offsets,
                "normalization": norm,
                "training": training,
                "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        candidate.update(
            at=now(),
            directory=directory.name,
            feature_id=features["id"],
            production_fit_before=cut,
            production_training_rows=len(train),
            production_calibration_rows=len(cal),
            validation_reference_id=reference_id,
            validation_evaluated_at=reference["at"],
            validation_metrics_feature_id=reference["feature_id"],
        )
        candidate = freeze_fusion_artifacts(store, candidate)
    subset = {k: v[cal] for k, v in arrays.items() if k in INPUTS}
    _, conditional, _ = candidate_predict(store, candidate, subset, kind)
    body = {k: v for k, v in active.items() if k != "id"}
    body.update(
        at=now(),
        fusion_model_id=candidate["id"],
        feature_id=features["id"],
        residuals=residual_pools(arrays, meta, cal, conditional),
        previous_decision_model_id=active["id"],
        production_training_rows=len(train),
        production_calibration_rows=len(cal),
        production_fit_before=cut,
        validation_evaluated_at=reference["at"],
        validation_reference_id=reference_id,
    )
    minutes = arrays["minutes"][cal]
    bands = minute_band(minutes)
    body["minute_quantiles"] = {
        str(k): np.quantile(minutes[bands == k], np.linspace(0, 1, 101)).tolist() for k in range(4)
    }
    body["id"] = store.put("decision_model", body)
    atomic_json(
        Path("artifacts/evaluations/decision_model.json"),
        {k: v for k, v in body.items() if k not in ("predictions", "residuals", "curves")},
    )
    return body
