"""Purged multiweek value models and additional advanced-statistics ablations."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from .decision import NAMES, canonical_rows, player_key, score_metrics
from .features import number
from .fusion import load_data
from .fusion_data import tree_matrix
from .storage import Store, atomic_json, digest, now

ADVANCED = (
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "penalties_saved",
    "penalties_missed",
    "defensive_contribution",
    "goals_conceded",
)


def advanced_vector(past):
    values = []
    for field in ADVANCED:
        for window in (3, 12):
            rows = past[-window:]
            valid = [r for r in rows if r.get(field) not in (None, "")]
            exposure = sum(number(r, "minutes") / 90 for r in valid)
            values.append(sum(number(r, field) for r in valid) / (exposure + 2))
        values.append(sum(r.get(field) not in (None, "") for r in past[-12:]) / max(1, min(12, len(past))))
    return values


def advanced_history(store, metadata):
    rows, _, _ = canonical_rows(store)
    bygroup = defaultdict(list)
    history: dict[tuple, list[dict]] = defaultdict(list)
    output = {}
    keys = {(r["season"], r["player_id"], r["fixture_id"]) for r in metadata}
    for row in rows:
        bygroup[f"{row['season']}-GW{int(row['round']):02d}"].append(row)
    for _, current in sorted(bygroup.items()):
        for row in current:
            key = (row["season"], int(row["element"]), int(row["fixture"]))
            if key in keys:
                output[key] = advanced_vector(history[player_key(row)])
        for row in current:
            key = player_key(row)
            history[key].append(row)
            history[key] = history[key][-12:]
    return np.array(
        [output[r["season"], r["player_id"], r["fixture_id"]] for r in metadata], dtype=np.float32
    )


def train_horizon(store: Store, horizon=4):
    arrays, meta, features = load_data(store)
    source = store.get("fusion_model")
    pointrows = pl.read_parquet(store.artifact_path(source["prediction_path"])).to_dicts()
    pointmap: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in pointrows:
        for field in ("baseline", "tree", "network"):
            pointmap[row["season"], row["event"], row["player_id"]][field] += row[field]
    matrix = tree_matrix(
        arrays["tab"],
        arrays["news"],
        arrays["market"],
        arrays["workload"],
        arrays["text"],
        arrays["sequence"],
    )
    advanced = advanced_history(store, meta)
    whole = np.column_stack([matrix, advanced])
    weeks = defaultdict(set)
    outcomes: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(2))
    first: dict[tuple, int] = {}
    counts: dict[tuple, int] = defaultdict(int)
    for i, row in enumerate(meta):
        key = (row["season"], row["event"], row["player_id"])
        weeks[row["season"]].add(row["event"])
        outcomes[key] += [arrays["base_target"][i], arrays["target"][i]]
        first.setdefault(key, i)
        counts[key] += 1
    indices: Any = []
    labels: Any = []
    actual: Any = []
    records = []
    dc: Any = []
    for key, index in sorted(first.items()):
        season, event, player = key
        if not all(e in weeks[season] for e in range(event, event + horizon)):
            continue
        values = sum((outcomes[season, e, player] for e in range(event, event + horizon)), start=np.zeros(2))
        row = meta[index]
        start = float(arrays["tab"][index, NAMES.index("p_start")])
        cameo = float(arrays["tab"][index, NAMES.index("p_cameo")])
        long = float(arrays["tab"][index, NAMES.index("long_probability")])
        prior = np.array([max(0, 1 - start - cameo), cameo, start * (1 - long), start * long])
        prior /= prior.sum()
        expected_dc = float(np.dot(prior, arrays["dc_prior"][index]) * 2 * horizon)
        indices.append(index)
        labels.append(values[0])
        actual.append(values[1])
        dc.append(expected_dc)
        records.append(
            row
            | {
                "label_end": f"{season}-GW{event + horizon - 1:02d}",
                "actual": float(values[1]),
                "baseline": float(pointmap[key]["baseline"] * horizon) if key in pointmap else None,
                "nextweek_tree": float(pointmap[key]["tree"]) if key in pointmap else None,
                "nextweek_reference": float(pointmap[key]["baseline"]) if key in pointmap else None,
                "fixture_count": counts[key],
            }
        )
    indices = np.array(indices)
    x0 = matrix[indices]
    x1 = whole[indices]
    labels = np.array(labels)
    actual = np.array(actual)
    dc = np.array(dc)
    groups = np.array([r["deadline_group"] for r in records])
    ends = np.array([r["label_end"] for r in records])
    unique = sorted(set(groups))
    testing = [g for g in unique if g >= "2025/26-GW01"]
    predictions = []
    folds = []
    params: dict[str, Any] = {
        "n_estimators": 240,
        "num_leaves": 20,
        "learning_rate": 0.035,
        "min_child_samples": 100,
        "reg_lambda": 20,
        "random_state": 73,
        "n_jobs": 1,
        "verbosity": -1,
    }
    for first_test in range(0, len(testing), 8):
        block = testing[first_test : first_test + 8]
        available = sorted(set(groups[ends < block[0]]))
        calgroups = available[-4:]
        cut = calgroups[0]
        train = np.flatnonzero(ends < cut)
        cal = np.flatnonzero(np.isin(groups, calgroups) & (ends < block[0]))
        test = np.flatnonzero(np.isin(groups, block))
        predictions_by_kind = {}
        for kind, x in (("context", x0), ("advanced", x1)):
            model = lgb.LGBMRegressor(**params).fit(x[train], labels[train])
            raw = model.predict(x[cal]) + dc[cal]
            offset = float((actual[cal] - raw).sum() / (len(cal) + 100))
            predictions_by_kind[kind] = model.predict(x[test]) + dc[test] + offset
        for j, sample_index in enumerate(test):
            index = int(sample_index)
            if records[index]["baseline"] is None:
                continue
            predictions.append(
                records[index] | {kind: float(values[j]) for kind, values in predictions_by_kind.items()}
            )
        folds.append(
            {
                "test": block,
                "train_rows": len(train),
                "calibration_rows": len(cal),
                "purged_training_label_end_before": cut,
                "calibration_label_end_before": block[0],
            }
        )
        print("Purged horizon validation", block[-1], flush=True)
    y = np.array([r["actual"] for r in predictions])
    reference = np.array([r["baseline"] for r in predictions])
    stats = {}
    for kind in ("context", "advanced"):
        values = np.array([r[kind] for r in predictions])
        stats[kind] = {
            "metrics": score_metrics(y, values),
            "paired_vs_nextweek_repeated_95ci": block_interval(predictions, kind),
        }
    extra = block_interval(predictions, "advanced", "context")
    selected = "advanced" if extra[1] < 0 else "context"
    report = {
        "at": now(),
        "horizon": horizon,
        "feature_id": features["id"],
        "fusion_model_id": source["id"],
        "parameters": params,
        "protocol": "Expanding eight-deadline test blocks; training and calibration target windows end strictly before the next partition starts; no random split",
        "folds": folds,
        "baseline": score_metrics(y, reference),
        "candidates": stats,
        "advanced_vs_context_95ci": extra,
        "selected": selected,
        "advanced_fields": ADVANCED,
        "limits": [
            "Four-week labels are realized outcomes, never inference features",
            "Only weeks with complete four-week outcomes are scored; current season evaluates starting GW1–GW5",
            "Historical archive/fixture registration limitations remain; this is statistical replay",
            "Defensive-contribution expectation uses a declared current-profile prior, not invented historic action labels",
            "Repeating a one-week reference is a horizon baseline, not a claim to reproduce the full multiweek optimizer",
        ],
    }
    # Final production-value model follows the same purged contract.
    calgroups = unique[-4:]
    cut = calgroups[0]
    train = np.flatnonzero(ends < cut)
    cal = np.flatnonzero(np.isin(groups, calgroups))
    x = x1 if selected == "advanced" else x0
    model = lgb.LGBMRegressor(**params).fit(x[train], labels[train])
    offset = float((actual[cal] - model.predict(x[cal]) - dc[cal]).sum() / (len(cal) + 100))
    directory = (
        store.root
        / "models"
        / (
            "horizon-"
            + digest(
                {"feature": features["hash"], "parameters": params, "horizon": horizon, "selected": selected}
            )[:20]
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "value.txt"
    model.booster_.save_model(str(path))
    predpath = directory / "heldout.parquet"
    pl.DataFrame(predictions, infer_schema_length=None).write_parquet(predpath)
    report.update(
        directory=directory.name,
        offset=offset,
        prediction_path=str(predpath),
        model_hash=__import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        production_fit_label_end_before=cut,
        production_training_rows=len(train),
        production_calibration_rows=len(cal),
    )
    report["id"] = store.put("horizon_model", report)
    atomic_json(Path("artifacts/evaluations/horizon_model.json"), report)
    return report


def block_interval(rows, field, baseline="baseline", length=4):
    groups = sorted({r["deadline_group"] for r in rows})
    summary: Any = []
    for group in groups:
        part = [r for r in rows if r["deadline_group"] == group]
        summary.append(
            [
                sum((r[field] - r["actual"]) ** 2 for r in part),
                sum((r[baseline] - r["actual"]) ** 2 for r in part),
                len(part),
            ]
        )
    summary = np.array(summary)
    rng = np.random.default_rng(73)
    n = len(groups)
    starts = rng.integers(0, n, (1000, (n + length - 1) // length))
    indices = ((starts[:, :, None] + np.arange(length)) % n).reshape(1000, -1)[:, :n]
    total = summary[indices].sum(1)
    delta = np.sqrt(total[:, 0] / total[:, 2]) - np.sqrt(total[:, 1] / total[:, 2])
    return np.quantile(delta, [0.025, 0.975]).tolist()


def live_values(store: Store):
    import hashlib

    from .decision import future_inputs
    from .fusion import candidate_predict
    from .fusion_data import live_inputs
    from .ingestion import load_market

    model = store.get("horizon_model")
    fusion = store.get("fusion_model", model["fusion_model_id"])
    players, market = load_market(store)
    event = next(e["id"] for e in market["events"] if e.get("is_next"))
    rows, _ = future_inputs(store, market, players, event, event)
    kickoff = {f["id"]: f.get("kickoff_time") or "" for f in market["fixtures"]}
    unique: dict[int, dict] = {}
    for row in sorted(rows, key=lambda r: kickoff[r["fixture_id"]]):
        unique.setdefault(row["player_id"], row)
    rows = list(unique.values())
    inputs = live_inputs(store, rows, market, players)
    matrix = tree_matrix(
        inputs["tab"],
        inputs["news"],
        inputs["market"],
        inputs["workload"],
        inputs["text"],
        inputs["sequence"],
    )
    if model["selected"] == "advanced":
        original, _, _ = canonical_rows(store)
        history = defaultdict(list)
        for row in original:
            history[player_key(row)].append(row)
        matrix = np.column_stack(
            [matrix, np.array([advanced_vector(history["code", players[r["player_id"]].code]) for r in rows])]
        )
    path = store.root / "models" / model["directory"] / "value.txt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != model["model_hash"]:
        raise ValueError("Horizon model integrity failed")
    raw = np.asarray(lgb.Booster(model_file=str(path)).predict(matrix, num_threads=1)) + model["offset"]
    probability, conditional, _ = candidate_predict(store, fusion, inputs, "tree")
    output = {}
    for i, row in enumerate(rows):
        x = row["x"]
        start = x[NAMES.index("p_start")]
        cameo = x[NAMES.index("p_cameo")]
        long = x[NAMES.index("long_probability")]
        prior = np.array([max(0, 1 - start - cameo), cameo, start * (1 - long), start * long])
        prior /= prior.sum()
        expected_dc = float(np.dot(prior, row["dc_probability"]) * 2 * model["horizon"])
        output[row["player_id"]] = {
            "fourweek": float(raw[i] + expected_dc),
            "short": float(np.dot(probability[i], conditional[i] + 2 * np.array(row["dc_probability"]))),
        }
    return output, players, market, model
