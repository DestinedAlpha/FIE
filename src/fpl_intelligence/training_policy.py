"""Predeclared, chronological training-age and calibration policy comparisons.

Player form inputs are held fixed: this experiment changes training observations,
not the meaning of a player's recent form. No current-season label selects policy.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .decision import PARAMETERS, minute_band
from .models import curve_apply, curve_fit, save_booster_atomic
from .role_minutes import PARAMS as ROLE_PARAMS
from .search_control import check_cancelled
from .storage import atomic_json, digest, now

VERSION = "training-age-v1"
POLICIES: dict[str, dict[str, Any]] = {
    "B_expanding_cal4": {"calibration": 4},
    "C_half26": {"half_life": 26},
    "C_half52": {"half_life": 52},
    "C_half104": {"half_life": 104},
    "D_window52": {"window": 52},
    "D_window78": {"window": 78},
    "E_season4": {"season_weight": 4},
    "G_crossfit8": {"crossfit": True},
}
DERIVED = ("F_adaptive", "H_recent25", "H_recent50")


def protocol():
    return {
        "version": VERSION,
        "policies": POLICIES,
        "derived": DERIVED,
        "reference": "Exact accepted role-v3 / conditional-points immediate composition; future expert unchanged in this experiment",
        "selection": "2024/25 GW1-19 only; minimum relevant-weighted squared points error plus 0.5 multiclass participation Brier",
        "acceptance": "2024/25 GW20-38 and independent full 2025/26; current completed 2026/27 is diagnostic only",
        "folds": "Expanding eight-deadline test blocks; labels from a target deadline, including both DGW fixtures, never enter its fit or calibration",
        "point_gate": "Both periods: upper 95% deadline-block relevant-weighted MSE delta below zero; all/regular/premium/position RMSE at most 1% worse; participation Brier no more than 0.002 worse",
        "captain_gate": "Whole-deadline paired captain difference interval wholly below zero is evidence of harm and rejects. An uncertain one-point aggregate change alone does not establish harm or superiority. Full legal-account policy and distribution gates are mandatory before any activation.",
        "promotion": "Only the independently selected overall winner is eligible; do not select a holdout winner after looking at acceptance results",
        "adaptive": "Disagreement-triggered mixture of frozen long and rolling-52 experts: min(0.75, total variation of participation + 0.25 min(1, absolute mean-point disagreement/3))",
        "hybrid": "H uses fixed 25% or 50% rolling expert. E upweights current-season fitting labels fourfold while preserving earlier historical rows. No player-specific overrides.",
        "crossfit": "Eight earlier calibration deadlines split into two blocks. Each calibration prediction is made by a model fitted before that block; final weights fit all earlier eligible labels. Calibrators never see in-sample predictions.",
        "limits": "Previously studied archives; this is reproducible historical reconstruction, not a blinded prospective performance claim.",
    }


def training_indices(groups, before, *, calibration=8, window=None, half_life=None, season_weight=1):
    unique = sorted(set(groups))
    earlier = [g for g in unique if g < before]
    if len(earlier) <= calibration:
        raise ValueError("Insufficient earlier training/calibration deadlines")
    cut = earlier[-calibration]
    training = np.flatnonzero(groups < cut)
    if window is not None:
        start = earlier[max(0, len(earlier) - calibration - window)]
        training = training[groups[training] >= start]
    cal = np.flatnonzero((groups >= cut) & (groups < before))
    ranks = {g: i for i, g in enumerate(unique)}
    ages = np.array([ranks[cut] - ranks[g] for g in groups[training]])
    weight = np.ones(len(training)) if half_life is None else np.exp2(-ages / half_life)
    weight *= np.array([season_weight if g[:7] == before[:7] else 1 for g in groups[training]])
    weight /= weight.mean()
    return training, cal, weight, cut


def fit_raw(arrays, role, indices, weight):
    check_cancelled()
    x = arrays["tab"][indices]
    minutes = arrays["minutes"][indices]
    band = minute_band(minutes)
    present = minutes > 0
    known = arrays["start_mask"][indices]
    point = lgb.LGBMRegressor(**PARAMETERS).fit(
        np.column_stack([x[present], band[present]]),
        arrays["base_target"][indices][present],
        sample_weight=weight[present],
    )
    starter = lgb.LGBMClassifier(**PARAMETERS).fit(
        x[known], arrays["start"][indices][known], sample_weight=weight[known]
    )
    participant = lgb.LGBMClassifier(**ROLE_PARAMS).fit(role[indices], band, sample_weight=weight)
    if participant.classes_.tolist() != [0, 1, 2, 3]:
        raise ValueError("Training lost a participation class")
    return {"points": point, "starter": starter, "role": participant}


def raw_predictions(models, arrays, role, indices):
    def predict(key, x, prob=False):
        model = models[key]
        if hasattr(model, "predict_proba") and prob:
            return np.asarray(model.predict_proba(x))
        return np.asarray(model.predict(x, num_threads=1))

    p = predict("role", role[indices], True)
    start = predict("starter", arrays["tab"][indices], True)
    if start.ndim == 2:
        start = start[:, 1]
    c = np.column_stack(
        [np.zeros(len(indices))]
        + [
            predict("points", np.column_stack([arrays["tab"][indices], np.full(len(indices), k)]))
            for k in (1, 2, 3)
        ]
    )
    return p, c, start


def calibrators(raw, arrays, cal):
    p, _, starts = raw
    bands = minute_band(arrays["minutes"][cal])
    known = arrays["start_mask"][cal]
    return {
        **{str(k): curve_fit(p[:, k], (bands == k).astype(int)) for k in range(4)},
        "start": curve_fit(starts[known], arrays["start"][cal][known]),
    }


def calibrated(raw, curves, news):
    p, c, starts = raw
    p = np.column_stack([curve_apply(p[:, k], curves[str(k)]) for k in range(4)])
    p /= np.maximum(p.sum(1, keepdims=True), 1e-9)
    p[(news[:, 5] > 0) & (news[:, 6] == 0)] = [1, 0, 0, 0]
    return p, c, np.minimum(curve_apply(starts, curves["start"]), 1 - p[:, 0])


def mix(left, right, weight):
    lp, lc, ls = left
    rp, rc, rs = right
    w = np.broadcast_to(np.asarray(weight), (len(lp),))[:, None]
    p = (1 - w) * lp + w * rp
    c = np.divide((1 - w) * lp * lc + w * rp * rc, np.maximum(p, 1e-12))
    return p, c, (1 - w[:, 0]) * ls + w[:, 0] * rs


def fitted_fold(directory, name, arrays, role, groups, before, test, config, reference_models):
    folder = directory / (before.replace("/", "-") + "-" + name)
    folder.mkdir(exist_ok=True)
    manifest = folder / "model.json"
    if manifest.exists():
        spec = json.loads(manifest.read_text())
        models = {}
        for key, item in spec["models"].items():
            path = folder / item["file"]
            models[key] = cached_booster(path, item["hash"])
        return calibrated(
            raw_predictions(models, arrays, role, test), spec["curves"], arrays["news"][test]
        ), spec
    if config.get("crossfit"):
        _, cal, _, cut = training_indices(groups, before)
        cal_groups = sorted(set(groups[cal]))
        pivot = cal_groups[4]
        first = cal[groups[cal] < pivot]
        second = cal[groups[cal] >= pivot]
        # The cal4 expert is fitted before the second calibration block.
        bfolder = directory / (before.replace("/", "-") + "-B_expanding_cal4")
        bspec = json.loads((bfolder / "model.json").read_text())
        bmodels = {k: cached_booster(bfolder / v["file"], v["hash"]) for k, v in bspec["models"].items()}
        raws = [
            raw_predictions(reference_models, arrays, role, first),
            raw_predictions(bmodels, arrays, role, second),
        ]
        raw_cal = tuple(np.concatenate([v[k] for v in raws]) for k in range(3))
        cal = np.concatenate([first, second])
        indices = np.flatnonzero(groups < before)
        weight = np.ones(len(indices))
        curves = calibrators(raw_cal, arrays, cal)
    else:
        indices, cal, weight, cut = training_indices(groups, before, **config)
    models = fit_raw(arrays, role, indices, weight)
    if not config.get("crossfit"):
        curves = calibrators(raw_predictions(models, arrays, role, cal), arrays, cal)
    files = {}
    for key, model in models.items():
        path = folder / (key + ".txt")
        save_booster_atomic(model.booster_, path)
        files[key] = {"file": path.name, "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    spec = {
        "models": files,
        "curves": curves,
        "fit_min": str(min(groups[indices])),
        "fit_max": str(max(groups[indices])),
        "calibration_min": str(min(groups[cal])),
        "calibration_max": str(max(groups[cal])),
        "test_first": before,
        "fit_rows": len(indices),
        "calibration_rows": len(cal),
        "effective_fit_rows": float(weight.sum() ** 2 / (weight @ weight)),
        "crossfitted": bool(config.get("crossfit")),
        "crossfit_blocks": [str(cut), str(pivot)] if config.get("crossfit") else None,
    }
    if spec["fit_max"] >= before or spec["calibration_max"] >= before:
        raise ValueError("Training-policy target leakage")
    if not config.get("crossfit") and spec["fit_max"] >= spec["calibration_min"]:
        raise ValueError("Calibration overlaps fitting labels")
    atomic_json(manifest, spec)
    print("Training policy", before, name, len(indices), flush=True)
    return calibrated(raw_predictions(models, arrays, role, test), curves, arrays["news"][test]), spec


def summary(frame, name):
    from .forecast_population import binary_metrics
    from .immediate_points import metrics

    values = frame.to_dicts()
    points = metrics(values, name)
    y = frame["actual"].to_numpy()
    pred = frame[name].to_numpy()
    relevant = (frame["prior_start"].to_numpy() >= 0.6) | (frame["price_lag"].to_numpy() >= 80)
    points.update(
        mae=float(np.abs(pred - y).mean()),
        weighted_mse=float(np.average((pred - y) ** 2, weights=1 + 4 * relevant)),
    )
    p = np.column_stack([frame[name + f"_p{k}"].to_numpy() for k in range(4)])
    band = minute_band(frame["minutes"].to_numpy())
    points["participation_brier"] = float(np.mean(np.sum((p - np.eye(4)[band]) ** 2, axis=1)))
    points["appearance"] = binary_metrics(band > 0, 1 - p[:, 0])
    points["sixty"] = binary_metrics(frame["minutes"].to_numpy() >= 60, p[:, 2:].sum(1))
    points["positions"] = {
        str(pos): {"rows": int(mask.sum()), "rmse": float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2)))}
        for pos in (1, 2, 3, 4)
        if (mask := frame["position"].to_numpy() == pos).any()
    }
    return points


def paired_intervals(frame, name):
    from .availability import block_interval

    groups = frame["deadline_group"].to_list()
    y = frame["actual"].to_numpy()
    p = frame[name].to_numpy()
    ref = frame["A_reference"].to_numpy()
    relevant = (frame["prior_start"].to_numpy() >= 0.6) | (frame["price_lag"].to_numpy() >= 80)
    weights = 1 + 4 * relevant
    by = defaultdict(list)
    events: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for i, r in enumerate(frame.to_dicts()):
        by[groups[i]].append(i)
        key = r["player_id"]
        v = events[groups[i]].setdefault(key, [0.0, 0.0, 0.0])
        v[0] += r[name]
        v[1] += r["A_reference"]
        v[2] += r["actual"]
    delta = []
    capt = []
    for group, idx in sorted(by.items()):
        delta.append(
            float(np.average((p[idx] - y[idx]) ** 2 - (ref[idx] - y[idx]) ** 2, weights=weights[idx]))
        )
        players = events[group]

        def winner(index):
            return max(players, key=lambda k: (players[k][index], -k))

        capt.append(players[winner(0)][2] - players[winner(1)][2])
    dates = sorted(by)
    return {
        "weighted_mse_delta_95ci": block_interval(delta, dates),
        "captain_delta_95ci": block_interval(capt, dates),
        "captain_mean_delta": float(np.mean(capt)),
    }


def run(store):
    from .fusion import load_data
    from .role_minutes import historical_inputs

    active = store.get("decision_model")
    role_spec = store.get("role_minutes_model", active["role_minutes_model_id"])
    validation = store.get("role_minutes_validation", role_spec["validation_id"])
    arrays, meta, features = load_data(store, role_spec["feature_id"])
    role = historical_inputs(store, arrays, meta, features, validation["context_archive_id"])
    groups = np.array([r["deadline_group"] for r in meta])
    lookup = {(r["season"], r["event"], r["fixture_id"], r["player_id"]): i for i, r in enumerate(meta)}
    population = store.get("population_diagnostics")
    source = store.artifact_path(population["directory"] + "/protocol.json").parent
    contract = {
        "protocol": protocol(),
        "features": features["id"],
        "role_validation": validation["id"],
        "population_contract": population["contract"],
        "parameters": PARAMETERS,
        "role_parameters": ROLE_PARAMS,
    }
    directory = store.root / "models" / ("training-policy-" + digest(contract)[:24])
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "protocol.json", contract)
    frames = []
    folds = []
    with threadpool_limits(limits=1):
        for saved in population["folds"]:
            before = saved["fold"]["test"][0]
            checkpoint = directory / (before.replace("/", "-") + ".parquet")
            receipt = checkpoint.with_suffix(".json")
            if checkpoint.exists() and receipt.exists():
                old = json.loads(receipt.read_text())
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != old["hash"]:
                    raise ValueError("Policy predictions checksum mismatch")
                frames.append(pl.read_parquet(checkpoint))
                folds.append(old)
                continue
            original = source / (before.replace("/", "-") + ".parquet")
            if hashlib.sha256(original.read_bytes()).hexdigest() != saved["prediction_hash"]:
                raise ValueError("Accepted OOF reference checksum mismatch")
            base = pl.read_parquet(original)
            records = base.to_dicts()
            test = np.array(
                [lookup[r["season"], r["event"], r["fixture_id"], r["player_id"]] for r in records]
            )
            reference = (
                np.column_stack([base[f"p{k}"].to_numpy() for k in range(4)]),
                np.column_stack([base[f"c{k}"].to_numpy() for k in range(4)]),
                base["p_start"].to_numpy(),
            )
            outputs = {"A_reference": reference}
            models = {}
            for key in ("points", "starter", "role"):
                item = saved["models"][key]
                path = source / item["file"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != item["hash"]:
                    raise ValueError("Reference OOF weights changed")
                models[key] = cached_booster(path, item["hash"])
            evidence = {}
            for name, config in POLICIES.items():
                outputs[name], evidence[name] = fitted_fold(
                    directory, name, arrays, role, groups, before, test, config, models
                )
            recent = outputs["D_window52"]
            reference = outputs["A_reference"]
            rm = np.sum(reference[0] * (reference[1] + 2 * arrays["dc_prior"][test]), 1)
            sm = np.sum(recent[0] * (recent[1] + 2 * arrays["dc_prior"][test]), 1)
            drift = np.minimum(
                0.75, np.abs(reference[0] - recent[0]).sum(1) / 2 + 0.25 * np.minimum(1, np.abs(rm - sm) / 3)
            )
            outputs.update(
                F_adaptive=mix(reference, recent, drift),
                H_recent25=mix(reference, recent, 0.25),
                H_recent50=mix(reference, recent, 0.5),
            )
            columns: dict[str, Any] = {
                k: base[k]
                for k in (
                    "season",
                    "event",
                    "deadline_group",
                    "player_id",
                    "fixture_id",
                    "position",
                    "prior_start",
                    "price_lag",
                    "minutes",
                    "actual",
                )
            }
            for name, (p, c, starts) in outputs.items():
                columns[name] = np.sum(p * (c + 2 * arrays["dc_prior"][test]), 1)
                for k in range(4):
                    columns[name + f"_p{k}"] = p[:, k]
                    columns[name + f"_c{k}"] = c[:, k]
                columns[name + "_start"] = starts
            frame = pl.DataFrame(columns)
            if np.max(np.abs(frame["A_reference"].to_numpy() - base["reference"].to_numpy())) > 1e-6:
                raise ValueError("Accepted composition differs")
            frame.write_parquet(checkpoint)
            old = {
                "test": saved["fold"]["test"],
                "models": evidence,
                "hash": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            atomic_json(receipt, old)
            folds.append(old)
            frames.append(frame)
            print("Training policies evaluated", before, flush=True)
    frame = pl.concat(frames)
    names = ["A_reference", *POLICIES, *DERIVED]
    select = frame.filter((pl.col("season") == "2024/25") & (pl.col("event") <= 19))
    selection = {name: summary(select, name) for name in names}
    selected = min(
        names, key=lambda n: selection[n]["weighted_mse"] + 0.5 * selection[n]["participation_brier"]
    )
    periods = {}
    for label, part in (
        ("acceptance_2024", frame.filter((pl.col("season") == "2024/25") & (pl.col("event") > 19))),
        ("replication_2025", frame.filter(pl.col("season") == "2025/26")),
        ("current_diagnostic", frame.filter(pl.col("season") == "2026/27")),
    ):
        if not len(part):
            continue
        values = {name: summary(part, name) for name in names}
        intervals = paired_intervals(part, selected)
        ref = values["A_reference"]
        can = values[selected]
        passed = (
            selected != "A_reference"
            and intervals["weighted_mse_delta_95ci"][1] < 0
            and all(can[k] <= 1.01 * ref[k] for k in ("rmse", "regular_rmse", "premium_rmse"))
            and all(
                can["positions"][p]["rmse"] <= 1.01 * ref["positions"][p]["rmse"] for p in ref["positions"]
            )
            and can["participation_brier"] <= ref["participation_brier"] + 0.002
            and intervals["captain_delta_95ci"][1] >= 0
        )
        periods[label] = {
            "rows": len(part),
            "models": values,
            "selected_intervals": intervals,
            "forecast_gate_passed": bool(passed),
        }
    result = {
        "at": now(),
        "version": VERSION,
        "protocol": contract,
        "selected": selected,
        "selection": selection,
        "periods": periods,
        "folds": folds,
        "directory": str(directory),
        "forecast_gate_passed": all(
            periods[k]["forecast_gate_passed"] for k in ("acceptance_2024", "replication_2025")
        ),
        "promoted": False,
        "production_model_unchanged": store.get("decision_model")["id"] == active["id"],
    }
    result["id"] = store.put("training_policy_validation", result)
    atomic_json(Path("artifacts/evaluations/training_policy_0_14_4.json"), result)
    return result


def cached_booster(path, checksum):
    import gzip

    raw = (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
    )
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("Training-policy cache weight checksum mismatch")
    return lgb.Booster(model_str=raw.decode())


def compress_checkpoints(directory):
    import gzip
    import os

    for path in directory.rglob("*.txt"):
        zipped = path.with_suffix(".txt.gz")
        raw = path.read_bytes()
        body = gzip.compress(raw, compresslevel=6, mtime=0)
        temporary = zipped.with_suffix(".gz.tmp")
        temporary.write_bytes(body)
        os.replace(temporary, zipped)
        if hashlib.sha256(gzip.decompress(zipped.read_bytes())).digest() != hashlib.sha256(raw).digest():
            raise ValueError("Research compression changed weights")
        path.unlink()


def confirm_weekly(store):
    """Confirm the frozen selector using the actual weekly refit cadence."""
    from .fusion import load_data
    from .role_minutes import historical_inputs

    base = store.get("training_policy_validation")
    selected = base["selected"]
    if selected not in (*POLICIES, *DERIVED):
        raise ValueError("The incumbent won selection; no challenger requires weekly confirmation")
    active = store.get("decision_model")
    role_spec = store.get("role_minutes_model", active["role_minutes_model_id"])
    validation = store.get("role_minutes_validation", role_spec["validation_id"])
    arrays, meta, features = load_data(store, role_spec["feature_id"])
    role = historical_inputs(store, arrays, meta, features, validation["context_archive_id"])
    groups = np.array([r["deadline_group"] for r in meta])
    directory = store.artifact_path(base["directory"] + "/protocol.json").parent
    plan = {
        "at": now(),
        "screening_result_id": base["id"],
        "selected": selected,
        "selection": "Frozen 2024/25 GW1-19 selector from eight-deadline screening; no new parameter selection",
        "evaluation": "Every deadline in 2024/25 GW20-38 and all 2025/26; both models refit before every deadline with exact independent calibration",
        "reason": "Eight-deadline frozen folds isolate training-age effects, but production refreshes after each finalized deadline. The actual cadence requires separate confirmation.",
        "gate": base["protocol"]["protocol"]["point_gate"],
        "captain_gate": base["protocol"]["protocol"]["captain_gate"],
        "production_models": "No activation before distribution and full legal-account gates",
    }
    atomic_json(directory / "weekly-confirmation-protocol.json", plan)
    frames = []
    folds = []
    with threadpool_limits(limits=1):
        for before in sorted(set(groups)):
            if not ("2024/25-GW20" <= before <= "2024/25-GW38" or before.startswith("2025/26")):
                continue
            checkpoint = directory / (before.replace("/", "-") + "-weekly.parquet")
            receipt = checkpoint.with_suffix(".json")
            if checkpoint.exists() and receipt.exists():
                saved = json.loads(receipt.read_text())
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != saved["hash"]:
                    raise ValueError("Weekly confirmation predictions checksum mismatch")
                frames.append(pl.read_parquet(checkpoint))
                folds.append(saved)
                continue
            test = np.flatnonzero(groups == before)
            reference, rs = fitted_fold(directory, "A_weekly", arrays, role, groups, before, test, {}, None)
            config = POLICIES.get(selected, {})
            teachers = None
            if config.get("crossfit"):
                fitted_fold(
                    directory,
                    "B_expanding_cal4",
                    arrays,
                    role,
                    groups,
                    before,
                    test,
                    POLICIES["B_expanding_cal4"],
                    None,
                )
                folder = directory / (before.replace("/", "-") + "-A_weekly")
                teachers = {k: cached_booster(folder / v["file"], v["hash"]) for k, v in rs["models"].items()}
            if selected in DERIVED:
                recent, cs = fitted_fold(
                    directory, "D_window52", arrays, role, groups, before, test, POLICIES["D_window52"], None
                )
                drift = np.minimum(
                    0.75,
                    0.5 * np.abs(reference[0] - recent[0]).sum(1)
                    + 0.25
                    * np.minimum(
                        1, np.abs(np.sum(reference[0] * reference[1] - recent[0] * recent[1], 1)) / 3
                    ),
                )
                weight = drift if selected == "F_adaptive" else 0.25 if selected == "H_recent25" else 0.5
                candidate = mix(reference, recent, weight)
            else:
                candidate, cs = fitted_fold(
                    directory, selected, arrays, role, groups, before, test, config, teachers
                )
            rows = []
            for j, i in enumerate(test):
                row = {
                    k: meta[i][k]
                    for k in (
                        "season",
                        "event",
                        "deadline_group",
                        "player_id",
                        "fixture_id",
                        "position",
                        "prior_start",
                        "price_lag",
                    )
                }
                row.update(actual=float(arrays["target"][i]), minutes=float(arrays["minutes"][i]))
                for name, (p, c, starts) in (("A_reference", reference), (selected, candidate)):
                    row[name] = float(np.sum(p[j] * (c[j] + 2 * arrays["dc_prior"][i])))
                    row.update({name + f"_p{k}": float(p[j, k]) for k in range(4)})
                    row.update({name + f"_c{k}": float(c[j, k]) for k in range(4)})
                    row[name + "_start"] = float(starts[j])
                rows.append(row)
            frame = pl.DataFrame(rows)
            frame.write_parquet(checkpoint)
            saved = {
                "before": before,
                "reference": rs,
                "candidate": cs,
                "hash": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            atomic_json(receipt, saved)
            frames.append(frame)
            folds.append(saved)
            compress_checkpoints(directory)
            print("Weekly policy confirmation", before, flush=True)
    frame = pl.concat(frames)
    periods = {}
    for name, part in (
        ("acceptance_2024", frame.filter(pl.col("season") == "2024/25")),
        ("replication_2025", frame.filter(pl.col("season") == "2025/26")),
    ):
        ref, can = summary(part, "A_reference"), summary(part, selected)
        intervals = paired_intervals(part, selected)
        passed = (
            intervals["weighted_mse_delta_95ci"][1] < 0
            and all(can[k] <= 1.01 * ref[k] for k in ("rmse", "regular_rmse", "premium_rmse"))
            and all(
                can["positions"][p]["rmse"] <= 1.01 * ref["positions"][p]["rmse"] for p in ref["positions"]
            )
            and can["participation_brier"] <= ref["participation_brier"] + 0.002
            and intervals["captain_delta_95ci"][1] >= 0
        )
        periods[name] = {
            "reference": ref,
            "candidate": can,
            "intervals": intervals,
            "forecast_gate_passed": bool(passed),
            "rows": len(part),
        }
    result = {
        "at": now(),
        "protocol": plan,
        "selected": selected,
        "periods": periods,
        "folds": folds,
        "directory": str(directory),
        "forecast_gate_passed": all(p["forecast_gate_passed"] for p in periods.values()),
        "promoted": False,
    }
    result["id"] = store.put("training_policy_weekly", result)
    atomic_json(Path("artifacts/evaluations/training_policy_weekly_0_14_4.json"), result)
    return result
