"""Purged future-expert calibration experiment; never activates research weights.

The source deadline is the unit of fitting. Long-horizon target labels are
eligible only after their finalization time, including within teacher folds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .decision import NAMES, minute_band, predict_models
from .fixture_horizon import FEATURES, PARAMS, build_data, fit_models, partitions, predict
from .horizon_refinement import differences, distributions, loss
from .horizon_trial import reference
from .models import curve_fit, save_booster_atomic
from .search_control import check_cancelled
from .storage import atomic_json, digest, now
from .training_policy import cached_booster, compress_checkpoints


def raw(bundle, frame):
    x = frame.select(FEATURES).to_numpy()

    def probabilities(name):
        model = bundle[name]
        return np.asarray(
            model.predict(x, num_threads=1) if isinstance(model, lgb.Booster) else model.predict_proba(x)
        )

    return probabilities("minutes"), probabilities("starter")


def crossfit(frame, before, a, b):
    _, cal, cut = partitions(frame, before)
    _, _, pivot = partitions(frame, before, calibration_groups=4)
    first = cal.filter(pl.col("source_group") < pivot)
    second = cal.filter(pl.col("source_group") >= pivot)
    # Each teacher's latest fitting label is before its predicted source block.
    if a["fit_max_label"] >= cut or b["fit_max_label"] >= pivot:
        raise ValueError("Future crossfit teacher label leakage")
    cal = pl.concat([first, second])
    predictions = [raw(a, first), raw(b, second)]
    p, start = (np.concatenate([r[k] for r in predictions]) for k in (0, 1))
    if start.ndim == 2:
        start = start[:, 1]
    bands = minute_band(cal["minutes_target"].to_numpy())
    known = cal["start_target"].is_not_null().to_numpy()
    curves = {str(k): curve_fit(p[:, k], bands == k) for k in range(4)}
    curves["start"] = curve_fit(start[known], cal["start_target"].fill_null(0).to_numpy()[known] > 0)
    clock = frame.filter(pl.col("source_group") == before)["source_deadline"][0]
    train = frame.filter(
        (pl.col("source_group") < before) & (pl.col("label_end") < before) & (pl.col("label_known") < clock)
    )
    x = train.select(FEATURES).to_numpy()
    bands = minute_band(train["minutes_target"].to_numpy())
    present = train["minutes_target"].to_numpy() > 0
    known = train["start_target"].is_not_null().to_numpy()
    models = {
        "minutes": lgb.LGBMClassifier(**PARAMS).fit(x, bands),
        "points": lgb.LGBMRegressor(**PARAMS).fit(
            np.column_stack([x[present], bands[present]]), train["base_target"].to_numpy()[present]
        ),
        "starter": lgb.LGBMClassifier(**PARAMS).fit(
            x[known], train["start_target"].fill_null(0).to_numpy()[known] > 0
        ),
    }
    return models | {
        "curves": curves,
        "fit_max_label": train["label_end"].max(),
        "cal_max_label": cal["label_end"].max(),
        "cut": cut,
        "training_rows": len(train),
        "calibration_rows": len(cal),
        "crossfit_pivot": pivot,
    }


def fitted(frame, before, name, directory, teachers=None):
    folder = directory / (before.replace("/", "-") + "-" + name)
    manifest = folder / "model.json"
    if manifest.exists():
        spec = json.loads(manifest.read_text())
        return spec | {
            k: cached_booster(folder / (k + ".txt"), checksum) for k, checksum in spec["hashes"].items()
        }
    check_cancelled()
    bundle = (
        crossfit(frame, before, *teachers)
        if name == "G_crossfit8"
        else fit_models(frame, before, "uniform", calibration_groups=4 if name == "B_cal4" else 8)
    )
    folder.mkdir(parents=True, exist_ok=True)
    spec = {k: v for k, v in bundle.items() if k not in ("minutes", "points", "starter")}
    spec["hashes"] = {}
    for key in ("minutes", "points", "starter"):
        path = folder / (key + ".txt")
        save_booster_atomic(bundle[key].booster_, path)
        spec["hashes"][key] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(manifest, spec)
    print("Future temporal fit", before, name, spec["training_rows"], flush=True)
    return bundle


def mixture_brier(base, expert, frame):
    _, p, _ = predict_models(base, frame.select(NAMES).to_numpy(), frame["dc_prior"].to_list())
    availability = 1 - (1 - frame["availability"].to_numpy()) * np.exp(-frame["lead_weeks"].to_numpy() / 5)
    p[:, 1:] *= availability[:, None]
    p[:, 0] = 1 - p[:, 1:].sum(1)
    _, hp, _ = predict(expert, frame)
    combined = 0.5 * (p + hp)
    combined[frame["selectable"].to_numpy() == 0] = [1, 0, 0, 0]
    band = minute_band(frame["minutes_target"].to_numpy())
    return ((combined - np.eye(4)[band]) ** 2).sum(1)


def run(store, source_path=None):
    from .consensus import paired_interval
    from .decision import score_metrics

    path = Path(source_path) if source_path is not None else build_data(store)
    contract = {
        "version": "future-calibration-v1",
        "data_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "models": ["A_cal8", "B_cal4", "G_crossfit8"],
        "features": FEATURES,
        "parameters": PARAMS,
        "source_events": [4, 12, 20, 28, 36],
        "selection": "2024/25 source GW4/12, only target labels through GW19; frozen lowest ownership-weighted MSE",
        "acceptance": "2024/25 source GW20/28/36; independent 2025/26 five source deadlines; only lead>0, same 50% future mixture and immediate reference",
        "fit": "Each sampled source deadline is independently refitted. Both source and target label groups precede source cutoff; label finalization time also precedes cutoff. Crossfit teachers purge against each calibration block.",
        "gate": "Independent periods require upper 95% target-deadline block weighted-MSE delta below zero and all/relevant RMSE no worse; distribution and legal continuous-account gates required before activation",
        "limits": "Five scheduled refits per season sample the actual per-deadline procedure; previously studied archives, not a blinded prospective claim. Overlapping horizons are blocked by target deadline.",
    }
    directory = store.root / "models" / ("future-temporal-" + digest(contract)[:24])
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "protocol.json", contract)
    frame = pl.read_parquet(path)
    outputs = []
    with threadpool_limits(limits=1):
        for season in ("2024/25", "2025/26"):
            for event in contract["source_events"]:
                before = f"{season}-GW{event:02d}"
                target = directory / (before.replace("/", "-") + ".parquet")
                receipt = target.with_suffix(".json")
                test = frame.filter((pl.col("source_group") == before) & (pl.col("lead_weeks") > 0))
                if target.exists() and receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if hashlib.sha256(target.read_bytes()).hexdigest() != saved["hash"]:
                        raise ValueError("Future temporal prediction checksum mismatch")
                    result = pl.read_parquet(target)
                    if saved.get("probability_contract") != "production-mixture-v1":
                        base = reference(store, before, directory)
                        a = fitted(frame, before, "A_cal8", directory)
                        b = fitted(frame, before, "B_cal4", directory)
                        g = fitted(frame, before, "G_crossfit8", directory, (a, b))
                        for name, expert in (("A", a), ("B", b), ("G", g)):
                            result = result.with_columns(
                                pl.Series(name + "_brier", mixture_brier(base, expert, test))
                            )
                        result.write_parquet(target)
                        atomic_json(
                            receipt,
                            {
                                "hash": hashlib.sha256(target.read_bytes()).hexdigest(),
                                "before": before,
                                "probability_contract": "production-mixture-v1",
                                "supersedes_head_only_brier_hash": saved["hash"],
                            },
                        )
                    outputs.append(result)
                    continue
                base = reference(store, before, directory)
                a = fitted(frame, before, "A_cal8", directory)
                b = fitted(frame, before, "B_cal4", directory)
                g = fitted(frame, before, "G_crossfit8", directory, (a, b))
                means = distributions(test, base, {"A": a, "B": b, "G": g})
                result = test.select(
                    "season",
                    "event",
                    "source_group",
                    "label_end",
                    "player_id",
                    "fixture_id",
                    "lead_weeks",
                    "target",
                    "selected_by_percent",
                    "now_cost",
                    "minutes_target",
                )
                for name in ("A", "B", "G"):
                    result = result.with_columns(
                        pl.Series(name, means[name + "_0.5"]),
                        pl.Series(name + "_brier", mixture_brier(base, {"A": a, "B": b, "G": g}[name], test)),
                    )
                result.write_parquet(target)
                atomic_json(
                    receipt,
                    {
                        "hash": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "before": before,
                        "probability_contract": "production-mixture-v1",
                    },
                )
                outputs.append(result)
                # Only this experiment's checkpoints; accepted weights untouched.
                # References use the existing raw-text loader, so retain those.
                for folder in directory.glob(before.replace("/", "-") + "-*"):
                    if not folder.name.startswith("reference-"):
                        compress_checkpoints(folder)
                print("Future temporal deadline", before, flush=True)
    result = pl.concat(outputs)
    select = (
        (result["season"].to_numpy() == "2024/25")
        & (result["event"].to_numpy() <= 19)
        & (result["label_end"].to_numpy() <= "2024/25-GW19")
    )
    scores = {name: loss(result, result[name].to_numpy(), select) for name in ("A", "B", "G")}
    selected = min(scores, key=lambda k: scores[k])
    periods = {}
    for season in ("2024/25", "2025/26"):
        mask = (result["season"].to_numpy() == season) & (
            (result["event"].to_numpy() >= 20) if season == "2024/25" else True
        )
        relevant = mask & (
            (result["now_cost"].to_numpy() >= 60) | (result["selected_by_percent"].to_numpy() >= 5)
        )
        y = result["target"].to_numpy()
        interval = paired_interval(
            differences(result, result["A"].to_numpy(), result[selected].to_numpy(), mask)
        )
        metrics = {
            name: {
                "all": score_metrics(y[mask], result[name].to_numpy()[mask]),
                "relevant": score_metrics(y[relevant], result[name].to_numpy()[relevant]),
                "participation_brier": float(result[name + "_brier"].to_numpy()[mask].mean()),
            }
            for name in ("A", "B", "G")
        }
        passed = (
            selected != "A"
            and interval[1] < 0
            and all(metrics[selected][k]["rmse"] <= metrics["A"][k]["rmse"] for k in ("all", "relevant"))
            and metrics[selected]["participation_brier"] <= metrics["A"]["participation_brier"] + 0.002
        )
        periods[season] = {
            "rows": int(mask.sum()),
            "metrics": metrics,
            "weighted_mse_delta_95ci": interval,
            "forecast_gate_passed": bool(passed),
        }
    report = {
        "at": now(),
        "protocol": contract,
        "selection_losses": scores,
        "selected": selected,
        "periods": periods,
        "forecast_gate_passed": all(p["forecast_gate_passed"] for p in periods.values()),
        "promoted": False,
        "directory": str(directory),
    }
    report["id"] = store.put("future_temporal_validation", report)
    atomic_json(Path("artifacts/evaluations/future_temporal_0_14_4.json"), report)
    return report
