"""Select a future-only expert while preserving the immediate production model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .consensus import mix_distributions, paired_interval
from .decision import NAMES, predict_models, score_metrics
from .fixture_horizon import build_data
from .horizon_trial import fitted, prediction, reference
from .storage import atomic_json, now


def future_mask(frame, season, first, last):
    """Selection cannot use later labels from an early long-range forecast."""
    return (
        (frame["season"].to_numpy() == season)
        & (frame["event"].to_numpy() >= first)
        & (frame["event"].to_numpy() <= last)
        & (frame["lead_weeks"].to_numpy() > 0)
        & (frame["label_end"].to_numpy() <= f"{season}-GW{last:02d}")
    )


def distributions(frame, baseline, experts):
    with threadpool_limits(limits=1):
        _, p, c = predict_models(baseline, frame.select(NAMES).to_numpy(), frame["dc_prior"].to_list())
        availability = 1 - (1 - frame["availability"].to_numpy()) * np.exp(
            -frame["lead_weeks"].to_numpy() / 5
        )
        p[:, 1:] *= availability[:, None]
        p[:, 0] = 1 - p[:, 1:].sum(1)
        original = (p, c, np.zeros(len(frame)))
        output = {"reference": original}
        for family, bundle in experts.items():
            _, hp, hc = prediction(bundle, frame)
            for weight in (0.25, 0.5, 1.0):
                output[f"{family}_{weight:g}"] = mix_distributions(
                    original, (hp, hc, np.zeros(len(frame))), weight
                )
    dc = np.asarray(frame["dc_prior"].to_list())
    result = {name: (value[0] * (value[1] + 2 * dc)).sum(1) for name, value in output.items()}
    for means in result.values():
        means[frame["selectable"].to_numpy() == 0] = 0
    return result


def loss(frame, predicted, mask):
    actual = frame["target"].to_numpy()
    weights = 1 + 4 * np.clip(np.nan_to_num(frame["selected_by_percent"].to_numpy()) / 20, 0, 1)
    return float(np.average((predicted[mask] - actual[mask]) ** 2, weights=weights[mask]))


def differences(frame, baseline, candidate, mask):
    values = []
    for group in sorted(set(frame.filter(pl.Series(mask))["label_end"].to_list())):
        chosen = mask & (frame["label_end"].to_numpy() == group)
        values.append(loss(frame, candidate, chosen) - loss(frame, baseline, chosen))
    return values


def run_refinement(store):
    trial = store.get("horizon_trial")
    directory = store.artifact_path(trial["directory"] + "/protocol.json").parent
    protocol = {
        "at": now(),
        "scope": "Future-only forecast expert; immediate forecasts remain exactly the reference",
        "selection": "2024/25 source GW1-19, target labels ending by GW19, lead > 0",
        "acceptance": "2024/25 source GW20-38; separate frozen-configuration 2025/26 replication",
        "weights": [0.25, 0.5, 1.0],
        "loss": "Predeadline ownership-weighted squared error, 1 + 4 min(ownership/20,1)",
        "promotion": "Paired future loss interval below zero, no all-player or relevant-player RMSE deterioration, followed by positive full transfer/chip replay gain; current season never selects configuration",
        "limits": "Historical reconstruction on previously studied archives; not blinded prospective evidence",
    }
    atomic_json(directory / "future-protocol.json", protocol)
    all_data = pl.read_parquet(build_data(store))
    data = all_data.filter(pl.col("season") == "2024/25")
    models = {
        family: fitted(store, all_data, "2024/25-GW01", family, directory) for family in ("uniform", "squad")
    }
    base = reference(store, "2024/25-GW01", directory)
    means = distributions(data, base, models)
    select = future_mask(data, "2024/25", 1, 19)
    scores = {name: loss(data, pred, select) for name, pred in means.items()}
    selected = min(scores, key=lambda name: scores[name])
    atomic_json(
        directory / "future-selection.json", {"selected": selected, "scores": scores, "protocol": protocol}
    )
    print("Future expert selected", selected, scores, flush=True)
    metrics = {}
    delta = []
    seasons = ["2024/25", "2025/26"] if selected != "reference" else ["2024/25"]
    for season in seasons:
        if season != "2024/25":
            family = selected.rsplit("_", 1)[0]
            data = all_data.filter(pl.col("season") == season)
            models = {family: fitted(store, all_data, season + "-GW01", family, directory)}
            means = distributions(data, reference(store, season + "-GW01", directory), models)
        mask = future_mask(data, season, 20 if season == "2024/25" else 1, 38)
        changes = differences(data, means["reference"], means[selected], mask)
        delta.extend(changes)
        relevant = mask & (
            (data["now_cost"].to_numpy() >= 60) | (data["selected_by_percent"].to_numpy() >= 5)
        )
        actual = data["target"].to_numpy()
        metrics[season] = {
            "reference": score_metrics(actual[mask], means["reference"][mask]),
            "candidate": score_metrics(actual[mask], means[selected][mask]),
            "relevant_reference": score_metrics(actual[relevant], means["reference"][relevant]),
            "relevant_candidate": score_metrics(actual[relevant], means[selected][relevant]),
            "weighted_loss_delta_95ci": paired_interval(changes),
        }
        print("Future acceptance", season, metrics[season], flush=True)
    interval = paired_interval(delta)
    passed = (
        selected != "reference"
        and interval[1] < 0
        and all(
            m["candidate"]["rmse"] <= m["reference"]["rmse"]
            and m["relevant_candidate"]["rmse"] <= m["relevant_reference"]["rmse"]
            for m in metrics.values()
        )
    )
    result = {
        "at": now(),
        "protocol": protocol,
        "selected": selected,
        "selection_losses": scores,
        "metrics": metrics,
        "weighted_loss_delta_95ci": interval,
        "forecast_gate_passed": bool(passed),
        "promoted": False,
        "trial_id": trial["id"],
        "directory": str(directory),
        "next_gate": "Full production transfer/chip comparison",
    }
    result["id"] = store.put("horizon_refinement", result)
    atomic_json(Path("artifacts/evaluations/future-horizon-refinement.json"), result)
    return result
