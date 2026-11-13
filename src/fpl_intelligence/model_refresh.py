"""Refit a previously accepted family without rerunning model selection.

Eligibility is a content fingerprint of finalized football observations, rather
than changing future fixture payloads or the wall-clock age of a download.
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

from .decision import (
    NAMES,
    PARAMETERS,
    VERSION,
    build_dataset,
    canonical_rows,
    fit_components,
    minute_band,
    residual_contract,
)
from .models import save_booster_atomic
from .search_control import check_cancelled
from .storage import digest, now


def observation_signature(store):
    rows, _, _ = canonical_rows(store)
    groups = [f"{r['season']}-GW{int(r['round']):02d}" for r in rows]
    through = max(groups, default="")
    dependencies = {}
    market = store.get("market")
    event = int(through.split("GW")[-1]) if through.startswith(market["season"]) else None
    cutoff = next((e["deadline_time"] for e in market["events"] if e["id"] == event), None)
    if cutoff and store.list("workload", 1):
        from .workload import records, time_value

        eligible = [r for r in records(store) if time_value(r["known_after"]) < time_value(cutoff)]
        dependencies["workload"] = digest(
            sorted(
                (r["player_code"], r["kickoff"], r["known_after"], r["minutes"], r["extra_time"])
                for r in eligible
            )
        )
    if store.list("context_archive", 1):
        context = store.get("context_archive")
        dependencies["context"] = digest(
            sorted(
                [
                    {k: r.get(k) for k in ("season", "event", "source", "hash")}
                    for r in context["entries"]
                    if f"{r['season']}-GW{int(r['event']):02d}" <= through
                ],
                key=lambda r: (r["season"], r["event"], r["source"]),
            )
        )
    return {
        "signature": digest(rows),
        "through": through,
        "dependencies": dependencies,
        "rows": len(rows),
        "seasons": {
            season: sum(r["season"] == season for r in rows) for season in sorted({r["season"] for r in rows})
        },
    }


def training_needed(store, active, observations=None):
    """Migrate a legacy receipt only with a reconstructed matching feature hash."""
    observations = observations or observation_signature(store)
    matches = store.list_matching("model_training_state", {"decision_model_id": active.get("id")}, 1)
    if matches:
        return store.get("model_training_state", matches[0]["id"])["observations"] != observations
    if not active.get("feature_id") or active.get("fusion_model_id"):
        return True
    old = store.get("decision_features", active["feature_id"])
    current, _ = build_dataset(store)
    if old["hash"] != current["hash"]:
        return True
    if active.get("role_minutes_model_id"):
        role = store.get("role_minutes_model", active["role_minutes_model_id"])
        if role.get("completed_labels_through", "") < observations["through"]:
            return True
        from .fusion import load_data
        from .fusion_data import build_fusion_data
        from .role_minutes import historical_inputs, input_signature

        arrays, meta, features = load_data(store, role["feature_id"])
        prior_role_input = role.get("input_signature") or input_signature(
            historical_inputs(store, arrays, meta, features, role["context_archive_id"]),
            arrays["minutes"],
            meta,
        )
        _, _, rebuilt = build_fusion_data(store)
        arrays, meta, features = load_data(store, rebuilt["id"])
        current_role_input = input_signature(
            historical_inputs(store, arrays, meta, features, store.get("context_archive")["id"]),
            arrays["minutes"],
            meta,
        )
        if prior_role_input != current_role_input:
            return True
    if active.get("future_horizon_model_id"):
        future = store.get("future_horizon_model", active["future_horizon_model_id"])
        if future["training_evidence"]["cal_max_label"] < observations["through"]:
            return True
        from .availability import moment
        from .fixture_horizon import FEATURES, build_data, partitions
        from .future_expert import load_expert

        load_expert(store, future)
        old_path = store.artifact_path(future["data_path"])
        if hashlib.sha256(old_path.read_bytes()).hexdigest() != future["data_hash"]:
            raise ValueError("Accepted future training data checksum mismatch")
        consumed = [*FEATURES, "source_group", "label_end", "minutes_target", "start_target", "base_target"]

        def future_signature(path):
            frame = pl.read_parquet(path)
            train, cal, _ = partitions(frame, future["fit_before"], moment(future["fit_as_of"]).timestamp())
            return [
                hashlib.sha256(part.select(consumed).hash_rows(seed=137).to_numpy().tobytes()).hexdigest()
                for part in (train, cal)
            ]

        if future_signature(old_path) != future_signature(build_data(store)):
            return True
    from .decision import load_bundle

    load_bundle(store, active)
    store.put(
        "model_training_state",
        {
            "at": now(),
            "decision_model_id": active["id"],
            "observations": observations,
            "refitted": False,
            "verification": "Legacy core record hash matches rebuilt finalized features; role consumed input/target matrix and purged future fitting/calibration matrices match their pinned artifacts; component completed-label coverage and core/future weight hashes verified",
        },
    )
    return False


def represented_group(model):
    return model.get("completed_labels_through") or max(
        (
            r.get("deadline_group", f"{r['season']}-GW{int(r['event']):02d}")
            for r in model.get("predictions", [])
        ),
        default="",
    )


def fit_bounds(frame):
    groups = sorted(frame["deadline_group"].unique().to_list())
    if len(groups) < 9:
        raise ValueError("At least nine completed deadlines are required")
    cut = groups[-8]
    train = frame.filter(pl.col("deadline_group") < cut)
    cal = frame.filter(pl.col("deadline_group") >= cut)
    return (
        train,
        cal,
        {
            "fit_min": train["deadline_group"].min(),
            "fit_max": train["deadline_group"].max(),
            "calibration_min": cal["deadline_group"].min(),
            "calibration_max": cal["deadline_group"].max(),
            "completed_labels_through": groups[-1],
            "training_rows": len(train),
            "calibration_rows": len(cal),
        },
    )


def refit_reference(store, active):
    """Fixed accepted architecture; calibration and weights share one manifest."""
    if active.get("version") != VERSION or not active.get("promoted") or active.get("fusion_model_id"):
        raise ValueError("Operational refit requires the accepted direct-points family")
    feature, frame = build_dataset(store)
    train, cal, bounds = fit_bounds(frame)
    check_cancelled()
    from .search_control import report_phase

    report_phase("Fitting points and participation models on finalized historical labels")
    bundle = fit_components(train, cal)
    parameters = {
        "version": VERSION,
        "names": NAMES,
        "parameters": PARAMETERS,
        "feature": feature["hash"],
        "file_hash": feature["file_hash"],
        "policy": "expanding-reserved8-v1",
    }
    directory = store.root / "models" / ("reference-refit-" + digest(parameters)[:24])
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for key in ("minutes", "starter", "points"):
        path = directory / (key + ".txt")
        save_booster_atomic(bundle[key].booster_, path)
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    residuals, dependence = residual_contract(bundle, cal)
    observed = cal["minutes_target"].to_numpy()
    bands = minute_band(observed)
    minute_quantiles = {
        str(k): np.quantile(observed[bands == k], np.linspace(0, 1, 101)).tolist() for k in range(4)
    }
    body = {k: v for k, v in active.items() if k not in ("id", "previous_decision_model_id")}
    body.update(
        at=now(),
        directory=directory.name,
        model_hashes=hashes,
        feature_id=feature["id"],
        seasons=feature["seasons"],
        curves=bundle["curves"],
        residuals=residuals,
        dependence=dependence,
        minute_quantiles=minute_quantiles,
        production_training_rows=len(train),
        production_calibration_rows=len(cal),
        production_fit_before=bounds["calibration_min"],
        completed_labels_through=bounds["completed_labels_through"],
        training_evidence=bounds,
        training_policy="expanding-reserved8-v1",
        accepted_family_evaluation_id=active.get("accepted_family_evaluation_id", active["id"]),
        evaluation_scope="Frozen accepted family and training policy; retained historical OOF scores are not grades of these newly refitted weights",
        previous_decision_model_id=active["id"],
    )
    if active.get("future_horizon_model_id"):
        from .future_training import refit_expert

        acceptance = store.get("future_horizon_model", active["future_horizon_model_id"])["acceptance_id"]
        body["future_horizon_model_id"] = refit_expert(store, acceptance)["id"]
    body["id"] = store.put("decision_model", body)
    return body
