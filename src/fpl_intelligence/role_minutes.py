"""Role-aware participation with separately held-out, deadline-level acceptance.

Only immediate fixtures use this head. Conditional point returns and the
independently validated future expert retain their original contracts.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from .decision import NAMES, PARAMETERS, fit_components, minute_band, predict_models
from .models import curve_apply, curve_fit
from .observations import evaluation_report
from .storage import digest, now
from .workload import WORKLOAD_TRANSFORM, workload_features
from .workload import records as workload_records

VERSION = "role-minutes-v3"
PARAMS = PARAMETERS | {"n_estimators": 220, "num_leaves": 16, "min_child_samples": 100, "reg_lambda": 20}
ROLE_NAMES = (
    "starts_per_elapsed_gameweek",
    "minutes_per_elapsed_gameweek",
    "minutes_per_start",
    "recent_history_known",
    "last_start",
    "recent_three_starts",
    "recent_five_starts",
    "season_starts_known",
    "season_minutes_known",
)


def role_vector(packet, sequence, mask, event):
    packet = packet or {}
    games = max(1, event - 1)
    # Stored sequences use float16, while live inputs use float32. Aggregate in
    # one precision so a five-start rate cannot cross a tree split at serving.
    recent = np.asarray(sequence[mask], dtype=np.float32)
    starts = float(packet.get("starts") or 0)
    minutes = float(packet.get("minutes") or 0)
    return [
        starts / games,
        minutes / (90 * games),
        minutes / (90 * max(1, starts)),
        float(len(recent) > 0),
        *(float(recent[-k:, 1].mean()) if len(recent) else 0 for k in (1, 3, 5)),
        float(packet.get("starts") is not None),
        float(packet.get("minutes") is not None),
    ]


def matrix(tab, roles, news, workload, quality):
    return np.column_stack([tab, roles, news, workload, quality[:, [1, 3]]]).astype(np.float32)


def historical_inputs(store, arrays, metadata, features, context_id=None):
    from .context_archive import load_context

    packets = load_context(store, context_id or features.get("lineage", {}).get("context_archive"))
    workload_id = features.get("lineage", {}).get("workload")
    workload = workload_records(store, workload_id) if workload_id else []
    by_code = defaultdict(list)
    for row in workload:
        by_code[row["player_code"]].append(row)
    roles, loads = [], []
    for i, row in enumerate(metadata):
        # Historical FPL IDs reset every season; the archive is keyed by stable code.
        packet = packets.get((row["season"], row["event"], row["code"]))
        roles.append(role_vector(packet, arrays["sequence"][i], arrays["sequence_mask"][i], row["event"]))
        loads.append(workload_features(by_code[row["code"]], row["code"], row["deadline"]))
    return matrix(arrays["tab"], roles, arrays["news"], loads, arrays["quality"])


def fit_head(x, minutes, training, calibration):
    head = lgb.LGBMClassifier(**PARAMS).fit(x[training], minute_band(minutes[training]))
    raw = np.asarray(head.predict_proba(x[calibration]))
    truth = minute_band(minutes[calibration])
    curves = {str(k): curve_fit(raw[:, k], (truth == k).astype(int)) for k in range(4)}
    return head, curves


def predict(head, curves, x, news):
    raw = head.predict_proba(x) if hasattr(head, "predict_proba") else head.predict(x, num_threads=1)
    p = np.column_stack([curve_apply(raw[:, k], curves[str(k)]) for k in range(4)])
    p /= np.maximum(p.sum(1, keepdims=True), 1e-9)
    # Published chance is an input, rather than an additional multiplicative penalty.
    p[(news[:, 5] > 0) & (news[:, 6] == 0)] = [1, 0, 0, 0]
    return p


def delta_interval(rows, metric, weight, repeats=2000):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row[str(weight)][metric] - row["0"][metric])
    values = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(631)
    samples = values[rng.integers(0, len(values), (repeats, len(values)))].mean(1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def validate(store):
    from .fusion import load_data, published_availability

    arrays, metadata, features = load_data(store)
    context_id = store.get("context_archive")["id"]
    if features.get("workload_transform") != WORKLOAD_TRANSFORM:
        raise ValueError("Rebuild source features before validating the current role contract")
    x = historical_inputs(store, arrays, metadata, features, context_id)
    groups = np.array([r["deadline_group"] for r in metadata])
    seasons = np.array([r["season"] for r in metadata])
    unique = sorted(set(groups))
    frame = pl.DataFrame(arrays["tab"], schema=NAMES).with_columns(
        pl.Series("minutes_target", arrays["minutes"]),
        pl.Series(
            "start_target",
            [float(v) if known else None for v, known in zip(arrays["start"], arrays["start_mask"])],
        ),
        pl.Series("base_target", arrays["base_target"]),
    )
    rows, folds = [], []
    for season in ("2024/25", "2025/26"):
        testing = sorted(set(groups[seasons == season]))
        for first in range(0, len(testing), 8):
            block = testing[first : first + 8]
            cut = [g for g in unique if g < block[0]][-8]
            train = np.flatnonzero(groups < cut)
            cal = np.flatnonzero((groups >= cut) & (groups < block[0]))
            test = np.flatnonzero(np.isin(groups, block))
            ref = fit_components(frame[train], frame[cal])
            _, rp, conditional = predict_models(ref, arrays["tab"][test], arrays["dc_prior"][test])
            rp[:, 1:] *= np.array([published_availability(r) for r in arrays["news"][test]])[:, None]
            rp[:, 0] = 1 - rp[:, 1:].sum(1)
            head, curves = fit_head(x, arrays["minutes"], train, cal)
            candidate = predict(head, curves, x[test], arrays["news"][test])
            bands = minute_band(arrays["minutes"][test])
            for j, i in enumerate(test):
                item = {k: metadata[i][k] for k in ("season", "event", "position", "player_id", "fixture_id")}
                item.update(
                    group=groups[i],
                    actual=float(arrays["target"][i]),
                    minutes=float(arrays["minutes"][i]),
                    regular=bool(metadata[i]["prior_start"] >= 0.6),
                )
                for weight in (0, 0.25, 0.5, 1):
                    p = (1 - weight) * rp[j] + weight * candidate[j]
                    mean = float(p @ (conditional[j] + 2 * arrays["dc_prior"][i]))
                    item[str(weight)] = {
                        "brier": float(((p - np.eye(4)[bands[j]]) ** 2).sum()),
                        "points_squared": float((mean - arrays["target"][i]) ** 2),
                        "any_brier": float((1 - p[0] - int(bands[j] > 0)) ** 2),
                        "sixty_brier": float((p[2:].sum() - int(arrays["minutes"][i] >= 60)) ** 2),
                        "mean": mean,
                    }
                rows.append(item)
            folds.append(
                {
                    "test": block,
                    "train_rows": len(train),
                    "calibration_rows": len(cal),
                    "test_rows": len(test),
                }
            )
            print("Role minutes validated", block[0], "through", block[-1], flush=True)
    selection = [r for r in rows if r["season"] == "2024/25" and r["event"] <= 19]
    weight = min((0, 0.25, 0.5, 1), key=lambda w: np.mean([r[str(w)]["brier"] for r in selection]))
    metrics: dict[str, Any] = {}
    for name, period in (
        ("acceptance_2024", [r for r in rows if r["season"] == "2024/25" and r["event"] > 19]),
        ("replication_2025", [r for r in rows if r["season"] == "2025/26"]),
    ):
        metrics[name] = {"rows": len(period), "deadlines": len({r["group"] for r in period})}
        for label, w in (("reference", 0), ("candidate", weight)):
            metrics[name][label] = {
                k: float(np.mean([r[str(w)][k] for r in period]))
                for k in ("brier", "points_squared", "any_brier", "sixty_brier")
            }
            metrics[name][label]["regular_points_squared"] = float(
                np.mean([r[str(w)]["points_squared"] for r in period if r["regular"]])
            )
        metrics[name]["delta_95ci"] = {
            k: delta_interval(period, k, weight) for k in ("brier", "points_squared")
        }
        metrics[name]["positions"] = {
            str(position): {
                label: float(
                    np.mean([r[str(w)]["points_squared"] for r in period if int(r["position"]) == position])
                )
                for label, w in (("reference", 0), ("candidate", weight))
            }
            for position in (1, 2, 3, 4)
        }
    passed = weight > 0 and all(
        m["delta_95ci"]["brier"][1] < 0
        and m["candidate"]["points_squared"] < m["reference"]["points_squared"]
        and m["candidate"]["regular_points_squared"] <= m["reference"]["regular_points_squared"]
        for m in metrics.values()
    )
    path = (
        store.root
        / "features"
        / ("role-minutes-" + digest({"features": features["id"], "version": VERSION})[:24] + ".parquet")
    )
    pl.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("0", "0.25", "0.5", "1")}
            | {"reference": r["0"]["mean"], "candidate": r[str(weight)]["mean"]}
            for r in rows
        ]
    ).write_parquet(path)
    result = {
        "at": now(),
        "version": VERSION,
        "feature_id": features["id"],
        "context_archive_id": context_id,
        "workload_transform": WORKLOAD_TRANSFORM,
        "weight": weight,
        "passed": bool(passed),
        "metrics": metrics,
        "folds": folds,
        "prediction_path": str(path),
        "prediction_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "protocol": "Expanding eight-deadline blocks; eight earlier calibration deadlines excluded from fits. Mixture selected on 2024/25 GW1–19 only; acceptance GW20–38; independent full 2025/26 replication. Confidence intervals resample whole deadlines. No current-season results select the model.",
        "scope": "Immediate participation probabilities. Conditional scoring and future horizon are unchanged; this forecast test does not certify transfer-policy improvement.",
        "feature_contract": "Float32 role aggregation; explicitly pinned full deadline context and positive-exposure workload. Live and historical role vectors share the same transform.",
    }
    result["id"] = store.put("role_minutes_validation", result)
    evaluation_report(store, "role_minutes_validation.json", result)
    return result


def train(store, feature_id=None):
    from .fusion import load_data

    validation = (
        store.get("role_minutes_validation") if store.list("role_minutes_validation", 1) else validate(store)
    )
    if not validation["passed"] or validation["version"] != VERSION:
        raise ValueError("Role minutes failed its chronological forecast gate")
    arrays, metadata, features = load_data(store, feature_id or validation["feature_id"])
    context_id = store.get("context_archive")["id"] if feature_id else validation["context_archive_id"]
    from .search_control import report_phase

    report_phase("Preparing role-aware minutes training inputs")
    x = historical_inputs(store, arrays, metadata, features, context_id)
    groups = np.array([r["deadline_group"] for r in metadata])
    cut = sorted(set(groups))[-8]
    training, calibration = np.flatnonzero(groups < cut), np.flatnonzero(groups >= cut)
    report_phase("Fitting the accepted role-aware minutes model")
    head, curves = fit_head(x, arrays["minutes"], training, calibration)
    directory = (
        store.root
        / "models"
        / (
            "role-minutes-"
            + digest({"validation": validation["id"], "features": features["id"], "context": context_id})[:24]
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "minutes.txt"
    from .models import save_booster_atomic

    save_booster_atomic(head.booster_, path)
    result = {
        "at": now(),
        "version": VERSION,
        "validation_id": validation["id"],
        "feature_id": features["id"],
        "context_archive_id": context_id,
        "path": str(path),
        "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "curves": curves,
        "weight": validation["weight"],
        "training_rows": len(training),
        "calibration_rows": len(calibration),
        "completed_labels_through": max(groups),
        "width": x.shape[1],
        "workload_transform": WORKLOAD_TRANSFORM,
        "input_signature": input_signature(x, arrays["minutes"], metadata),
        "training_evidence": {
            "fit_min": str(min(groups[training])),
            "fit_max": str(max(groups[training])),
            "calibration_min": str(min(groups[calibration])),
            "calibration_max": str(max(groups[calibration])),
            "training_rows": len(training),
            "calibration_rows": len(calibration),
        },
        "promoted": True,
        "role_features": ROLE_NAMES,
    }
    result["id"] = store.put("role_minutes_model", result)
    return result


def apply(store, model_id, rows, market, players, probabilities, start_event):
    from .fusion_data import live_inputs

    spec = store.get("role_minutes_model", model_id)
    if not spec["promoted"] or spec["version"] not in (VERSION, "role-minutes-v2"):
        raise ValueError("Unsupported role minutes model")
    selected = [
        i
        for i, row in enumerate(rows)
        if row["event"] == start_event
        and not row.get("suspension_served")
        and row["availability"] == row.get("base_availability", row["availability"])
    ]
    current = [rows[i] for i in selected]
    if not current:
        return probabilities, np.zeros(len(rows), dtype=bool)
    inputs = live_inputs(store, current, market, players, spec["feature_id"])
    from .observations import information_time
    from .workload import time_value

    by_code = defaultdict(list)
    cutoff = information_time(store)
    for record in workload_records(store):
        if time_value(record["known_after"]) < time_value(cutoff):
            by_code[record["player_code"]].append(record)
    roles, loads = [], []
    for j, row in enumerate(current):
        player = players[row["player_id"]]
        roles.append(role_vector(player.raw, inputs["sequence"][j], inputs["sequence_mask"][j], start_event))
        # Physical exposure is measured at information time, never at a distant future deadline.
        loads.append(
            workload_features(
                by_code[player.code], player.code, market["events"][start_event - 1]["deadline_time"]
            )
        )
    x = matrix(inputs["tab"], roles, inputs["news"], loads, inputs["quality"])
    if x.shape[1] != spec["width"]:
        raise ValueError("Role minutes feature contract mismatch")
    path = store.artifact_path(spec["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["hash"]:
        raise ValueError("Role minutes checksum mismatch")
    candidate = predict(lgb.Booster(model_file=str(path)), spec["curves"], x, inputs["news"])
    result = probabilities.copy()
    for j, i in enumerate(selected):
        reference = probabilities[i].copy()
        reference[1:] *= rows[i]["availability"]
        reference[0] = 1 - reference[1:].sum()
        result[i] = (1 - spec["weight"]) * reference + spec["weight"] * candidate[j]
        if rows[i]["availability"] == 0:
            result[i] = [1, 0, 0, 0]
    included = np.zeros(len(rows), dtype=bool)
    included[selected] = True
    return result, included


def attach(store, model, role_id=None):
    """Bind the accepted component explicitly; never alter a frozen model object."""
    if model.get("fusion_model_id"):
        return model
    spec = store.get("role_minutes_model", role_id)
    if model.get("role_minutes_model_id") == spec["id"]:
        return model
    body = {k: v for k, v in model.items() if k != "id"}
    body.update(
        at=now(),
        role_minutes_model_id=spec["id"],
        previous_decision_model_id=model["id"],
        evaluation_scope=(
            model["evaluation_scope"] + "; role minutes retains its separate model-family validation"
        )
        if model.get("training_evidence") and model.get("evaluation_scope")
        else "Base conditional-points validation is reported below; role minutes has its separate two-season held-out validation",
    )
    body["id"] = store.put("decision_model", body)
    return body


def audit_input_contract(store, event=None):
    """Compare stored historical tensors with the production input builder."""
    import tempfile

    from .consensus import replay_packets
    from .decision import canonical_rows, future_inputs
    from .fusion import load_data
    from .fusion_data import live_inputs
    from .observations import completed_before
    from .pipeline_replay import attach_market, attach_observations
    from .storage import Store

    active = store.get("decision_model")
    spec = store.get("role_minutes_model", active["role_minutes_model_id"])
    if not spec.get("context_archive_id"):
        raise ValueError("Role weights have no pinned context; migrate before auditing")
    arrays, meta, features = load_data(store, spec["feature_id"])
    historical = historical_inputs(store, arrays, meta, features, spec["context_archive_id"])
    season = store.get("market")["season"]
    event = event or max(r["event"] for r in meta if r["season"] == season)
    packet = replay_packets(store)[season, event]
    originals, _, _ = canonical_rows(store)
    past = completed_before(originals, packet["deadline"], f"{season}-GW{event:02d}")
    with tempfile.TemporaryDirectory(prefix="fie-role-contract-") as temporary:
        root = Path(temporary)
        path = root / "features/history.parquet"
        path.parent.mkdir()
        pl.DataFrame(past, infer_schema_length=None, strict=False).write_parquet(path)
        staged = Store(root / "data")
        attach_observations(
            staged, {"path": path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, packet
        )
        players, market = attach_market(staged, packet, originals)
        for kind in ("shot_archive", "temporal_model", "elite_cohort", "workload"):
            identity = features.get("lineage", {}).get(kind)
            if identity:
                body = store.get(kind, identity)
                staged.put(kind, {k: v for k, v in body.items() if k != "id"})
        staged.put("fusion_features", {k: v for k, v in features.items() if k != "id"})
        rows, _ = future_inputs(staged, market, players, event, event)
        inputs = live_inputs(staged, rows, market, players, features["id"])
        roles, loads = [], []
        work = defaultdict(list)
        for row in workload_records(staged):
            work[row["player_code"]].append(row)
        for j, row in enumerate(rows):
            player = players[row["player_id"]]
            roles.append(role_vector(player.raw, inputs["sequence"][j], inputs["sequence_mask"][j], event))
            loads.append(workload_features(work[player.code], player.code, packet["deadline"]))
        served = matrix(inputs["tab"], roles, inputs["news"], loads, inputs["quality"])
        lookup = {
            (r["player_id"], r["fixture_id"]): i
            for i, r in enumerate(meta)
            if r["season"] == season and r["event"] == event
        }
        differences = []
        mismatches = []
        labels = (
            list(NAMES)
            + list(ROLE_NAMES)
            + [f"news_{i}" for i in range(13)]
            + [f"load_{i}" for i in range(7)]
            + ["news_known", "workload_known"]
        )
        for j, row in enumerate(rows):
            old = lookup.get((row["player_id"], row["fixture_id"]))
            if old is not None:
                difference = np.abs(historical[old] - served[j])
                differences.append(difference)
                if difference.max() > 1e-4:
                    mismatches.append(
                        {
                            "player_id": row["player_id"],
                            "name": players[row["player_id"]].web_name,
                            "fields": {
                                labels[k]: [float(historical[old, k]), float(served[j, k])]
                                for k in np.flatnonzero(difference > 1e-4)
                            },
                        }
                    )
        delta = np.asarray(differences)
        maxima = delta.max(axis=0)
        # Short-core rest scheduling for a newly registered player can carry
        # additional fixture information; the role/news/workload contract must
        # remain identical when fed the same archived observations.
        exact = bool(np.all(maxima <= 1e-4))
        report = {
            "at": now(),
            "version": VERSION,
            "event": event,
            "season": season,
            "role_minutes_model_id": spec["id"],
            "feature_id": features["id"],
            "context_id": spec["context_archive_id"],
            "rows": len(delta),
            "maximum_core_delta": float(maxima[: len(NAMES)].max()),
            "maximum_role_source_delta": float(maxima[len(NAMES) :].max()),
            "core_differences": {name: float(maxima[i]) for i, name in enumerate(NAMES) if maxima[i] > 1e-4},
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:20],
            "passed": exact,
            "temporary_inputs_removed": True,
        }
    report["id"] = store.put("role_contract_audit", report)
    evaluation_report(store, "role_contract_audit.json", report)
    return report


def reference_predictions(store, model, feature_id=None):
    """Reconstruct the composed primary for challenger/manager training.

    Frozen final weights must never score their own earlier training weeks.
    The accepted mixture is fixed; every forecast uses earlier fit/calibration.
    """
    import inspect

    from .fusion import load_data, published_availability

    spec = store.get("role_minutes_model", model["role_minutes_model_id"])
    if spec["version"] != VERSION or not spec["promoted"]:
        raise ValueError("Unsupported role reference contract")
    arrays, meta, features = load_data(store, feature_id)
    context_id = spec.get("context_archive_id") or features.get("lineage", {}).get("context_archive")
    key = digest(
        {
            "feature": features["id"],
            "context": context_id,
            "version": VERSION,
            "weight": spec["weight"],
            "parameters": PARAMS,
            "implementation": [
                inspect.getsource(f)
                for f in (
                    reference_predictions,
                    fit_head,
                    predict,
                    role_vector,
                    historical_inputs,
                    fit_components,
                    predict_models,
                )
            ],
        }
    )
    for item in store.list("role_minutes_reference", 30):
        cached = store.get("role_minutes_reference", item["id"])
        if cached["key"] == key:
            path = store.artifact_path(cached["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != cached["hash"]:
                raise ValueError("Role reference checksum mismatch")
            return pl.read_parquet(path).to_dicts()
    x = historical_inputs(store, arrays, meta, features, context_id)
    groups = np.array([r["deadline_group"] for r in meta])
    unique = sorted(set(groups))
    testing = [g for g in unique if g >= "2025/26-GW01"]
    frame = pl.DataFrame(arrays["tab"], schema=NAMES).with_columns(
        pl.Series("minutes_target", arrays["minutes"]),
        pl.Series(
            "start_target",
            [float(v) if known else None for v, known in zip(arrays["start"], arrays["start_mask"])],
        ),
        pl.Series("base_target", arrays["base_target"]),
    )
    rows: list[dict[str, Any]] = []
    for first in range(0, len(testing), 8):
        block = testing[first : first + 8]
        cut = [g for g in unique if g < block[0]][-8]
        train = np.flatnonzero(groups < cut)
        cal = np.flatnonzero((groups >= cut) & (groups < block[0]))
        test = np.flatnonzero(np.isin(groups, block))
        reference = fit_components(frame[train], frame[cal])
        _, rp, conditional = predict_models(reference, arrays["tab"][test], arrays["dc_prior"][test])
        rp[:, 1:] *= np.array([published_availability(r) for r in arrays["news"][test]])[:, None]
        rp[:, 0] = 1 - rp[:, 1:].sum(1)
        head, curves = fit_head(x, arrays["minutes"], train, cal)
        probability = (1 - spec["weight"]) * rp + spec["weight"] * predict(
            head, curves, x[test], arrays["news"][test]
        )
        mean = (probability * (conditional + 2 * arrays["dc_prior"][test])).sum(1)
        rows.extend(
            meta[i]
            | {
                "predicted": float(mean[j]),
                "actual": float(arrays["target"][i]),
                "minutes": float(arrays["minutes"][i]),
                "availability_included": True,
            }
            for j, i in enumerate(test)
        )
        print("Composed role reference", block[0], "through", block[-1], flush=True)
    path = store.root / "features" / ("role-reference-" + key[:24] + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(path)
    store.put(
        "role_minutes_reference",
        {
            "at": now(),
            "key": key,
            "path": str(path),
            "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
            "feature_id": features["id"],
            "role_minutes_model_id": spec["id"],
            "scope": "Composed primary out-of-fold reference; fixed accepted mixture, eight earlier calibration deadlines; no final weights reused on historical labels",
        },
    )
    return rows


def ensure(store, model):
    """Ordinary retraining retains the accepted head and learns newly finalized labels."""
    if not model.get("promoted") or model.get("fusion_model_id"):
        return model
    import asyncio

    from .context_archive import import_context
    from .fusion_data import build_fusion_data

    latest = model.get("completed_labels_through") or max(
        (r.get("deadline_group", f"{r['season']}-GW{r['event']:02d}") for r in model.get("predictions", [])),
        default="",
    )
    spec = (
        store.get("role_minutes_model", model.get("role_minutes_model_id"))
        if store.list("role_minutes_model", 1)
        else {}
    )
    if (
        spec.get("version") != VERSION
        or not spec.get("context_archive_id")
        or spec.get("completed_labels_through", "") < latest
        or getattr(store, "role_refit_required", False)
    ):
        if not store.list("replay_context", 1):
            asyncio.run(import_context(store))
        elif not store.list("context_archive", 1):
            raise ValueError("Historical refitting requires a pinned context archive")
        _, _, features = build_fusion_data(store)
        if (
            spec.get("version") == VERSION
            and spec.get("context_archive_id")
            and spec.get("completed_labels_through", "") >= latest
        ):
            from .fusion import load_data

            old_arrays, old_meta, old_features = load_data(store, spec["feature_id"])
            old_signature = spec.get("input_signature") or input_signature(
                historical_inputs(store, old_arrays, old_meta, old_features, spec["context_archive_id"]),
                old_arrays["minutes"],
                old_meta,
            )
            arrays, meta, _ = load_data(store, features["id"])
            current_signature = input_signature(
                historical_inputs(store, arrays, meta, features, store.get("context_archive")["id"]),
                arrays["minutes"],
                meta,
            )
            if old_signature == current_signature:
                return attach(store, model, spec["id"])
        validation = store.get("role_minutes_validation") if store.list("role_minutes_validation", 1) else {}
        if validation.get("version") != VERSION:
            validation = validate(store)
        if not validation.get("passed"):
            return model
        spec = train(store, features["id"])
    return attach(store, model, spec.get("id"))


def input_signature(matrix, minutes, metadata):
    checksum = hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.float32).tobytes())
    checksum.update(np.ascontiguousarray(minutes, dtype=np.float32).tobytes())
    checksum.update(
        digest([(r["season"], r["event"], r["player_id"], r["fixture_id"]) for r in metadata]).encode()
    )
    return checksum.hexdigest()
