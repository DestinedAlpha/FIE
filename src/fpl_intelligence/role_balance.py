"""Research-only team-minute reconciliation; never silently alters accepted heads."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from .decision import BAND_MINUTES, NAMES, fit_components, minute_band, predict_models
from .fusion import load_data, published_availability
from .role_minutes import delta_interval, fit_head, historical_inputs, predict
from .storage import atomic_json, now


def reconcile(probabilities, metadata, band_minutes=BAND_MINUTES):
    """Minimum relative-entropy tilting; unavailable classes remain impossible."""
    result = probabilities.copy()
    groups = defaultdict(list)
    for i, row in enumerate(metadata):
        groups[row["season"], row["event"], row["fixture_id"], row["team"], row["position"] == 1].append(i)
    for key, indices in groups.items():
        if len(indices) < (2 if key[-1] else 15):
            continue  # Incomplete registered cohorts cannot establish a team constraint.
        p = probabilities[indices]
        target = 90.0 if key[-1] else 900.0
        supported = p > 0
        low = np.where(supported, band_minutes, np.inf).min(1).sum()
        high = np.where(supported, band_minutes, -np.inf).max(1).sum()
        if not low < target < high:
            continue  # Never invent an available substitute to repair a missing registry.
        logits = np.full(p.shape, -np.inf)
        np.log(p, out=logits, where=supported)
        left, right = -1.0, 1.0
        for _ in range(45):
            tilt = (left + right) / 2
            shifted = logits + tilt * band_minutes
            adjusted = np.exp(shifted - logsumexp(shifted, axis=1, keepdims=True))
            if float(np.sum(adjusted @ band_minutes)) < target:
                left = tilt
            else:
                right = tilt
        result[indices] = adjusted
    return result


def validate(store):
    """Compare to accepted role mixtures on earlier seasons; current GW scores never select this."""
    before = store.get("decision_model")["id"]
    role = store.get("role_minutes_model", store.get("decision_model")["role_minutes_model_id"])
    arrays, metadata, features = load_data(store, role["feature_id"])
    x = historical_inputs(store, arrays, metadata, features, role["context_archive_id"])
    groups = np.array([r["deadline_group"] for r in metadata])
    unique = sorted(set(groups))
    frame = pl.DataFrame(arrays["tab"], schema=NAMES).with_columns(
        pl.Series("minutes_target", arrays["minutes"]),
        pl.Series(
            "start_target",
            [float(v) if known else None for v, known in zip(arrays["start"], arrays["start_mask"])],
        ),
        pl.Series("base_target", arrays["base_target"]),
    )
    rows = []
    for season in ("2024/25", "2025/26"):
        testing = sorted(g for g in unique if g.startswith(season))
        for first in range(0, len(testing), 8):
            block = testing[first : first + 8]
            cut = [g for g in unique if g < block[0]][-8]
            tr = np.flatnonzero(groups < cut)
            ca = np.flatnonzero((groups >= cut) & (groups < block[0]))
            te = np.flatnonzero(np.isin(groups, block))
            with threadpool_limits(limits=1):
                core = fit_components(frame[tr], frame[ca])
                _, rp, conditional = predict_models(core, arrays["tab"][te], arrays["dc_prior"][te])
                rp[:, 1:] *= np.array([published_availability(v) for v in arrays["news"][te]])[:, None]
                rp[:, 0] = 1 - rp[:, 1:].sum(1)
                head, curves = fit_head(x, arrays["minutes"], tr, ca)
                p = (1 - role["weight"]) * rp + role["weight"] * predict(
                    head, curves, x[te], arrays["news"][te]
                )
            meta = [metadata[i] for i in te]
            # Band exposure is estimated from earlier calibration minutes only.
            cal_bands = minute_band(arrays["minutes"][ca])
            band_minutes = np.array([np.mean(arrays["minutes"][ca][cal_bands == k]) for k in range(4)])
            adjusted = reconcile(p, meta, band_minutes)
            truth = minute_band(arrays["minutes"][te])
            values = conditional + 2 * arrays["dc_prior"][te]
            for j, i in enumerate(te):
                item = {
                    **metadata[i],
                    "group": groups[i],
                    "actual": float(arrays["target"][i]),
                    "premium": metadata[i]["price_lag"] >= 90,
                    "regular": metadata[i]["prior_start"] >= 0.6,
                }
                for weight in (0, 0.25, 0.5, 1):
                    probability = (1 - weight) * p[j] + weight * adjusted[j]
                    mean = float(probability @ values[j])
                    item[str(weight)] = {
                        "brier": float(((probability - np.eye(4)[truth[j]]) ** 2).sum()),
                        "points_squared": float((mean - item["actual"]) ** 2),
                        "mean": mean,
                    }
                rows.append(item)
            print("Team-minute constraint", block[0], "through", block[-1], flush=True)
    selection = [r for r in rows if r["season"] == "2024/25" and r["event"] <= 19]
    weight = min((0, 0.25, 0.5, 1), key=lambda w: np.mean([r[str(w)]["points_squared"] for r in selection]))
    metrics: dict[str, Any] = {}
    passed = weight > 0
    for name, period in [
        ("acceptance_2024", [r for r in rows if r["season"] == "2024/25" and r["event"] > 19]),
        ("replication_2025", [r for r in rows if r["season"] == "2025/26"]),
    ]:
        result: dict[str, Any] = {
            "rows": len(period),
            "delta_95ci": {k: delta_interval(period, k, weight) for k in ("brier", "points_squared")},
        }
        for label, w in [("reference", 0), ("candidate", weight)]:
            summary = {k: float(np.mean([r[str(w)][k] for r in period])) for k in ("brier", "points_squared")}
            for subset in ("regular", "premium"):
                summary[subset + "_points_squared"] = float(
                    np.mean([r[str(w)]["points_squared"] for r in period if r[subset]])
                )
            weekly: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
            for r in period:
                predicted, actual = weekly[r["group"]].get(r["player_id"], (0, 0))
                weekly[r["group"]][r["player_id"]] = predicted + r[str(w)]["mean"], actual + r["actual"]
            summary["captain_points"] = sum(
                max(v.items(), key=lambda item: (item[1][0], -item[0]))[1][1] for v in weekly.values()
            )
            result[label] = summary
        passed &= (
            result["delta_95ci"]["points_squared"][1] < 0
            and result["delta_95ci"]["brier"][1] <= 0
            and all(
                result["candidate"][k] <= result["reference"][k]
                for k in ("regular_points_squared", "premium_points_squared")
            )
            and result["candidate"]["captain_points"] >= result["reference"]["captain_points"]
        )
        metrics[name] = result
    report = {
        "at": now(),
        "hypothesis": "Reconcile club minute mass without forcing specific players or reviving known absences",
        "feature_id": features["id"],
        "role_model_id": role["id"],
        "selected_weight": weight,
        "metrics": metrics,
        "forecast_gate_passed": bool(passed),
        "promoted": False,
        "production_unchanged": store.get("decision_model")["id"] == before,
        "scope": "Earlier-season chronological reconstruction, selection 2024 GW1–19; acceptance GW20–38; 2025 replication. Inspected archives, not untouched evidence. Any promotion additionally requires matched continuous-policy acceptance.",
    }
    report["id"] = store.put("role_balance_validation", report)
    atomic_json(Path("artifacts/evaluations/role_balance_validation.json"), report)
    return report
