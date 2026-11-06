"""Shrunk player-specific forecast bias, learned only after match completion."""

from __future__ import annotations

import hashlib
import heapq
from collections import defaultdict, deque
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .decision import BAND_MINUTES, PARAMETERS, minute_band
from .storage import atomic_json, digest, now

VERSION = "online-points-bias-v1"
CHOICES = tuple((decay, strength) for decay in (0.8, 0.95) for strength in (4, 12, 32))


class BiasHistory:
    def __init__(self):
        self.players = defaultdict(lambda: deque(maxlen=12))
        self.positions = defaultdict(lambda: deque(maxlen=2000))

    def add(self, row):
        if row["minutes"] < 60:
            return
        residual = float(np.clip(row["residual90"], -8, 8))
        self.players[row["code"]].append(residual)
        self.positions[row["position"]].append(residual)

    def correction(self, code, position, decay, strength):
        values = self.players[code]
        weights = np.array([decay**i for i in range(len(values) - 1, -1, -1)])
        center = float(np.mean(self.positions[position])) if self.positions[position] else 0.0
        return float(np.dot(weights, np.asarray(values) - center) / (weights.sum() + strength))


def reference_rows(store):
    from .decision import canonical_rows
    from .enrichment import epoch
    from .fusion import load_data
    from .immediate_points import conditional
    from .role_minutes import VERSION as ROLE_VERSION
    from .role_minutes import fit_head, historical_inputs, predict

    arrays, meta, features = load_data(store)
    context_id = store.get("context_archive")["id"]
    x_role = historical_inputs(store, arrays, meta, features, context_id)
    originals, _, _ = canonical_rows(store)
    completed = {
        (r["season"], int(r["fixture"])): epoch(r["kickoff_time"]) + 14400
        for r in originals
        if r.get("kickoff_time")
    }
    groups = np.array([r["deadline_group"] for r in meta])
    unique = sorted(set(groups))
    bands = minute_band(arrays["minutes"])
    identity = digest(
        {
            "version": VERSION,
            "feature_id": features["id"],
            "context_id": context_id,
            "parameters": PARAMETERS,
            "choices": CHOICES,
            "reference_role_version": ROLE_VERSION,
        }
    )
    directory = store.root / "models" / ("online-calibration-" + identity[:24])
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for season in ("2024/25", "2025/26", "2026/27"):
        testing = sorted(g for g in unique if g.startswith(season))
        for first in range(0, len(testing), 8):
            block = testing[first : first + 8]
            path = directory / (block[0].replace("/", "-") + ".parquet")
            if path.exists():
                result.extend(pl.read_parquet(path).to_dicts())
                continue
            cut = [g for g in unique if g < block[0]][-8]
            train = np.flatnonzero(groups < cut)
            cal = np.flatnonzero((groups >= cut) & (groups < block[0]))
            test = np.flatnonzero(np.isin(groups, block))
            fit_rows = train[bands[train] > 0]
            points = lgb.LGBMRegressor(**PARAMETERS).fit(
                np.column_stack([arrays["tab"][fit_rows], bands[fit_rows]]), arrays["base_target"][fit_rows]
            )
            head, curves = fit_head(x_role, arrays["minutes"], train, cal)
            p = predict(head, curves, x_role[test], arrays["news"][test])
            c = conditional(points, arrays["tab"][test])
            base = (p * (c + 2 * arrays["dc_prior"][test])).sum(1)
            exposure = p @ (BAND_MINUTES / 90)
            part = []
            for j, i in enumerate(test):
                row = {
                    k: meta[i][k]
                    for k in (
                        "season",
                        "event",
                        "deadline_group",
                        "player_id",
                        "fixture_id",
                        "code",
                        "position",
                        "price_lag",
                        "prior_start",
                        "deadline",
                    )
                }
                row.update(
                    actual=float(arrays["target"][i]),
                    minutes=float(arrays["minutes"][i]),
                    reference=float(base[j]),
                    exposure=float(exposure[j]),
                    known_after=completed[meta[i]["season"], meta[i]["fixture_id"]],
                    residual90=float(
                        (arrays["base_target"][i] - c[j, bands[i]]) * 90 / max(60, arrays["minutes"][i])
                    ),
                )
                part.append(row)
            pl.DataFrame(part).write_parquet(path)
            result.extend(part)
            print("Prepared online calibration reference", block[0], "through", block[-1], flush=True)
    return result, features, context_id, directory


def evaluate(rows):
    from .enrichment import epoch

    history = BiasHistory()
    pending: list[tuple[float, int, dict]] = []
    serial = 0
    groups = defaultdict(list)
    for row in rows:
        groups[row["deadline_group"]].append(row)
    result = []
    for group, part in sorted(groups.items()):
        cutoff = epoch(part[0]["deadline"])
        while pending and pending[0][0] < cutoff:
            _, _, known = heapq.heappop(pending)
            history.add(known)
        for row in part:
            row = dict(row)
            for decay, strength in CHOICES:
                row[f"bias_{decay}_{strength}"] = row["reference"] + row["exposure"] * history.correction(
                    row["code"], row["position"], decay, strength
                )
            result.append(row)
        # No teammate or same-deadline labels enter another player's decision.
        for row in part:
            heapq.heappush(pending, (row["known_after"], serial, row))
            serial += 1
    return result


def validate(store):
    from .immediate_points import interval, metrics
    from .role_minutes import VERSION as ROLE_VERSION

    reference, features, context_id, directory = reference_rows(store)
    rows = evaluate(reference)
    selection = [r for r in rows if r["season"] == "2024/25" and r["event"] <= 19]
    fields = [f"bias_{d}_{s}" for d, s in CHOICES]

    def loss(field):
        return float(
            np.mean(
                [(1 + 4 * float(r["prior_start"] >= 0.6)) * (r[field] - r["actual"]) ** 2 for r in selection]
            )
        )

    chosen = min(fields, key=loss)
    results = {}
    for label, period in (
        ("acceptance_2024", [r for r in rows if r["season"] == "2024/25" and r["event"] > 19]),
        ("replication_2025", [r for r in rows if r["season"] == "2025/26"]),
    ):
        for r in period:
            r["delta"] = (r[chosen] - r["actual"]) ** 2 - (r["reference"] - r["actual"]) ** 2
        results[label] = {
            "rows": len(period),
            "reference": metrics(period, "reference"),
            "candidate": metrics(period, chosen),
            "mse_delta_95ci": interval(period, "delta"),
        }
    forecast = all(
        p["mse_delta_95ci"][1] < 0
        and p["candidate"]["regular_rmse"] <= p["reference"]["regular_rmse"]
        and p["candidate"]["premium_rmse"] <= p["reference"]["premium_rmse"]
        for p in results.values()
    )
    captain = all(
        p["candidate"]["captain_points"] >= p["reference"]["captain_points"] for p in results.values()
    )
    path = directory / "predictions.parquet"
    pl.DataFrame(rows).write_parquet(path)
    report = {
        "at": now(),
        "version": VERSION,
        "reference_role_version": ROLE_VERSION,
        "reference_decision_model_id": store.get("decision_model")["id"],
        "feature_id": features["id"],
        "context_id": context_id,
        "selected": chosen,
        "selection_losses": {f: loss(f) for f in fields},
        "metrics": results,
        "forecast_passed": bool(forecast),
        "captain_passed": bool(captain),
        "passed": bool(forecast and captain),
        "prediction_path": str(path),
        "prediction_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "protocol": "Player bias from earlier out-of-fold conditional-points errors after match completion; clip at eight points/90, center by position and shrink to zero. Decay and shrinkage selected on 2024/25 GW1-19 only. Acceptance GW20-38; separate 2025/26 replication. Current-season results are diagnostic only.",
    }
    report["id"] = store.put("online_calibration_validation", report)
    atomic_json(Path("artifacts/evaluations/online_calibration_validation.json"), report)
    return report
