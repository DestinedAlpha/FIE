"""Check the served numerical forecast against the worlds the planner consumes."""

from __future__ import annotations

import numpy as np


def validate_forecast(forecast, arrays):
    ids = forecast["player_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Forecast contains duplicate player identities")
    index = {p: i for i, p in enumerate(ids)}
    records = {(r["event"], r["player_id"]): r for r in forecast["players"]}
    expected = {
        (event, pid) for event in range(forecast["start_event"], forecast["end_event"] + 1) for pid in ids
    }
    if len(records) != len(forecast["players"]) or records.keys() != expected:
        raise ValueError("Forecast market/event coverage is incomplete or duplicated")
    fixtures = {}
    for row in forecast["fixtures"]:
        key = (row["event"], row["player_id"], row["fixture_id"])
        if key in fixtures:
            raise ValueError("Forecast contains a duplicate player-fixture")
        fixtures[key] = row
        if not np.isfinite([row["mean"], row["minutes"], row["p_zero"], row["p_60"]]).all():
            raise ValueError("Nonfinite fixture forecast")
        if not (0 <= row["p_60"] <= 1 - row["p_zero"] + 1e-9 and 0 <= row["minutes"] <= 90):
            raise ValueError("Invalid fixture participation forecast")
    for event in range(forecast["start_event"], forecast["end_event"] + 1):
        points, minutes = arrays[f"points_{event}"], arrays[f"minutes_{event}"]
        shape = (forecast["worlds"], len(ids))
        if (
            points.shape != shape
            or minutes.shape != shape
            or not np.isfinite(points).all()
            or not np.isfinite(minutes).all()
        ):
            raise ValueError("Invalid forecast world dimensions or nonfinite draws")
        if np.any(points[minutes == 0] != 0) or np.any(minutes < 0):
            raise ValueError("Forecast points contradict player participation")
        for pid, col in index.items():
            row = records[event, pid]
            values = [row[k] for k in ("mean", "sd", "minutes_mean", "p_zero", "p_start", "p_60", "p_haul")]
            if not np.isfinite(values).all() or row["sd"] < 0:
                raise ValueError("Nonfinite or invalid numerical forecast")
            if any(not 0 <= row[k] <= 1 for k in ("p_zero", "p_start", "p_60", "p_haul")):
                raise ValueError("Invalid forecast probability")
            if row["p_60"] > 1 - row["p_zero"] + 1e-9:
                raise ValueError("Forecast participation probabilities contradict each other")
            for field, expected_value in [
                ("mean", points[:, col].mean()),
                ("sd", points[:, col].std()),
                ("minutes_mean", minutes[:, col].mean()),
                ("p_zero", (minutes[:, col] == 0).mean()),
            ]:
                if not np.isclose(row[field], expected_value, atol=1e-7):
                    raise ValueError("Forecast summary differs from optimizer worlds: " + field)
            if np.any(minutes[:, col] > 90 * row["fixture_count"]):
                raise ValueError("Forecast minutes exceed scheduled match exposure")
            for name in ("point_quantiles", "minute_quantiles"):
                q = np.asarray(row[name])
                if not np.isfinite(q).all() or np.any(np.diff(q) < 0):
                    raise ValueError("Forecast quantiles are invalid")
            if abs(sum(row["components"].values()) - row["mean"]) > 0.0003:
                raise ValueError("Forecast scoring components do not reconcile")
    return {
        "passed": True,
        "player_gameweeks": len(records),
        "player_fixtures": len(fixtures),
        "scope": "Finite probabilities, physical minute bounds, complete coverage and exact reconciliation with optimizer worlds; statistical accuracy is evaluated separately",
    }
