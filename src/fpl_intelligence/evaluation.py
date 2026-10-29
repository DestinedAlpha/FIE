"""Predeclared, deadline-grouped evaluation and immutable model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import FEATURE_NAMES
from .storage import Store, atomic_json, now

PROTOCOL: dict[str, Any] = {
    "version": "walk-forward-v1",
    "primary_metric": "RMSE",
    "minimum_train_events": 3,
    "split_unit": "gameweek deadline",
    "feature_cutoff": "strictly earlier event",
    "hyperparameters": {
        "n_estimators": 100,
        "num_leaves": 12,
        "learning_rate": 0.04,
        "min_child_samples": 45,
        "reg_lambda": 5,
        "verbosity": -1,
        "n_jobs": 1,
    },
    "promotion": "paired deadline bootstrap upper 95% RMSE difference < 0",
    "seeds": [11, 29, 47],
    "track": "historical_statistical_replay",
    "prospective_claims": False,
}


def metrics(y, p) -> dict:
    return {
        "n": len(y),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "bias": float(np.mean(p - y)),
    }


def paired_interval(rows: list[dict], seed=11) -> list[float]:
    rng = np.random.default_rng(seed)
    groups = sorted({r.get("deadline_group", str(r["event"])) for r in rows})
    data = {g: [r for r in rows if r.get("deadline_group", str(r["event"])) == g] for g in groups}
    differences = []
    for _ in range(500):
        sampled = [r for g in rng.choice(groups, len(groups), replace=True) for r in data[g]]
        y = np.array([r["actual"] for r in sampled])
        a = np.array([r["challenger"] for r in sampled])
        b = np.array([r["baseline"] for r in sampled])
        differences.append(np.sqrt(np.mean((a - y) ** 2)) - np.sqrt(np.mean((b - y) ** 2)))
    return np.quantile(differences, [0.025, 0.975]).tolist()


def train_evaluate(store: Store) -> dict:
    feature_artifact = store.get("features")
    frame = pl.read_parquet(store.artifact_path(feature_artifact["path"]))
    # Persist protocol BEFORE accessing held-out labels.
    protocol_id = store.put("protocol", PROTOCOL)
    events = sorted(frame["deadline_group"].unique().to_list())
    predictions = []
    for event in events[PROTOCOL["minimum_train_events"] :]:
        train = frame.filter(pl.col("deadline_group") < event)
        test = frame.filter(pl.col("deadline_group") == event)
        estimates = []
        for seed in PROTOCOL["seeds"]:
            model = lgb.LGBMRegressor(**PROTOCOL["hyperparameters"], random_state=seed)
            model.fit(train.select(FEATURE_NAMES).to_numpy(), train["target"].to_numpy())
            estimates.append(np.maximum(0, model.predict(test.select(FEATURE_NAMES).to_numpy())))
        predicted = np.mean(estimates, axis=0)
        for row, estimate in zip(test.to_dicts(), predicted):
            predictions.append(
                {
                    "event": row["event"],
                    "deadline_group": event,
                    "season": row["season"],
                    "player_id": row["player_id"],
                    "position": int(row["position"]),
                    "actual": row["target"],
                    "baseline": row["baseline"],
                    "challenger": float(estimate),
                    "prior_start": row["p_start"],
                    "started": row["start_target"],
                    "minutes": row["minutes_target"],
                }
            )
    if not predictions:
        raise ValueError("Need at least four completed events for walk-forward evaluation")
    y = np.array([p["actual"] for p in predictions])
    baseline = np.array([p["baseline"] for p in predictions])
    direct = np.array([p["challenger"] for p in predictions])
    interval = paired_interval(predictions)
    promoted = interval[1] < 0
    by_position = {}
    for pos in range(1, 5):
        mask = np.array([p["position"] == pos for p in predictions])
        if mask.any():
            by_position[str(pos)] = {
                "baseline": metrics(y[mask], baseline[mask]),
                "challenger": metrics(y[mask], direct[mask]),
            }
    relevant = np.array([p["prior_start"] >= 0.6 for p in predictions])
    p_start = np.clip([r["prior_start"] for r in predictions], 0.001, 0.999)
    starts = np.array([r["started"] for r in predictions])
    output = {
        "created_at": now(),
        "protocol_id": protocol_id,
        "feature_id": feature_artifact["id"],
        "track": PROTOCOL["track"],
        "events": sorted({p["event"] for p in predictions}),
        "baseline": metrics(y, baseline),
        "challenger": metrics(y, direct),
        "paired_rmse_difference_95ci": interval,
        "by_position": by_position,
        "regular_starters": {
            "baseline": metrics(y[relevant], baseline[relevant]),
            "challenger": metrics(y[relevant], direct[relevant]),
        }
        if relevant.any()
        else None,
        "start_brier": float(np.mean((p_start - starts) ** 2)),
        "start_log_loss": float(-np.mean(starts * np.log(p_start) + (1 - starts) * np.log(1 - p_start))),
        "promoted": promoted,
        "champion": "direct" if promoted else "shrinkage",
        "limitations": [
            "Completed-match statistical replay, not frozen pre-deadline news",
            "Prospective competitive strength requires future deadline outcomes",
        ],
        "predictions": predictions,
    }
    directory = store.root / "models"
    directory.mkdir(exist_ok=True)
    model = lgb.LGBMRegressor(**PROTOCOL["hyperparameters"], random_state=11)
    model.fit(frame.select(FEATURE_NAMES).to_numpy(), frame["target"].to_numpy())
    model.booster_.save_model(str(directory / "direct.txt"))
    output["model_path"] = str(directory / "direct.txt")
    output["id"] = store.put("evaluation", output)
    atomic_json(
        Path("artifacts/evaluations/walk_forward.json"),
        {k: v for k, v in output.items() if k != "predictions"},
    )
    return output


def evaluate_frozen(store: Store) -> dict:
    """Score only frozen predictions captured before their actual event deadline."""
    from .features import histories
    from .ingestion import load_market

    players, market = load_market(store)
    actuals = histories(store, players)
    deadlines = {e["id"]: e["deadline_time"] for e in market["events"]}
    finished = {e["id"] for e in market["events"] if e.get("data_checked")}
    records = []
    from datetime import datetime

    for entry in store.list("forecast", 1000):
        forecast = store.get("forecast", entry["id"])
        issued = datetime.fromisoformat(forecast["issued_at"])
        for row in forecast["players"]:
            event = row["event"]
            if event not in finished or issued >= datetime.fromisoformat(
                deadlines[event].replace("Z", "+00:00")
            ):
                continue
            actual = sum(r["total_points"] for r in actuals[row["player_id"]] if r["round"] == event)
            records.append(
                {
                    "forecast_id": entry["id"],
                    "event": event,
                    "player_id": row["player_id"],
                    "actual": actual,
                    "predicted": row["mean"],
                }
            )
    result = {
        "at": now(),
        "eligible_predictions": len(records),
        "track": "prospective_frozen",
        "metrics": metrics(
            np.array([r["actual"] for r in records]), np.array([r["predicted"] for r in records])
        )
        if records
        else None,
    }
    result["id"] = store.put("prospective", result)
    return result
