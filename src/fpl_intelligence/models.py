"""Chronological probability calibration and production minutes gates.

All model and calibration fitting precedes the held-out groups. Artifacts are
LightGBM text and JSON, never executable pickle objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from .features import FEATURE_NAMES
from .storage import Store, atomic_json, digest, now

PARAMS: dict[str, Any] = {
    "n_estimators": 120,
    "num_leaves": 12,
    "min_child_samples": 60,
    "learning_rate": 0.04,
    "reg_lambda": 5,
    "verbosity": -1,
    "n_jobs": 1,
    "random_state": 47,
}


def reliability(y, p) -> dict:
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "n": len(y),
    }


def curve_fit(prediction, actual):
    if len(set(actual)) < 2:
        return {"x": [0.0, 1.0], "y": [float(np.mean(actual))] * 2}
    model = IsotonicRegression(out_of_bounds="clip").fit(prediction, actual)
    return {"x": model.X_thresholds_.tolist(), "y": model.y_thresholds_.tolist()}


def curve_apply(values, curve):
    return np.interp(values, curve["x"], curve["y"])


def bands(minutes):
    return np.select([minutes <= 0, minutes < 60, minutes < 90], [0, 1, 2], default=3).astype(int)


def fit_model_suite(store: Store) -> dict:
    feature = store.get("features")
    frame = pl.read_parquet(store.artifact_path(feature["path"]))
    groups = sorted(frame["deadline_group"].unique().to_list())
    if len(groups) < 12:
        raise ValueError("Calibrated production models need at least twelve chronological deadline groups")
    train_end = groups[int(len(groups) * 0.6)]
    cal_end = groups[int(len(groups) * 0.8)]
    protocol = {
        "version": "calibrated-minutes-v1",
        "train_before": train_end,
        "calibrate_before": cal_end,
        "test_from": cal_end,
        "features": FEATURE_NAMES,
        "parameters": PARAMS,
        "promotion": "start Brier improves and availability Brier does not worsen on held-out groups",
    }
    protocol_id = store.put("protocol", protocol)
    training = frame.filter(pl.col("deadline_group") < train_end)
    calibration = frame.filter((pl.col("deadline_group") >= train_end) & (pl.col("deadline_group") < cal_end))
    test = frame.filter(pl.col("deadline_group") >= cal_end)

    def x(data):
        return data.select(FEATURE_NAMES).to_numpy()

    start = lgb.LGBMClassifier(**PARAMS)
    minute = lgb.LGBMClassifier(**PARAMS)
    start.fit(x(training), training["start_target"].to_numpy())
    minute.fit(x(training), bands(training["minutes_target"].to_numpy()))
    cal_start = np.asarray(start.predict_proba(x(calibration)))[:, 1]
    cal_bands = np.asarray(minute.predict_proba(x(calibration)))
    if minute.classes_.tolist() != [0, 1, 2, 3]:
        raise ValueError("Minutes training lacks one of the four required exposure bands")
    curves = {"start": curve_fit(cal_start, calibration["start_target"].to_numpy())}
    for k in range(4):
        curves[str(k)] = curve_fit(
            cal_bands[:, k], (bands(calibration["minutes_target"].to_numpy()) == k).astype(int)
        )
    predicted_start = curve_apply(np.asarray(start.predict_proba(x(test)))[:, 1], curves["start"])
    raw_bands = np.asarray(minute.predict_proba(x(test)))
    calibrated = np.column_stack([curve_apply(raw_bands[:, k], curves[str(k)]) for k in range(4)])
    calibrated /= np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-9)
    prior_start = test["p_start"].to_numpy()
    prior_any = prior_start + (1 - prior_start) * test["p_cameo"].to_numpy()
    truth_start = test["start_target"].to_numpy()
    truth_any = (test["minutes_target"].to_numpy() > 0).astype(int)
    estimated_any = 1 - calibrated[:, 0]
    predicted_start = np.minimum(predicted_start, estimated_any)
    baseline = {
        "start": reliability(truth_start, prior_start),
        "appearance": reliability(truth_any, prior_any),
    }
    candidate = {
        "start": reliability(truth_start, predicted_start),
        "appearance": reliability(truth_any, estimated_any),
    }
    promoted = (
        candidate["start"]["brier"] < baseline["start"]["brier"]
        and candidate["appearance"]["brier"] <= baseline["appearance"]["brier"]
    )
    train_minutes = training["minutes_target"].to_numpy()
    band_means = [float(train_minutes[bands(train_minutes) == k].mean()) for k in range(4)]
    directory = store.root / "models" / digest({"protocol": protocol, "feature_hash": feature["hash"]})[:16]
    directory.mkdir(parents=True, exist_ok=True)
    start.booster_.save_model(str(directory / "start.txt"))
    minute.booster_.save_model(str(directory / "minutes.txt"))
    # The held-out promotion gate is preserved. Production refitting is separate,
    # and calibration remains based exclusively on chronological validation data.
    report = {
        "at": now(),
        "protocol_id": protocol_id,
        "feature_id": feature["id"],
        "training_rows": len(training),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "baseline": baseline,
        "candidate": candidate,
        "promoted": bool(promoted),
        "curves": curves,
        "band_means": band_means,
        "start_model": str(directory / "start.txt"),
        "minutes_model": str(directory / "minutes.txt"),
        "model_files_hashes": {
            name: __import__("hashlib").sha256((directory / name).read_bytes()).hexdigest()
            for name in ("start.txt", "minutes.txt")
        },
        "track": "chronological historical statistical replay",
        "test_deadlines": test["deadline_group"].unique().sort().to_list(),
    }
    report["id"] = store.put("model_suite", report)
    atomic_json(
        Path("artifacts/evaluations/calibrated_models.json"),
        {k: v for k, v in report.items() if k not in ("curves",)},
    )
    return report


def artifact_location(store: Store, value: str) -> Path:
    # Suite directories are content-addressed underneath models, portable on restore.
    path = Path(value)
    return store.root / "models" / path.parent.name / path.name


class MinutesModel:
    def __init__(self, store: Store):
        self.spec = store.get("model_suite")
        if not self.spec["promoted"]:
            raise ValueError("Calibrated challenger did not pass its promotion gate")
        import hashlib

        for name, reference in [
            ("start.txt", self.spec["start_model"]),
            ("minutes.txt", self.spec["minutes_model"]),
        ]:
            if (
                hashlib.sha256(artifact_location(store, reference).read_bytes()).hexdigest()
                != self.spec["model_files_hashes"][name]
            ):
                raise ValueError("Calibrated model integrity verification failed")
        self.start = lgb.Booster(model_file=str(artifact_location(store, self.spec["start_model"])))
        self.minute = lgb.Booster(model_file=str(artifact_location(store, self.spec["minutes_model"])))

    def predict(self, matrix):
        minute = np.asarray(self.minute.predict(matrix, num_threads=1))
        calibrated = np.column_stack(
            [curve_apply(minute[:, k], self.spec["curves"][str(k)]) for k in range(4)]
        )
        calibrated /= np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-9)
        pany = 1 - calibrated[:, 0]
        p60 = calibrated[:, 2] + calibrated[:, 3]
        start = np.minimum(
            pany, curve_apply(self.start.predict(matrix, num_threads=1), self.spec["curves"]["start"])
        )
        emins = calibrated @ np.array(self.spec["band_means"])
        cameo = np.clip((pany - start) / np.maximum(1 - start, 1e-6), 0, 1)
        conditional = np.clip((emins - 20 * (pany - start)) / np.maximum(start, 1e-6), 45, 90)
        return [
            {
                "p_start": float(np.clip(a, 0.001, 0.999)),
                "p_cameo": float(c),
                "p_long_start": float(np.clip(si / max(a, 1e-6), 0, 1)),
                "start_minutes": float(m),
            }
            for a, c, si, m in zip(start, cameo, p60, conditional)
        ]


def fit_ensemble(store: Store) -> dict:
    from .evaluation import metrics, paired_interval

    evaluation = store.get("evaluation")
    records = evaluation["predictions"]
    groups = sorted({r["deadline_group"] for r in records})
    cut = groups[int(len(groups) * 0.6)]
    fitting = [r for r in records if r["deadline_group"] < cut]
    testing = [r for r in records if r["deadline_group"] >= cut]
    b = np.array([r["baseline"] for r in fitting])
    d = np.array([r["challenger"] for r in fitting])
    y = np.array([r["actual"] for r in fitting])
    weight = float(np.clip(np.dot(d - b, y - b) / (np.dot(d - b, d - b) + 100), 0, 1))
    truth = np.array([r["actual"] for r in testing])
    prior = np.array([r["baseline"] for r in testing])
    direct = np.array([r["challenger"] for r in testing])
    mixture = (1 - weight) * prior + weight * direct
    comparator = direct if metrics(truth, direct)["rmse"] <= metrics(truth, prior)["rmse"] else prior
    interval = paired_interval(
        [{**r, "baseline": float(p), "challenger": float(q)} for r, p, q in zip(testing, comparator, mixture)]
    )
    result = {
        "at": now(),
        "version": "regularized-oof-mixture-v1",
        "evaluation_id": evaluation["id"],
        "training_oof_rows": len(fitting),
        "test_rows": len(testing),
        "direct_weight": weight,
        "mixture": metrics(truth, mixture),
        "baseline": metrics(truth, prior),
        "direct": metrics(truth, direct),
        "paired_rmse_difference_95ci": interval,
        "promoted": bool(interval[1] < 0),
        "scope": "Weights fit on earlier out-of-fold predictions; tested on later deadlines",
        "production_effect": "Only used if the held-out promotion gate passes",
    }
    result["id"] = store.put("ensemble", result)
    atomic_json(Path("artifacts/evaluations/ensemble.json"), result)
    return result
