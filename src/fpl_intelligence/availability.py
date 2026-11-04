"""Learned official recovery transitions and conditional congestion/minutes dynamics."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss

from .storage import atomic_json, digest, now

RECOVERY_FEATURES = (
    "chance",
    "chance_known",
    "injured",
    "doubtful",
    "report_age_days",
    "injury_snapshots",
    "gap_days",
    "position",
)
ROTATION_FEATURES = (
    "position",
    "rest_days",
    "recent_start_rate",
    "recent_minutes",
    "last_minutes",
    "last_started",
    "short_rest_load",
)


def block_interval(values, groups, seed=761):
    collected = defaultdict(list)
    for value, group in zip(values, groups):
        collected[group].append(float(value))
    x = np.array([np.mean(v) for _, v in sorted(collected.items())])
    n = len(x)
    if n < 2:
        return [float(x.mean()), float(x.mean())]
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(2000, (n + 3) // 4))
    idx = ((starts[:, :, None] + np.arange(4)) % n).reshape(2000, -1)[:, :n]
    return np.quantile(x[idx].mean(1), [0.025, 0.975]).tolist()


def moment(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def recovery_features(packet, at, gap_days, streak):
    chance = packet.get("chance_of_playing_next_round")
    added = packet.get("news_added")
    age = (
        max(0, (moment(at) - moment(added)).total_seconds() / 86400)
        if added and moment(added) <= moment(at)
        else 0
    )
    return [
        float(chance) / 100 if chance is not None else 0.5,
        float(chance is not None),
        float(packet.get("status") == "i"),
        float(packet.get("status") == "d"),
        min(age, 365),
        streak,
        min(gap_days, 30),
        float(packet["element_type"]),
    ]


def rotation_features(position, rest, recent, last_minutes, last_started):
    return [
        float(position),
        min(30, max(0, rest)),
        float(np.mean([r.get("starts") or 0 for r in recent])) if recent else 0.5,
        float(np.mean([r.get("minutes") or 0 for r in recent])) if recent else 45,
        last_minutes,
        last_started,
        last_minutes * max(0, 6 - rest) / 6,
    ]


def train_availability(store):
    from .context_archive import load_context
    from .decision import canonical_rows, minute_band, player_key

    packets = load_context(store)
    context = store.get("context_archive")
    dates = {(r["season"], r["event"]): r["deadline"] for r in context["entries"]}
    by_code = defaultdict(list)
    for (season, event, code), packet in packets.items():
        by_code[season, code].append((event, packet))
    examples = []
    for (season, _), values in by_code.items():
        values.sort()
        streak = 0
        for (event, p), (next_event, target) in zip(values, values[1:]):
            streak = streak + 1 if p.get("status") in ("i", "d") else 0
            if not streak or next_event != event + 1 or target.get("status") not in ("a", "i", "d"):
                continue
            at, next_at = dates[season, event], dates[season, next_event]
            gap = (moment(next_at) - moment(at)).total_seconds() / 86400
            if gap <= 0 or gap > 30:
                continue
            examples.append(
                {
                    "group": f"{season}-GW{next_event:02d}",
                    "x": recovery_features(p, at, gap, streak),
                    "y": int(target["status"] == "a"),
                    "prior": float(1 - np.exp(-gap / 35)),
                }
            )
    train = [r for r in examples if r["group"] < "2024/25-GW25"]
    cal = [r for r in examples if "2024/25-GW25" <= r["group"] < "2025/26-GW01"]
    test = [r for r in examples if r["group"] >= "2025/26-GW01"]
    if not all((train, cal, test)):
        raise ValueError("Recovery requires disjoint earlier train/calibration/test deadlines")
    params: dict[str, Any] = dict(
        n_estimators=100,
        num_leaves=12,
        min_child_samples=40,
        reg_lambda=20,
        learning_rate=0.04,
        n_jobs=1,
        verbosity=-1,
        random_state=761,
    )
    recovery = lgb.LGBMClassifier(**params).fit([r["x"] for r in train], [r["y"] for r in train])

    def probs(model, items):
        return np.clip(np.asarray(model.predict_proba([r["x"] for r in items]))[:, 1], 1e-5, 1 - 1e-5)

    cp = probs(recovery, cal)
    cy = np.array([r["y"] for r in cal])
    cb = np.array([r["prior"] for r in cal])
    weight = min(
        (0, 0.25, 0.5, 0.75, 1),
        key=lambda w: log_loss(cy, np.clip(w * cp + (1 - w) * cb, 1e-5, 1 - 1e-5), labels=[0, 1]),
    )
    ty = np.array([r["y"] for r in test])
    base = np.array([r["prior"] for r in test])
    p = weight * probs(recovery, test) + (1 - weight) * base

    def loss(p):
        return -(ty * np.log(np.clip(p, 1e-6, 1)) + (1 - ty) * np.log(np.clip(1 - p, 1e-6, 1)))

    ci = block_interval(loss(p) - loss(base), [r["group"] for r in test])
    recovery_report = {
        "training_rows": len(train),
        "calibration_rows": len(cal),
        "test_rows": len(test),
        "weight": weight,
        "log_loss": float(loss(p).mean()),
        "baseline_log_loss": float(loss(base).mean()),
        "block_difference_95ci": ci,
        "promoted": bool(weight > 0 and ci[1] < 0),
        "target": "Transition from published injured/doubtful status to published available status; not an inferred medical diagnosis",
    }
    rows, _, _ = canonical_rows(store)
    past: dict[tuple, list[dict]] = defaultdict(list)
    rotation = []
    for r in rows:
        key = player_key(r)
        recent = past[key]
        if recent and r.get("kickoff_time") and recent[-1].get("kickoff_time"):
            previous = recent[-1]
            rest = (moment(r["kickoff_time"]) - moment(previous["kickoff_time"])).total_seconds() / 86400
            if 0 < rest <= 10:
                rotation.append(
                    {
                        "group": f"{r['season']}-GW{int(r['round']):02d}",
                        "x": rotation_features(
                            r["position"],
                            rest,
                            recent[-8:],
                            float(previous.get("minutes") or 0),
                            float(previous.get("starts") or 0),
                        ),
                        "y": int(minute_band(np.array([float(r.get("minutes") or 0)]))[0]),
                    }
                )
        past[key].append(r)
        past[key] = past[key][-8:]
    tr = [r for r in rotation if r["group"] < "2023/24-GW01"]
    ca = [r for r in rotation if "2023/24-GW01" <= r["group"] < "2024/25-GW01"]
    te = [r for r in rotation if r["group"] >= "2024/25-GW01"]
    models = {}
    directory = (
        store.root
        / "models"
        / ("availability-v4-" + digest({"context": context["id"], "training": tr, "calibration": ca})[:18])
    )
    directory.mkdir(parents=True, exist_ok=True)
    for name, width in (("rotation", len(ROTATION_FEATURES)), ("rotation_baseline", 4)):
        models[name] = lgb.LGBMClassifier(**params).fit([r["x"][:width] for r in tr], [r["y"] for r in tr])
    ca_y = np.array([r["y"] for r in ca])
    te_y = np.array([r["y"] for r in te])
    cp = np.asarray(models["rotation"].predict_proba([r["x"] for r in ca]))
    cb = np.asarray(models["rotation_baseline"].predict_proba([r["x"][:4] for r in ca]))
    rw = min(
        (0, 0.25, 0.5, 0.75, 1), key=lambda w: log_loss(ca_y, w * cp + (1 - w) * cb, labels=[0, 1, 2, 3])
    )
    tp = np.asarray(models["rotation"].predict_proba([r["x"] for r in te]))
    tb = np.asarray(models["rotation_baseline"].predict_proba([r["x"][:4] for r in te]))
    mixed = rw * tp + (1 - rw) * tb
    difference = -np.log(np.clip(mixed[np.arange(len(te)), te_y], 1e-8, 1)) + np.log(
        np.clip(tb[np.arange(len(te)), te_y], 1e-8, 1)
    )
    rci = block_interval(difference, [r["group"] for r in te])
    rotation_report = {
        "training_rows": len(tr),
        "calibration_rows": len(ca),
        "test_rows": len(te),
        "weight": rw,
        "log_loss": float(log_loss(te_y, mixed, labels=[0, 1, 2, 3])),
        "baseline_log_loss": float(log_loss(te_y, tb, labels=[0, 1, 2, 3])),
        "block_difference_95ci": rci,
        "promoted": bool(rw > 0 and rci[1] < 0),
        "scope": "Conditional next-match minute-band test with earlier calibrated fit; production conditions on simulated previous minutes, never future actual minutes",
    }
    rotation_report["joint_validation"] = validate_joint_rotation(rows, models, store)
    models["recovery"] = recovery
    saved = {}
    for name, model in models.items():
        path = directory / (name + ".txt")
        model.booster_.save_model(str(path))
        saved[name] = {"path": str(path), "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {
        "at": now(),
        "context_id": context["id"],
        "models": saved,
        "recovery": recovery_report,
        "rotation": rotation_report,
        "features": {"recovery": RECOVERY_FEATURES, "rotation": ROTATION_FEATURES},
        "protocol": "Recovery: fit before 2024/25 GW25, calibrate that season remainder, test 2025 onward. Rotation: 2022/23 fit, 2023/24 calibrate, 2024/25 onward test; four-deadline block intervals and independent joint-DGW gate.",
    }
    report["id"] = store.put("availability_model", report)
    atomic_json(Path("artifacts/evaluations/availability.json"), report)
    return report


def load_availability(store):
    spec = store.get("availability_model")
    models = {}
    for name, item in spec["models"].items():
        path = store.artifact_path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["hash"]:
            raise ValueError("Availability model checksum mismatch")
        models[name] = lgb.Booster(model_file=str(path))
    return spec, models


def recovery_path(packet, at, days, loaded):
    spec, models = loaded
    chance = packet.get("chance_of_playing_next_round")
    initial = float(chance) / 100 if chance is not None else (1.0 if packet.get("status") == "a" else 0.5)
    baseline = 1 - (1 - initial) * np.exp(-max(0, days) / 35)
    if days <= 0 or packet.get("status") not in ("i", "d") or not spec["recovery"]["promoted"]:
        return float(baseline)
    survival = 1 - initial
    elapsed = 0.0
    while elapsed < days:
        step = min(7, days - elapsed)
        x = recovery_features(packet, at, 7, 1 + elapsed / 7)
        x[4] += elapsed
        p = float(models["recovery"].predict(np.array([x]), num_threads=1)[0])
        w = spec["recovery"]["weight"]
        p = w * p + (1 - w) * (1 - np.exp(-7 / 35))
        survival *= max(0, 1 - p) ** (step / 7)
        elapsed += step
    return float(1 - survival)


def condition_rotation(probabilities, position, rest, recent, previous_minutes, loaded):
    spec, models = loaded
    if not spec["rotation"].get("joint_validation", {}).get("promoted", False) or rest <= 0 or rest > 10:
        return probabilities
    unique, inverse = np.unique(previous_minutes, return_inverse=True)
    count = len(previous_minutes)
    x = np.array([rotation_features(position, rest, recent, float(m), float(m >= 60)) for m in unique])
    conditional = np.asarray(models["rotation"].predict(x, num_threads=1))
    reference_x = x.copy()
    reference_x[:, 4] = np.mean([r.get("minutes") or 0 for r in recent]) if recent else 45
    reference_x[:, 5] = np.mean([r.get("starts") or 0 for r in recent]) if recent else 0.5
    reference_x[:, 6] = reference_x[:, 4] * max(0, 6 - rest) / 6
    reference = np.asarray(models["rotation"].predict(reference_x, num_threads=1))
    w = spec["rotation"]["joint_validation"]["weight"]
    ratio = (np.clip(conditional / np.maximum(reference, 0.02), 0.2, 5) ** w)[inverse]
    p = np.broadcast_to(np.asarray(probabilities), (count, 4)) * ratio
    return p / np.maximum(p.sum(1, keepdims=True), 1e-9)


def validate_joint_rotation(rows, models, store):
    """Score complete DGW minute-band distributions frozen before the deadline.

    The joint has all 16 outcomes. Actual first-match minutes select a label for
    scoring only; prediction sums over every possible first-match minute band.
    """
    import polars as pl
    from threadpoolctl import threadpool_limits

    from .context_archive import load_context
    from .decision import BAND_MINUTES, NAMES, fit_components, minute_band, player_key, predict_models

    feature = store.get("decision_features")
    frame = pl.read_parquet(store.artifact_path(feature["path"]))
    packets = load_context(store)
    prediction_cache = {}

    def base_probability(group, first, second):
        if group not in prediction_cache:
            previous = frame.filter(pl.col("deadline_group") < group)
            groups = sorted(previous["deadline_group"].unique().to_list())
            cut = groups[-8]
            train = previous.filter(pl.col("deadline_group") < cut)
            calibration = previous.filter(pl.col("deadline_group") >= cut)
            target = frame.filter(pl.col("deadline_group") == group)
            with threadpool_limits(limits=1):
                fitted = fit_components(train, calibration, False)
                _, probs, _ = predict_models(
                    fitted, target.select(NAMES).to_numpy(), target["dc_probability"].to_list()
                )
            prediction_cache[group] = {
                (r["player_id"], r["fixture_id"]): p for r, p in zip(target.to_dicts(), probs)
            }
        return [prediction_cache[group][int(r["element"]), int(r["fixture"])].copy() for r in (first, second)]

    by_week = defaultdict(list)
    for row in rows:
        by_week[row["season"], int(row["round"]), player_key(row)].append(row)
    past: dict = defaultdict(list)
    examples = []
    for (season, event, key), matches in sorted(
        by_week.items(), key=lambda z: (z[0][0], z[0][1], str(z[0][2]))
    ):
        matches.sort(key=lambda r: r.get("kickoff_time") or "")
        recent = past[key][-8:]
        if (
            f"{season}-GW{event:02d}" >= "2023/24-GW01"
            and len(matches) == 2
            and recent
            and all(r.get("kickoff_time") for r in [*matches, recent[-1]])
        ):
            first, second = matches
            rest2 = (moment(second["kickoff_time"]) - moment(first["kickoff_time"])).total_seconds() / 86400
            if 0 < rest2 <= 10:
                p1, p2 = base_probability(f"{season}-GW{event:02d}", first, second)
                packet = packets.get((season, event, int(first.get("code") or 0)), {})
                chance = packet.get("chance_of_playing_next_round")
                available = (
                    float(chance) / 100
                    if chance is not None
                    else (1.0 if packet.get("status", "a") == "a" else 0.5)
                )
                for probability in (p1, p2):
                    probability[1:] *= available
                    probability[0] = 1 - probability[1:].sum()
                cx = [
                    rotation_features(second["position"], rest2, recent, float(m), float(m >= 60))
                    for m in BAND_MINUTES
                ]
                conditional = np.asarray(models["rotation"].predict_proba(cx))
                reference_x = rotation_features(
                    second["position"],
                    rest2,
                    recent,
                    float(np.mean([r.get("minutes") or 0 for r in recent])),
                    float(np.mean([r.get("starts") or 0 for r in recent])),
                )
                reference = np.asarray(models["rotation"].predict_proba([reference_x]))[0]
                truth = minute_band(
                    np.array([float(first.get("minutes") or 0), float(second.get("minutes") or 0)])
                )
                examples.append(
                    {
                        "group": f"{season}-GW{event:02d}",
                        "first": p1,
                        "second": p2,
                        "conditional": conditional,
                        "reference": reference,
                        "truth": truth,
                    }
                )
        past[key].extend(matches)
    cal = [r for r in examples if "2023/24-GW01" <= r["group"] < "2024/25-GW01"]
    test = [r for r in examples if r["group"] >= "2024/25-GW01"]
    result = {
        "calibration_pairs": len(cal),
        "test_pairs": len(test),
        "test_deadlines": len({r["group"] for r in test}),
        "promoted": False,
        "scope": "Exact production density-ratio transition over all 16 joint states, against freshly refitted earlier-only production minute forecasts; no realized first-match input",
    }
    if len(cal) < 20 or len(test) < 20 or len({r["group"] for r in test}) < 6:
        return result

    def losses(items, weight):
        values = []
        for r in items:
            a, b = r["truth"]
            second = (
                r["second"][None, :]
                * np.clip(r["conditional"] / np.maximum(r["reference"][None, :], 0.02), 0.2, 5) ** weight
            )
            second /= second.sum(axis=1, keepdims=True)
            joint = r["first"][:, None] * second
            values.append(-np.log(max(1e-9, joint[a, b])))
        return np.asarray(values)

    weight = min((0, 0.25, 0.5, 0.75, 1), key=lambda w: float(losses(cal, w).mean()))
    ci = block_interval(losses(test, weight) - losses(test, 0), [r["group"] for r in test])
    result.update(
        weight=weight,
        log_loss=float(losses(test, weight).mean()),
        baseline_log_loss=float(losses(test, 0).mean()),
        block_difference_95ci=ci,
        promoted=bool(weight > 0 and ci[1] < 0),
    )
    return result
