"""Learned event rates feeding the coherent football-world simulator.

Event conservation is structural. Unknown supplier BPS actions are represented by
an empirically fitted conditional BPS model, never claimed as observed events.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_poisson_deviance

from .decision import NAMES, canonical_rows
from .features import profile
from .fusion import load_data
from .fusion_data import tree_matrix
from .storage import atomic_json, now

FIELDS = ("goals90", "assists90", "saves90", "yellow90", "red90")
AUX = (0, 1, 2, 5, 6)


def train_components(store):
    arrays, metadata, features = load_data(store)
    matrix = tree_matrix(
        arrays["tab"],
        arrays["news"],
        arrays["market"],
        arrays["workload"],
        arrays["text"],
        arrays["sequence"],
    )
    group = np.array([r["deadline_group"] for r in metadata])
    groups = sorted(set(group))
    testing = [g for g in groups if g >= "2025/26-GW01"]
    minutes = np.asarray(arrays["minutes"])
    exposure = np.maximum(minutes / 90, 0.01)
    aux = arrays["aux"]
    raw, _, _ = canonical_rows(store)
    mapping = {(r["season"], int(r["element"]), int(r["fixture"])): r for r in raw}
    bps = np.array(
        [float(mapping[r["season"], r["player_id"], r["fixture_id"]].get("bps") or 0) for r in metadata]
    )
    prior = {}
    history: dict[Any, list[dict]] = defaultdict(list)
    lookup = defaultdict(list)
    for i, r in enumerate(metadata):
        lookup[r["deadline_group"]].append(i)
    for g, idx in sorted(lookup.items()):
        for i in idx:
            r = metadata[i]
            original = mapping[r["season"], r["player_id"], r["fixture_id"]]
            key = original.get("code") or (r["season"], r["player_id"])
            prior[i] = profile(int(r["position"]), history[key])
        for i in idx:
            r = metadata[i]
            original = mapping[r["season"], r["player_id"], r["fixture_id"]]
            key = original.get("code") or (r["season"], r["player_id"])
            history[key].append(original)
            history[key] = history[key][-24:]
    del raw, mapping, history
    params: dict[str, Any] = dict(
        n_estimators=100,
        num_leaves=12,
        min_child_samples=100,
        reg_lambda=20,
        learning_rate=0.04,
        objective="poisson",
        n_jobs=1,
        verbosity=-1,
        random_state=157,
    )
    predictions: dict[str, list[dict]] = {name: [] for name in FIELDS}
    final_models = {}
    final_weights = {}
    folds = []
    for start in range(0, len(testing), 8):
        block = testing[start : start + 8]
        before = [g for g in groups if g < block[0]]
        cut = before[-4]
        train = np.flatnonzero((group < cut) & (minutes > 0))
        cal = np.flatnonzero((group >= cut) & (group < block[0]) & (minutes > 0))
        test = np.flatnonzero(np.isin(group, block) & (minutes > 0))
        for name, k in zip(FIELDS, AUX):
            tr = train[aux[train, k] >= 0]
            ca = cal[aux[cal, k] >= 0]
            te = test[aux[test, k] >= 0]
            if not len(te):
                continue
            target = aux[:, k] / exposure
            model = lgb.LGBMRegressor(**params).fit(matrix[tr], target[tr], sample_weight=exposure[tr])
            cp = np.maximum(1e-5, model.predict(matrix[ca], num_threads=1))
            tp = np.maximum(1e-5, model.predict(matrix[te], num_threads=1))
            basecal = np.array([prior[i][name] for i in ca])
            basetest = np.array([prior[i][name] for i in te])
            weight = min(
                (0, 0.25, 0.5, 0.75, 1),
                key=lambda w: mean_poisson_deviance(
                    aux[ca, k], np.maximum(1e-5, ((1 - w) * basecal + w * cp) * exposure[ca])
                ),
            )
            blended = ((1 - weight) * basetest + weight * tp) * exposure[te]
            for i, value, base in zip(te, blended, basetest * exposure[te]):
                predictions[name].append(
                    {
                        "group": str(group[i]),
                        "actual": float(aux[i, k]),
                        "prediction": float(value),
                        "baseline": float(base),
                    }
                )
        folds.append({"test": block, "train_before": cut, "calibration_before": block[0]})
        print("Component-rate fold complete", block[-1], flush=True)
    cut = groups[-4]
    train = np.flatnonzero((group < cut) & (minutes > 0))
    cal = np.flatnonzero((group >= cut) & (minutes > 0))
    directory = store.root / "models" / ("coherent-" + features["id"][:20])
    directory.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, k in zip(FIELDS, AUX):
        tr = train[aux[train, k] >= 0]
        ca = cal[aux[cal, k] >= 0]
        target = aux[:, k] / exposure
        model = lgb.LGBMRegressor(**params).fit(matrix[tr], target[tr], sample_weight=exposure[tr])
        cp = np.maximum(1e-5, model.predict(matrix[ca], num_threads=1))
        base = np.array([prior[i][name] for i in ca])
        weight = min(
            (0, 0.25, 0.5, 0.75, 1),
            key=lambda w: mean_poisson_deviance(
                aux[ca, k], np.maximum(1e-5, ((1 - w) * base + w * cp) * exposure[ca])
            ),
        )
        path = directory / (name + ".txt")
        model.booster_.save_model(str(path))
        final_models[name] = {"path": str(path), "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
        final_weights[name] = weight
        rows = predictions[name]
        y = np.array([r["actual"] for r in rows])
        p = np.maximum(1e-5, [r["prediction"] for r in rows])
        b = np.maximum(1e-5, [r["baseline"] for r in rows])
        loss = (y - p) ** 2 - (y - b) ** 2
        byg = defaultdict(list)
        for r, v in zip(rows, loss):
            byg[r["group"]].append(float(v))
        values = np.array([np.mean(v) for _, v in sorted(byg.items())])
        rng = np.random.default_rng(157)
        n = len(values)
        starts = rng.integers(0, n, (2000, (n + 3) // 4))
        sample_indices = ((starts[:, :, None] + np.arange(4)) % n).reshape(2000, -1)[:, :n]
        ci = np.quantile(values[sample_indices].mean(1), [0.025, 0.975]).tolist()
        metrics[name] = {
            "rows": len(rows),
            "poisson_deviance": float(mean_poisson_deviance(y, p)),
            "baseline_deviance": float(mean_poisson_deviance(y, b)),
            "mse_difference_block_95ci": ci,
            "calibrated_weight": weight,
            "promoted": bool(ci[1] < 0 and weight > 0),
        }
        if not metrics[name]["promoted"]:
            final_weights[name] = 0
    # This head is explicitly conditioned on generated match events and exposure.
    event_inputs = np.column_stack([matrix, minutes / 90, aux[:, [0, 1, 2, 4, 5, 6]]])
    bps_model = lgb.LGBMRegressor(
        n_estimators=160,
        num_leaves=16,
        min_child_samples=100,
        reg_lambda=20,
        n_jobs=1,
        verbosity=-1,
        random_state=158,
    ).fit(event_inputs[train], bps[train])
    residual = bps[cal] - bps_model.predict(event_inputs[cal], num_threads=1)
    path = directory / "bps.txt"
    bps_model.booster_.save_model(str(path))
    final_models["bps"] = {"path": str(path), "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {
        "at": now(),
        "feature_id": features["id"],
        "models": final_models,
        "weights": final_weights,
        "metrics": metrics,
        "folds": folds,
        "fit_before": cut,
        "bps_residual_quantiles": np.quantile(residual, np.linspace(0, 1, 101)).tolist(),
        "bps_calibration_rmse": float(np.sqrt(np.mean(residual**2))),
        "track": "Count-rate validation conditional on observed exposure; structural joint-world and prospective policy audits are separate",
        "coherence": [
            "11 distinct starters per club, one goalkeeper, at most five paired substitutions",
            "Goals and assists allocated only to on-pitch players, no self-assist",
            "Shared team goals, conceded intervals, goalkeeper saves and official bonus ties",
        ],
        "limits": [
            "Supplier BPS actions are incomplete: conditional BPS plus earlier residuals is an explicit learned proxy",
            "Conditional rate scores do not certify complete football-world or rival-probability calibration",
            "Only heads passing deadline-block tests alter the established component priors",
        ],
    }
    report["id"] = store.put("coherent_model", report)
    atomic_json(Path("artifacts/evaluations/coherent_model.json"), report)
    return report


def component_models(store):
    spec = store.get("coherent_model")
    models = {}
    for name, item in spec["models"].items():
        path = store.artifact_path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["hash"]:
            raise ValueError("Coherent component model checksum mismatch")
        models[name] = lgb.Booster(model_file=str(path))
    return spec, models


def live_component_inputs(store, rows, market, players):
    from .fusion_data import live_inputs

    inputs = live_inputs(store, rows, market, players)
    matrix = tree_matrix(
        inputs["tab"],
        inputs["news"],
        inputs["market"],
        inputs["workload"],
        inputs["text"],
        inputs["sequence"],
    )
    return matrix


def component_overrides(store, rows, market, players):
    spec, models = component_models(store)
    matrix = live_component_inputs(store, rows, market, players)
    out = {}
    for i, row in enumerate(rows):
        out[row["fixture_id"], row["player_id"]] = {"component_context": matrix[i]}
    for name in FIELDS:
        w = spec["weights"][name]
        if w <= 0:
            continue
        prediction = models[name].predict(matrix, num_threads=1)
        for i, row in enumerate(rows):
            value = (1 - w) * row["profile"][name] + w * max(0, float(prediction[i]))
            out[row["fixture_id"], row["player_id"]][name] = value
    return out, spec, models


def bps_worlds(rng, ids, profiles, minutes, goals, assists, saves, clean, yellow, red, models):
    spec, bundle = models
    nworlds = len(minutes)
    out = np.zeros_like(minutes, dtype=float)
    for col, p in enumerate(ids):
        context = profiles[p].get("component_context")
        if context is None:
            raise ValueError("Missing registered component context for BPS prediction")
        # Group equal event tuples to avoid one tree prediction per Monte Carlo row.
        events = np.column_stack(
            [
                minutes[:, col] / 90,
                goals[:, col],
                assists[:, col],
                saves[:, col],
                clean[:, col],
                yellow[:, col],
                red[:, col],
            ]
        )
        unique, inverse = np.unique(events, axis=0, return_inverse=True)
        matrix = np.column_stack([np.repeat(np.asarray(context)[None, :], len(unique), axis=0), unique])
        base = bundle["bps"].predict(matrix, num_threads=1)[inverse]
        noise = np.zeros(nworlds)
        position = int(context[0])
        for scored in (0, 1):
            mask = (goals[:, col] > 0) == bool(scored)
            pool = spec.get("bps_pools", {}).get(f"{position}:{scored}", spec["bps_residual_quantiles"])
            noise[mask] = np.interp(rng.random(int(mask.sum())), np.linspace(0, 1, len(pool)), pool)
        out[:, col] = np.rint(base + noise)
    return out


def validate_bonus_model(store):
    """Held-out conditional BPS and joint bonus tests against the prior simulator."""
    from .horizon import block_interval

    arrays, meta, features = load_data(store)
    x = tree_matrix(
        arrays["tab"],
        arrays["news"],
        arrays["market"],
        arrays["workload"],
        arrays["text"],
        arrays["sequence"],
    )
    rows, _, _ = canonical_rows(store)
    lookup = {(r["season"], int(r["element"]), int(r["fixture"])): r.get("bps") for r in rows}
    bps = np.array([float(lookup[r["season"], r["player_id"], r["fixture_id"]] or 0) for r in meta])
    del rows, lookup
    mins = arrays["minutes"]
    aux = arrays["aux"]
    pos = np.array([int(r["position"]) for r in meta])
    groups = np.array([r["deadline_group"] for r in meta])
    unique = sorted(set(groups))
    testing = [g for g in unique if g >= "2025/26-GW01"]
    events = np.column_stack([mins / 90, aux[:, [0, 1, 2, 4, 5, 6]]])
    matrix = np.column_stack([x, events])
    proxy = (
        3
        + (mins >= 60) * 3
        + aux[:, 0] * np.where(pos == 4, 24, np.where(pos == 3, 18, 12))
        + aux[:, 1] * 9
        + aux[:, 4] * (pos <= 2) * 12
        + aux[:, 2] * 2
        + x[:, NAMES.index("actions90")] * mins / 90 * 0.5
        - aux[:, 5] * 3
        - aux[:, 6] * 9
    )
    records: list[dict[str, Any]] = []
    bonus_error: dict[str, list[float]] = {"learned": [], "baseline": []}
    last_pools = {}
    rng = np.random.default_rng(167)
    for first in range(0, len(testing), 8):
        block = testing[first : first + 8]
        before = [g for g in unique if g < block[0]]
        cut = before[-4]
        train = np.flatnonzero((groups < cut) & (mins > 0))
        cal = np.flatnonzero((groups >= cut) & (groups < block[0]) & (mins > 0))
        test = np.flatnonzero(np.isin(groups, block) & (mins > 0))
        model = lgb.LGBMRegressor(
            n_estimators=160,
            num_leaves=16,
            min_child_samples=100,
            reg_lambda=20,
            n_jobs=1,
            verbosity=-1,
            random_state=158,
        ).fit(matrix[train], bps[train])
        res = bps[cal] - model.predict(matrix[cal], num_threads=1)
        prediction = model.predict(matrix[test], num_threads=1)
        pools = {}
        for p in range(1, 5):
            parent = res[pos[cal] == p]
            if len(parent) < 30:
                parent = res
            for scored in (0, 1):
                pool = res[(pos[cal] == p) & ((aux[cal, 0] > 0) == bool(scored))]
                if len(pool) < 30:
                    pool = parent
                pools[f"{p}:{scored}"] = np.quantile(pool, np.linspace(0, 1, 101)).tolist()
        buckets = defaultdict(list)
        for j, i in enumerate(test):
            records.append(
                {
                    "deadline_group": str(groups[i]),
                    "actual": float(bps[i]),
                    "learned": float(prediction[j] + res.mean()),
                    "baseline": float(proxy[i]),
                }
            )
            buckets[meta[i]["season"], meta[i]["fixture_id"]].append((j, i))
        for indices in buckets.values():
            actual = np.array([aux[i, 3] for _, i in indices])
            new = []
            old = []
            for j, i in indices:
                pool = pools[f"{pos[i]}:{int(aux[i, 0] > 0)}"]
                new.append(prediction[j] + np.interp(rng.random(256), np.linspace(0, 1, 101), pool))
                old.append(proxy[i] + rng.normal(0, 4, 256))
            for kind, values in (("learned", new), ("baseline", old)):
                scores = np.rint(np.array(values).T)
                rank = (scores[:, :, None] > scores[:, None, :]).sum(axis=1)
                bonus = np.maximum(0, 3 - rank).mean(0)
                bonus_error[kind].extend(((bonus - actual) ** 2).tolist())
        print("Conditional joint bonus audit", block[-1], flush=True)
    ci = block_interval(records, "learned")
    mse = {k: float(np.mean(v)) for k, v in bonus_error.items()}
    spec = store.get("coherent_model")
    validation = {
        "rows": len(records),
        "conditional_bps_rmse": float(np.sqrt(np.mean([(r["learned"] - r["actual"]) ** 2 for r in records]))),
        "prior_bps_rmse": float(np.sqrt(np.mean([(r["baseline"] - r["actual"]) ** 2 for r in records]))),
        "bps_rmse_block_95ci": ci,
        "joint_bonus_rmse": {k: float(np.sqrt(v)) for k, v in mse.items()},
        "worlds_per_fixture": 256,
        "scope": "Conditioned on observed match events/exposure, each BPS model is fitted strictly earlier; neither score is a live future-event forecast",
    }
    spec = {k: v for k, v in spec.items() if k != "id"}
    spec["bps_validation"] = validation
    spec["bps_promoted"] = bool(ci[1] < 0 and mse["learned"] < mse["baseline"])
    # Residual pools for the actual final fit use only its earlier calibration slice.
    cut = spec["fit_before"]
    cal = np.flatnonzero((groups >= cut) & (mins > 0))
    _, models = component_models(store)
    res = bps[cal] - models["bps"].predict(matrix[cal], num_threads=1)
    for p in range(1, 5):
        parent = res[pos[cal] == p]
        if len(parent) < 30:
            parent = res
        for scored in (0, 1):
            pool = res[(pos[cal] == p) & ((aux[cal, 0] > 0) == bool(scored))]
            if len(pool) < 30:
                pool = parent
            last_pools[f"{p}:{scored}"] = np.quantile(pool, np.linspace(0, 1, 101)).tolist()
    spec["bps_pools"] = last_pools
    spec["id"] = store.put("coherent_model", spec)
    atomic_json(Path("artifacts/evaluations/coherent_model.json"), spec)
    return spec


def compare_football_worlds(store, worlds=1000):
    from .simulation import forecast, score_plan

    state = store.get("state")
    plan = store.get("plan", store.get("selected_plan")["plan_id"])
    production = store.get("forecast", plan["forecast_id"])
    if plan["state_id"] != state["id"]:
        raise ValueError("Account changed before football-world comparison")
    physical = forecast(
        store, state["event"], plan["end_event"], worlds, seed=production["seed"], component_baseline=True
    )
    physical_scores = score_plan(store, plan.copy(), physical)
    production_scores = score_plan(store, plan.copy(), production)
    prior = {r["player_id"]: r["mean"] for r in production["players"] if r["event"] == state["event"]}
    report = {
        "at": now(),
        "state_id": state["id"],
        "market_id": production["market_id"],
        "plan_id": plan["id"],
        "forecast_id": production["id"],
        "coherent_forecast_id": physical["id"],
        "coherent_model_id": physical.get("coherent_model_id"),
        "production_mean": float(production_scores.mean()),
        "football_world_mean": float(physical_scores.mean()),
        "football_world_quantiles": np.quantile(physical_scores, [0.1, 0.5, 0.9]).tolist(),
        "players": [
            {
                "player_id": r["player_id"],
                "production": prior.get(r["player_id"], 0),
                "football_world": r["mean"],
            }
            for r in physical["players"]
            if r["event"] == state["event"]
        ],
        "fixtures": len(physical["match_audit"]),
        "invalid_goal_allocations": sum(
            a["scored_goals"] + a["own_goals"] != a["team_goals"] for a in physical["match_audit"]
        ),
        "short_lineup_worlds": sum(
            q["short_lineup_worlds"] for a in physical["match_audit"] for q in a["lineups"]
        ),
        "scope": "Independent structural football-world sensitivity of the actual selected policy; distribution difference is model uncertainty, not a paired causal effect",
        "production_changed": False,
    }
    report["id"] = store.put("coherent_comparison", report)
    return report
