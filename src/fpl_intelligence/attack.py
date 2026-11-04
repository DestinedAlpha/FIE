"""Separated attacking channels, hierarchical finishing and FPL-assist attribution."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from scipy.stats import binom
from sklearn.metrics import mean_poisson_deviance

from .availability import block_interval
from .shot_data import CHANNELS
from .storage import atomic_json, digest, now

FEATURES = tuple(
    f"{kind}_{stat}" for kind in CHANNELS for stat in ("xg90", "goals90", "shots90", "finishing")
) + ("xa90", "key_passes90", "chain90", "buildup90", "exposure", "position")
PRIORS = {
    1: (0.002, 0.001, 0.0001),
    2: (0.025, 0.035, 0.001),
    3: (0.15, 0.04, 0.015),
    4: (0.29, 0.035, 0.055),
}


def attack_vector(past, position, decay=0.94, finishing_strength=30):
    recent = sorted(past, key=lambda r: r["known_after"])[-24:]
    weights = [decay**i for i in range(len(recent) - 1, -1, -1)]
    exposure = sum(float(r["minutes"]) * w / 90 for r, w in zip(recent, weights))
    values = []
    for kind, prior in zip(CHANNELS, PRIORS[int(position)]):
        xg = sum(float(r["xg"][kind]) * w for r, w in zip(recent, weights))
        goals = sum(float(r["goals"][kind]) * w for r, w in zip(recent, weights))
        shots = sum(float(r["shots"][kind]) * w for r, w in zip(recent, weights))
        finishing = (
            (goals + finishing_strength) / (xg + finishing_strength)
            if finishing_strength is not None
            else 1.0
        )
        values += [
            (xg + 5 * prior) / (exposure + 5),
            (goals + 10 * prior) / (exposure + 10),
            shots / (exposure + 3),
            finishing,
        ]
    values += [
        sum(float(r[field]) * w for r, w in zip(recent, weights)) / (exposure + 3)
        for field in ("xa", "key_passes", "xg_chain", "xg_buildup")
    ]
    return np.array(values + [exposure, float(position)], dtype=np.float32)


def assist_probability(counts, target, probabilities):
    distribution = np.array([1.0])
    for n, p in zip(counts, probabilities):
        distribution = np.convolve(distribution, binom.pmf(np.arange(int(n) + 1), int(n), p))
    return float(distribution[int(target)]) if 0 <= target < len(distribution) else 0.0


def train_attack(store, temporal_model_id=None):
    from .decision import canonical_rows

    archive = store.get("shot_archive")
    originals, _, _ = canonical_rows(store)
    temporal = store.get("temporal_model", temporal_model_id) if store.list("temporal_model", 1) else None
    temporal_params = temporal["adopted"] if temporal else {"decay": 0.94, "finishing_strength": None}
    rows = sorted(archive["records"], key=lambda r: (r["season"], r["event"], r["kickoff"], r["player_code"]))
    past = defaultdict(list)
    items = []
    fixture_goals: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=int))
    pending: list[dict] = []
    last_group = None
    for r in rows:
        group = f"{r['season']}-GW{r['event']:02d}"
        if group != last_group:
            for previous in pending:
                past[previous["player_code"]].append(previous)
                past[previous["player_code"]] = past[previous["player_code"]][-24:]
            pending = []
            last_group = group
        x = attack_vector(past[r["player_code"]], r["position"], **temporal_params)
        if r["minutes"] > 0:
            items.append(
                {
                    "group": group,
                    "x": x,
                    "exposure": r["minutes"] / 90,
                    "goals": np.array([r["goals"][k] for k in CHANNELS]),
                }
            )
        pending.append(r)
        key = (group, r["fixture_id"], r["team"])
        fixture_goals[key][:3] += np.array([r["goals"][k] for k in CHANNELS])
    # FPL assists and own-goal residuals use the entire registered fixture, not just matched players.
    official: dict[tuple, dict[str, Any]] = defaultdict(
        lambda: {"assists": 0, "own": 0, "goals": 0, "opponent": None}
    )
    for r in originals:
        key = (f"{r['season']}-GW{int(r['round']):02d}", int(r["fixture"]), r["team"])
        official[key]["assists"] += int(r.get("assists") or 0)
        official[key]["goals"] += int(r.get("goals_scored") or 0)
        official[key]["own"] += int(r.get("own_goals") or 0)
        official[key]["opponent"] = r["opponent_name"]
    assist_rows = []
    for key, counts in fixture_goals.items():
        if key not in official:
            continue
        fact = official[key]
        opponent = official.get((key[0], key[1], fact["opponent"]), {})
        counts[3] = int(opponent.get("own", 0))
        # Incomplete identity joins must not become false zero labels.
        if counts[:3].sum() != fact["goals"] or fact["assists"] > counts.sum():
            continue
        assist_rows.append({"group": key[0], "counts": counts.tolist(), "actual": fact["assists"]})
    train = [r for r in items if r["group"] < "2024/25-GW25"]
    cal = [r for r in items if "2024/25-GW25" <= r["group"] < "2025/26-GW01"]
    test = [r for r in items if r["group"] >= "2025/26-GW01"]
    if not all((train, cal, test)):
        raise ValueError("Attack-channel fitting needs earlier calibration and later matches")
    directory = (
        store.root
        / "models"
        / (
            "attack-v3-"
            + digest(
                {
                    "archive": archive["id"],
                    "parameters": temporal_params,
                    "protocol": "proper-poisson-goals-v3",
                }
            )[:20]
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    models = {}
    metrics = {}
    params: dict[str, Any] = dict(
        n_estimators=100,
        num_leaves=12,
        min_child_samples=80,
        reg_lambda=30,
        learning_rate=0.04,
        n_jobs=1,
        verbosity=-1,
        random_state=773,
        objective="poisson",
    )
    for k, kind in enumerate(CHANNELS):
        model = lgb.LGBMRegressor(**params).fit(
            np.array([r["x"] for r in train]),
            np.array([r["goals"][k] / r["exposure"] for r in train]),
            sample_weight=[r["exposure"] for r in train],
        )

        def prior(rs):
            return np.array([r["x"][4 * k] * r["x"][4 * k + 3] * r["exposure"] for r in rs])

        cp = np.maximum(1e-6, model.predict(np.array([r["x"] for r in cal]), num_threads=1)) * np.array(
            [r["exposure"] for r in cal]
        )
        base = prior(cal)
        y = np.array([r["goals"][k] for r in cal])
        weight = min(
            (0, 0.25, 0.5, 0.75, 1),
            key=lambda w: mean_poisson_deviance(y, np.maximum(1e-6, w * cp + (1 - w) * base)),
        )
        pred = weight * np.maximum(
            1e-6, model.predict(np.array([r["x"] for r in test]), num_threads=1)
        ) * np.array([r["exposure"] for r in test]) + (1 - weight) * prior(test)
        truth = np.array([r["goals"][k] for r in test])
        baseline = prior(test)
        diff = 2 * (xlogy(truth, np.maximum(1e-6, baseline) / np.maximum(1e-6, pred)) + pred - baseline)
        ci = block_interval(diff, [r["group"] for r in test])
        metrics[kind] = {
            "rows": len(test),
            "weight": weight,
            "deviance": float(mean_poisson_deviance(truth, np.maximum(1e-6, pred))),
            "baseline_deviance": float(mean_poisson_deviance(truth, np.maximum(1e-6, baseline))),
            "block_poisson_difference_95ci": ci,
            "promoted": bool(weight > 0 and ci[1] < 0),
        }
        path = directory / (kind + ".txt")
        model.booster_.save_model(str(path))
        models[kind] = {"path": str(path), "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    atr = [r for r in assist_rows if r["group"] < "2024/25-GW25"]
    aca = [r for r in assist_rows if "2024/25-GW25" <= r["group"] < "2025/26-GW01"]
    ate = [r for r in assist_rows if r["group"] >= "2025/26-GW01"]
    baseline = np.array([0.74, 0.74, 0.74, 0.0])
    assist_prior = np.array([0.74, 0.74, 0.55, 0.4])

    def nll(rs, p):
        return np.array([-np.log(max(1e-10, assist_probability(r["counts"], r["actual"], p))) for r in rs])

    candidates = []
    for ridge in (1, 10, 50):
        fit = minimize(
            lambda p: float(nll(atr, p).sum() + ridge * np.sum((p - assist_prior) ** 2)),
            assist_prior,
            bounds=[(0.001, 0.999)] * 4,
            method="L-BFGS-B",
        )
        if fit.success:
            candidates.append((float(nll(aca, fit.x).mean()), ridge, fit.x))
    if not candidates:
        raise ValueError("FPL assist attribution calibration failed")
    _, ridge, prob = min(candidates, key=lambda r: r[0])
    aci = block_interval(nll(ate, prob) - nll(ate, baseline), [r["group"] for r in ate])
    assist = {
        "channels": [*CHANNELS, "own_goal"],
        "probabilities": prob.tolist(),
        "baseline": baseline.tolist(),
        "training_matches": len(atr),
        "calibration_matches": len(aca),
        "test_matches": len(ate),
        "nll": float(nll(ate, prob).mean()),
        "baseline_nll": float(nll(ate, baseline).mean()),
        "block_nll_difference_95ci": aci,
        "promoted": bool(aci[1] < 0),
        "ridge": ridge,
        "method": "Poisson-binomial likelihood of official aggregate FPL assists conditional on observed goal types; no individual assist attribution is fabricated",
    }
    report = {
        "at": now(),
        "shot_archive_id": archive["id"],
        "temporal_params": temporal_params,
        "temporal_model_id": temporal["id"] if temporal else None,
        "models": models,
        "features": FEATURES,
        "metrics": metrics,
        "assist": assist,
        "hierarchy": "Channel/position opportunity priors and Gamma-Poisson finishing shrinkage; finite samples do not become unbounded finishing multipliers",
        "protocol": "Fit before 2024/25 GW25; calibrate remainder of that season; test later 2025/26 and finalized current matches",
    }
    report["id"] = store.put("attack_model", report)
    atomic_json(Path("artifacts/evaluations/attack_channels.json"), report)
    return report


def attack_overrides(store, players, profiles):
    spec = store.get("attack_model")
    archive = store.get("shot_archive")
    bycode = defaultdict(list)
    for r in archive["records"]:
        if r["known_after"] < now():
            bycode[r["player_code"]].append(r)
    ids = sorted(players)
    temporal_params = spec.get("temporal_params", {"decay": 0.94, "finishing_strength": 30})
    matrix = np.array(
        [attack_vector(bycode[players[p].code], players[p].position, **temporal_params) for p in ids]
    )
    out: dict[int, dict[str, Any]] = {p: {} for p in ids}
    for k, kind in enumerate(CHANNELS):
        item = spec["models"][kind]
        path = store.artifact_path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["hash"]:
            raise ValueError("Attack-channel checksum mismatch")
        model = lgb.Booster(model_file=str(path))
        pred = np.maximum(0, model.predict(matrix, num_threads=1))
        weight = spec["metrics"][kind]["weight"] if spec["metrics"][kind]["promoted"] else 0
        for i, p in enumerate(ids):
            out[p][kind] = float(weight * pred[i] + (1 - weight) * matrix[i, 4 * k] * matrix[i, 4 * k + 3])
    for p in ids:
        rates = out[p]
        total = sum(rates.values())
        # Separate allocation shares preserve the established expected goal total unless its rate head passed.
        any_promoted = any(m["promoted"] for m in spec["metrics"].values())
        if total > 0 and not any_promoted:
            rates = {k: v * profiles[p]["goals90"] / total for k, v in rates.items()}
        out[p] = {"goal_channels90": rates, "goals90": sum(rates.values())}
        if spec["assist"]["promoted"]:
            out[p]["assist_by_channel"] = dict(
                zip(spec["assist"]["channels"], spec["assist"]["probabilities"])
            )
    return out, spec
