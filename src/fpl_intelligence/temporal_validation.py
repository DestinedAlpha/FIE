"""Validate recency and finishing shrinkage before allowing a 'luck correction'."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import gamma

from .availability import block_interval
from .shot_data import CHANNELS
from .storage import atomic_json, now


def validate_temporal(store):
    archive = store.get("shot_archive")
    if not archive.get("records"):
        raise ValueError("Recency validation requires a nonempty verified shot archive")
    rows = sorted(archive["records"], key=lambda r: (r["season"], r["event"], r["kickoff"], r["player_code"]))
    choices = [(decay, strength) for decay in (0.85, 0.94, 0.98, 1.0) for strength in (None, 30.0, 100.0)]
    losses = defaultdict(list)
    groups = []
    past = defaultdict(list)
    priors = {
        1: (0.002, 0.001, 0.0001),
        2: (0.025, 0.035, 0.001),
        3: (0.15, 0.04, 0.015),
        4: (0.29, 0.035, 0.055),
    }
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
        key = r["player_code"]
        history = past[key][-24:]
        if r["minutes"] > 0:
            goals = np.array([r["goals"][k] for k in CHANNELS])
            exposure = r["minutes"] / 90
            for decay in (0.85, 0.94, 0.98, 1.0):
                w = np.array([decay**i for i in range(len(history) - 1, -1, -1)])
                xg = (
                    np.sum([np.array([h["xg"][k] for k in CHANNELS]) * v for h, v in zip(history, w)], axis=0)
                    if history
                    else np.zeros(3)
                )
                actual = (
                    np.sum(
                        [np.array([h["goals"][k] for k in CHANNELS]) * v for h, v in zip(history, w)], axis=0
                    )
                    if history
                    else np.zeros(3)
                )
                e = sum(h["minutes"] / 90 * v for h, v in zip(history, w))
                rate = (xg + 5 * np.array(priors[int(r["position"])])) / (e + 5)
                for strength in (None, 30.0, 100.0):
                    finishing = 1 if strength is None else (actual + strength) / (xg + strength)
                    mu = np.maximum(1e-7, rate * finishing * exposure)
                    losses[decay, strength].append(
                        float(np.sum(mu - goals * np.log(mu) + gammaln(goals + 1)))
                    )
            groups.append(f"{r['season']}-GW{r['event']:02d}")
        pending.append(r)
    group_array = np.array(groups)
    cal = (group_array >= "2024/25-GW25") & (group_array < "2025/26-GW01")
    test = group_array >= "2025/26-GW01"
    baseline = (1.0, None)
    chosen_decay = min((0.85, 0.94, 0.98, 1.0), key=lambda d: np.asarray(losses[d, None])[cal].mean())
    decay_ci = block_interval(
        np.asarray(losses[chosen_decay, None])[test] - np.asarray(losses[baseline])[test], group_array[test]
    )
    decay_passed = bool(chosen_decay != 1 and decay_ci[1] < 0)
    adopted_decay = chosen_decay if decay_passed else 1.0
    chosen_strength = min((None, 30.0, 100.0), key=lambda v: np.asarray(losses[adopted_decay, v])[cal].mean())
    finish_ci = block_interval(
        np.asarray(losses[adopted_decay, chosen_strength])[test]
        - np.asarray(losses[adopted_decay, None])[test],
        group_array[test],
    )
    finish_passed = bool(chosen_strength is not None and finish_ci[1] < 0)
    selected = (adopted_decay, chosen_strength)
    adopted = (adopted_decay, chosen_strength if finish_passed else None)
    ci = block_interval(
        np.asarray(losses[adopted])[test] - np.asarray(losses[baseline])[test], group_array[test]
    )
    passed = decay_passed or finish_passed
    report = {
        "at": now(),
        "shot_archive_id": archive["id"],
        "selected_on_calibration": {"decay": selected[0], "finishing_strength": selected[1]},
        "adopted": {"decay": adopted[0], "finishing_strength": adopted[1]},
        "promoted": bool(passed),
        "decay_validation": {
            "selected": chosen_decay,
            "promoted": decay_passed,
            "block_nll_difference_95ci": decay_ci,
        },
        "finishing_validation": {
            "selected": chosen_strength,
            "promoted": finish_passed,
            "block_nll_difference_95ci": finish_ci,
        },
        "test_rows": int(test.sum()),
        "test_nll": float(np.asarray(losses[adopted])[test].mean()),
        "baseline_nll": float(np.asarray(losses[baseline])[test].mean()),
        "block_nll_difference_95ci": ci,
        "choices": [
            {
                "decay": c[0],
                "finishing_strength": c[1],
                "calibration_nll": float(np.asarray(losses[c])[cal].mean()),
            }
            for c in choices
        ],
        "scope": "Earlier calibration chooses recency/finishing; later goal observations gate application. Goal-minus-xG is not identified as pure luck or used as a points penalty.",
    }
    report["id"] = store.put("temporal_model", report)
    atomic_json(Path("artifacts/evaluations/temporal_luck.json"), report)
    return report


def luck_report(store):
    from .ingestion import load_market

    players, market = load_market(store)
    archive = store.get("shot_archive")
    model = store.get("temporal_model")
    prior = model["adopted"]["finishing_strength"]
    records = []
    for p in players.values():
        matches = [
            r for r in archive["records"] if r["player_code"] == p.code and r["season"] == market["season"]
        ]
        xg = sum(sum(r["xg"].values()) for r in matches)
        goals = sum(sum(r["goals"].values()) for r in matches)
        alpha = prior or 30
        posterior = (alpha + goals) / (alpha + xg)
        interval = gamma.ppf([0.025, 0.975], alpha + goals, scale=1 / (alpha + xg)).tolist()
        records.append(
            {
                "player_id": p.id,
                "observed_goals": goals,
                "observed_xg": xg,
                "outcome_deviation": goals - xg,
                "matches": len(matches),
                "posterior_finishing": posterior,
                "finishing_95ci": interval,
                "applied": bool(prior is not None and model["promoted"]),
                "interpretation": "Deviation includes luck, finishing skill and xG measurement error; no hindsight player penalty",
            }
        )
    result = {"at": now(), "market_id": market["id"], "temporal_model_id": model["id"], "records": records}
    result["id"] = store.put("luck_report", result)
    return result
