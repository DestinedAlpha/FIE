"""Matched historical controls for the reference-only weekly team replay.

All baseline decisions are frozen before target scoring. These are
diagnostic comparisons on an already inspected period, not promotion evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .decision import canonical_rows
from .domain import Lineup
from .optimization import select_squad
from .replay import deadline_rows
from .rules import Rules, best_lineup, score_gameweek
from .storage import atomic_json, now

BASELINES = {
    "official_expected": "Archived ep_next; already availability-adjusted by its publisher",
    "points_per_game": "Archived points_per_game multiplied by published availability",
    "ownership": "Archived selected_by_percent multiplied by published availability; ranking signal, not xPts",
}


def baseline_signals(players, method):
    if method not in BASELINES:
        raise ValueError("Unknown weekly baseline")
    field = {
        "official_expected": "ep_next",
        "points_per_game": "points_per_game",
        "ownership": "selected_by_percent",
    }[method]
    result = {}
    for identity, player in players.items():
        if not player.selectable:
            result[identity] = 0.0
            continue
        raw = player.raw.get(field)
        if raw is None or raw == "":
            raise ValueError(f"Missing {field} for selectable player {identity}; coverage would differ")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("Nonfinite archived baseline signal")
        availability = player.chance if player.chance is not None else (1.0 if player.status == "a" else 0.5)
        result[identity] = (
            value if method == "official_expected" and availability > 0 else value * availability
        )
    return result


def compare_weekly_baselines(store, seconds=60):
    if not 1 <= seconds <= 300:
        raise ValueError("Baseline solve budget must be between 1 and 300 seconds")
    replay = store.get("weekly_replay")
    before = store.get("state")["id"] if store.list("state", 1) else None
    frozen = []
    packets = {}
    # Decisions use only archived signals; scoring waits for every frozen team.
    for week in replay["weeks"]:
        source = store.get("replay_inputs", week["input_id"])
        players, _ = deadline_rows(
            [],
            source["bootstrap"],
            source["fixture_proof"]["fixtures"],
            week["season"],
            week["event"],
            source["deadline"],
        )
        rules = Rules.from_bootstrap(source["bootstrap"])
        packets[week["event"]] = (players, rules)
        for method in BASELINES:
            signals = baseline_signals(players, method)
            solver: dict = {}
            squad = select_squad(
                None,
                players,
                rules,
                signals,
                seconds=seconds,
                initial_budget=1000,
                bench_weight=0,
                relative_gap=0,
                diagnostics=solver,
            )
            lineup = best_lineup(squad, signals, players, rules)
            decision = {
                "season": week["season"],
                "event": week["event"],
                "method": method,
                "definition": BASELINES[method],
                "input_id": source["id"],
                "players": [{"id": p, "name": players[p].web_name, "price": players[p].price} for p in squad],
                "bank": 1000 - sum(players[p].price for p in squad),
                "lineup": lineup.model_dump(),
                "solver": solver,
                "selection_utility": sum(signals[p] for p in lineup.xi) + signals[lineup.captain],
                "bench_policy": "Descending baseline signal; goalkeeper first; standard legal formation-preserving actual autosubs",
                "frozen_at": now(),
            }
            decision["id"] = store.put("weekly_baseline_decision", decision)
            frozen.append(decision)
    rows, _, _ = canonical_rows(store)
    actual: dict = defaultdict(lambda: defaultdict(float))
    minutes: dict = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row["season"] == replay["season"]:
            actual[int(row["round"])][int(row["element"])] += float(row.get("total_points") or 0)
            minutes[int(row["round"])][int(row["element"])] += int(row.get("minutes") or 0)
    scored = {}
    for decision in frozen:
        event = decision["event"]
        players, rules = packets[event]
        scored[event, decision["method"]] = {
            **decision,
            "actual_points": score_gameweek(
                actual[event], minutes[event], Lineup.model_validate(decision["lineup"]), players, rules
            ),
        }
    events = {e["id"]: e for e in store.get("market")["events"]}
    weeks = [
        {
            "event": w["event"],
            "reference_actual": w["actual_points"],
            "official_average_context_only": events[w["event"]].get("average_entry_score"),
            "baselines": {method: scored[w["event"], method] for method in BASELINES},
        }
        for w in replay["weeks"]
    ]
    comparisons = {}
    for method in BASELINES:
        delta = np.array([w["reference_actual"] - w["baselines"][method]["actual_points"] for w in weeks])
        rng = np.random.default_rng(716)
        n = len(delta)
        starts = rng.integers(0, n, size=(4000, (n + 1) // 2))
        indices = ((starts[:, :, None] + np.arange(2)) % n).reshape(4000, -1)[:, :n]
        comparisons[method] = {
            "baseline_total": sum(w["baselines"][method]["actual_points"] for w in weeks),
            "reference_minus_baseline_total": float(delta.sum()),
            "weeks_reference_wins": int((delta > 0).sum()),
            "mean_weekly_difference_two_week_block_95ci": np.quantile(
                delta[indices].mean(1), [0.025, 0.975]
            ).tolist(),
        }
    result = {
        "at": now(),
        "weekly_replay_id": replay["id"],
        "weeks": weeks,
        "reference_total": replay["total_points"],
        "comparisons": comparisons,
        "account_state_unchanged": before is None or store.get("state")["id"] == before,
        "matched_constraints": "Same archived registered market/prices, £100m, 15-player composition, club limits, no chips, independent new team each week, zero requested additive MIP gap and exact actual rule scorer",
        "limits": [
            "The 0.6 weekly replay exercises the refitted reference forecaster, not the complete integrated live system.",
            "Reference retains its world-optimized bench/vice policy; controls order by their own published signal. This is a policy comparison, not a pure isolated forecast-error experiment.",
            "Eight already-inspected weeks and a descriptive block interval cannot establish prospective superiority; do not tune models on these outcomes and call them untouched tests.",
            "Official average is context only: ordinary managers have transfer constraints and may use chips.",
        ],
    }
    result["id"] = store.put("weekly_baseline_comparison", result)
    atomic_json(Path("artifacts/evaluations/weekly-baseline-comparison.json"), result)
    return result
