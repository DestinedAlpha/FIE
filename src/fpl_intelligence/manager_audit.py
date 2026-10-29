"""Actual manager decisions are comparators, not privileged outcome labels."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import numpy as np

from .ingestion import Official, load_market
from .rules import Rules, best_lineup, score_gameweek
from .storage import Store, atomic_json, now


async def benchmark_managers(store: Store) -> dict:
    cohort = store.get("cohort")
    spec = store.get("decision_model")
    players, market = load_market(store)
    event = cohort["event"]
    forecast: dict[int, float] = {}
    actual: dict[int, int] = {}
    minutes: dict[int, int] = {}
    for row in spec["predictions"]:
        if row["season"] == market["season"] and row["event"] == event:
            p = row["player_id"]
            forecast[p] = forecast.get(p, 0.0) + row["predicted"]
            actual[p] = actual.get(p, 0) + int(row["actual"])
            minutes[p] = minutes.get(p, 0) + int(row["minutes"])
    if not forecast:
        raise ValueError("No out-of-fold forecast for the cohort's released event")
    official = Official(store)
    entries = cohort["sampled_entries"]
    async with httpx.AsyncClient(timeout=25, follow_redirects=False) as client:
        histories = await asyncio.gather(
            *(
                official.fetch(client, f"entry/{entry}/history/", f"manager_history_{entry}")
                for entry in entries
            )
        )
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    comparisons = []
    missing = 0
    for entry, history in zip(entries, histories):
        try:
            picks = store.latest(f"cohort_picks_{entry}_{event}")[0]
        except ValueError:
            missing += 1
            continue
        if not history or not picks.get("picks"):
            missing += 1
            continue
        prior = sorted(
            [r for r in history.get("past", []) if r.get("season_name", "") < market["season"]],
            key=lambda r: r["season_name"],
        )[-3:]
        qualified = sum(0 < (r.get("rank") or 0) <= 10000 for r in prior) >= 2
        squad = [r["element"] for r in picks["picks"]]
        if not all(p in players and p in forecast for p in squad):
            missing += 1
            continue
        chip = picks.get("active_chip")
        candidate = best_lineup(squad, forecast, players, rules)
        # Identical squad and chip. Existing transfer penalties cancel in the delta.
        observed = sum(actual[r["element"]] * r["multiplier"] for r in picks["picks"])
        hypothetical = score_gameweek(actual, minutes, candidate, players, rules, chip)
        comparisons.append(
            {
                "entry": entry,
                "prior_seasons": prior,
                "prior_top10k_twice": qualified,
                "observed_points": observed,
                "same_squad_model_points": hypothetical,
                "delta": hypothetical - observed,
            }
        )
    delta = np.array([r["delta"] for r in comparisons])
    if len(delta):
        rng = np.random.default_rng(47)
        bootstrap = delta[rng.integers(0, len(delta), (1000, len(delta)))].mean(axis=1)
        interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    else:
        interval = None
    report = {
        "at": now(),
        "decision_model_id": spec["id"],
        "event": event,
        "season": market["season"],
        "retrieved": len(comparisons),
        "missing": missing,
        "prior_top10k_twice": sum(r["prior_top10k_twice"] for r in comparisons),
        "qualification": "At least two top-10,000 overall finishes in the three recorded PRIOR seasons",
        "selection": "Frozen current-rank cohort; current-rank selection bias remains disclosed",
        "mean_same_squad_lineup_delta": float(delta.mean()) if len(delta) else None,
        "entry_bootstrap_interval_95": interval,
        "comparisons": comparisons,
        "errors": official.errors,
        "training_effect": "None: manager choices are an external comparator, not assumed true or used as hindsight training labels",
        "limits": [
            "One released gameweek is a diagnostic, not a skill or season-performance proof",
            "Current-rank selection is not a representative prior-elite sample",
            "Unavailable historical news can favor real managers or the retrospective model in different ways",
        ],
    }
    report["id"] = store.put("manager_audit", report)
    atomic_json(
        Path("artifacts/evaluations/manager_audit.json"),
        {k: v for k, v in report.items() if k not in ("comparisons", "errors")},
    )
    return report
