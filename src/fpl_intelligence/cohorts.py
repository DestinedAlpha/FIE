"""Disclosed public manager samples and realized effective ownership."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import numpy as np

from .ingestion import Official, load_market
from .storage import Store, atomic_json, now


async def sample_current_rank(store: Store, league_id: int = 314, limit: int = 50) -> dict:
    if not 1 <= limit <= 500:
        raise ValueError("Cohort budget is 1–500 actually retrieved managers")
    _, market = load_market(store)
    event = next((e["id"] for e in market["events"] if e.get("is_current")), None)
    if event is None:
        raise ValueError("No released event is available for cohort picks")
    official = Official(store)
    members = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        for page in range(1, 1 + (limit + 49) // 50):
            league = await official.fetch(
                client,
                f"leagues-classic/{league_id}/standings/?page_standings={page}",
                f"cohort_standings_{league_id}_{page}",
            )
            if not league:
                break
            members.extend(league["standings"]["results"])
            if not league["standings"]["has_next"]:
                break
        members = members[:limit]
        results = await asyncio.gather(
            *(
                official.fetch(
                    client,
                    f"entry/{row['entry']}/event/{event}/picks/",
                    f"cohort_picks_{row['entry']}_{event}",
                )
                for row in members
            )
        )
    successful = [
        (member, result) for member, result in zip(members, results) if result and result.get("picks")
    ]
    players = sorted({p["element"] for _, result in successful for p in result["picks"]})
    exposures = np.zeros((len(successful), len(players)))
    index = {p: i for i, p in enumerate(players)}
    for row, (_, result) in enumerate(successful):
        for pick in result["picks"]:
            exposures[row, index[pick["element"]]] = pick["multiplier"]
    uncertainty = []
    rng = np.random.default_rng(11)
    if len(successful):
        samples = np.array(
            [
                exposures[rng.integers(0, len(successful), len(successful))].mean(axis=0) * 100
                for _ in range(300)
            ]
        )
        for column, p in enumerate(players):
            uncertainty.append(
                {
                    "player_id": p,
                    "effective_ownership": float(exposures[:, column].mean() * 100),
                    "resampling_interval_95": np.quantile(samples[:, column], [0.025, 0.975]).tolist(),
                }
            )
    report = {
        "at": now(),
        "kind": "current_rank_sample",
        "selection": "first requested entries in official league rank order",
        "league_id": league_id,
        "requested": limit,
        "retrieved": len(successful),
        "misses": len(members) - len(successful),
        "event": event,
        "population": "current rank cohort; no prior-season elite label or future-rank selection",
        "sampled_entries": [member["entry"] for member, _ in successful],
        "effective_ownership": uncertainty,
        "finalized": any(e["id"] == event and e.get("data_checked") for e in market["events"]),
        "errors": official.errors,
        "limitations": [
            "Intervals describe entry resampling, not a representative overall-rank population",
            "Current-rank samples cannot establish prior-season skill or future top-1% probabilities",
        ],
    }
    report["id"] = store.put("cohort", report)
    atomic_json(
        Path("artifacts/evaluations/cohort_coverage.json"),
        {k: v for k, v in report.items() if k != "sampled_entries"},
    )
    return report
