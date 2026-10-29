"""A prior-season-qualified expert cohort and its released, lagged decisions."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from .ingestion import Official, load_market
from .storage import Store, atomic_json, now

URL = "https://plan.livefpl.net/elite"


def parse_hall(html: str, season: str):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table#main")
    if table is None:
        raise ValueError("Prior-season leaderboard table is missing")
    headers = [x.get_text(" ", strip=True) for x in table.select("th")]
    prior = [h for h in headers if re.fullmatch(r"\d{2}/\d{2}", h)]
    if not prior or any(int("20" + h[:2]) >= int(season[:4]) for h in prior):
        raise ValueError("Hall-of-fame ranking includes current/future-season results")
    result = []
    for row in table.select("tr"):
        cells = row.find_all("td", recursive=False)
        link = row.find("a", href=re.compile(r"fantasy\.premierleague\.com/(?:en/)?entry/\d+/history"))
        if len(cells) < 7 or link is None:
            continue
        entry_match = re.search(r"entry/(\d+)", str(link["href"]))
        if entry_match is None:
            continue
        ranks = []
        for cell in cells[4:7]:
            text = cell.get_text(strip=True).replace(",", "")
            ranks.append(int(text) if text.isdigit() else None)
        if sum(rank is not None and 0 < rank <= 10000 for rank in ranks) < 2:
            continue
        result.append(
            {
                "entry": int(entry_match.group(1)),
                "name": link.get_text(" ", strip=True),
                "published_prior_ranks": ranks,
                "published_prior_seasons": prior[:3],
            }
        )
    return result


async def collect_elite(store: Store, limit=100):
    if not 10 <= limit <= 500:
        raise ValueError("Expert cohort budget is 10–500 actually retrieved managers")
    _, market = load_market(store)
    first_deadline = min(e["deadline_time"] for e in market["events"])
    released = [e["id"] for e in market["events"] if e.get("data_checked")]
    official = Official(store)
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        response = await client.get(URL)
        response.raise_for_status()
        candidates = parse_hall(response.text, market["season"])[:limit]
        store.snapshot("elite_prior_leaderboard", URL, {"html": response.text, "season": market["season"]})
        histories = await asyncio.gather(
            *(
                official.fetch(client, f"entry/{r['entry']}/history/", f"elite_history_{r['entry']}")
                for r in candidates
            )
        )
        verified = []
        for entry, history in zip(candidates, histories):
            if not history:
                continue
            past = sorted(
                [r for r in history.get("past", []) if r.get("season_name", "") < market["season"]],
                key=lambda r: r["season_name"],
            )[-3:]
            if sum(0 < (r.get("rank") or 0) <= 10000 for r in past) >= 2:
                verified.append({**entry, "verified_prior_history": past})
        tasks = [(entry, event) for entry in verified for event in released]
        picks = await asyncio.gather(
            *(
                official.fetch(
                    client,
                    f"entry/{entry['entry']}/event/{event}/picks/",
                    f"elite_picks_{entry['entry']}_{event}",
                )
                for entry, event in tasks
            )
        )
    selections = []
    misses = []
    for (entry, event), body in zip(tasks, picks):
        if not body or not body.get("picks"):
            misses.append({"entry": entry["entry"], "event": event})
            continue
        for pick in body["picks"]:
            selections.append(
                {
                    "entry": entry["entry"],
                    "event": event,
                    "player_id": pick["element"],
                    "intended_xi": pick["position"] <= 11,
                    "captain": bool(pick["is_captain"]),
                    "effective_multiplier": pick["multiplier"],
                    "chip": body.get("active_chip"),
                }
            )
    report = {
        "at": now(),
        "season": market["season"],
        "cohort_definition": "At least two top-10,000 finishes in the three prior seasons; ranking source excludes current season",
        "qualification_cutoff": first_deadline,
        "source": URL,
        "candidate_count": len(candidates),
        "verified_managers": verified,
        "retrieved_managers": len(verified),
        "retrieved_manager_gameweeks": len(tasks) - len(misses),
        "misses": misses,
        "selections": selections,
        "errors": official.errors,
        "uses_current_rank_for_selection": False,
        "feature_policy": "At decision GW g, only intended ownership/XI/captain choices from GW < g; no realized autosubs as pre-deadline features",
        "temporal_track": "Retrospective published selections with verified prior qualification; local prospective capture starts now",
    }
    report["id"] = store.put("elite_cohort", report)
    atomic_json(
        Path("artifacts/evaluations/elite_coverage.json"),
        {k: v for k, v in report.items() if k not in ("verified_managers", "selections", "errors", "misses")},
    )
    return report


def elite_features(store: Store):
    report = store.get("elite_cohort")
    output = {}
    events = sorted({r["event"] for r in report["selections"]})
    for event in events:
        rows = [r for r in report["selections"] if r["event"] == event]
        managers = {r["entry"] for r in rows}
        count = len(managers)
        players = {r["player_id"] for r in rows}
        for player in players:
            own = [r for r in rows if r["player_id"] == player]
            output[report["season"], event + 1, player] = {
                "ownership": len(own) / count,
                "intended_xi": sum(r["intended_xi"] for r in own) / count,
                "captain": sum(r["captain"] for r in own) / count,
                "sample": count,
                "known_event": event,
            }
    return output, report
