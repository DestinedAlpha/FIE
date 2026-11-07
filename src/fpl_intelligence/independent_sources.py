"""Public FiveStat comparison with identity, fixture and capture-time alignment.

This is independent forecast evidence, never an unvalidated production blend.
A publication display without a timezone does not establish historical availability.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta

import httpx
import numpy as np
from bs4 import BeautifulSoup

from .availability import moment
from .ingestion import alias, load_market
from .shot_data import club_key
from .storage import now
from .transport import PublicTransport

URL = "https://www.fivestat.co.uk/fpl"


def parse_fivestat(html, players, market):
    soup = BeautifulSoup(html, "html.parser")
    payload = None
    for script in soup.find_all("script"):
        match = re.search(r"\bconst\s+capData\s*=\s*", script.get_text())
        if match:
            payload, _ = json.JSONDecoder().raw_decode(script.get_text()[match.end() :])
            break
    if not isinstance(payload, list) or not payload:
        raise ValueError("FiveStat public projection schema unavailable")
    teams = {t["id"]: t for t in market["teams"]}
    position = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    records, rejected, keys = [], [], set()
    for packet in payload:
        pid = packet.get("fpl_id")
        player = players.get(pid)
        if (
            not player
            or alias(packet.get("name", "")) != alias(player.web_name)
            or club_key(packet.get("team", "")) != club_key(teams[player.team]["name"])
            or position.get(packet.get("position")) != player.position
        ):
            rejected.append({"player_id": pid, "reason": "Identity/club/position mismatch"})
            continue
        for fixture in packet.get("fixtures", []):
            event = fixture["gw"]
            eligible = []
            for f in market["fixtures"]:
                if f.get("event") != event or player.team not in (f["team_h"], f["team_a"]):
                    continue
                home = player.team == f["team_h"]
                opponent = teams[f["team_a"] if home else f["team_h"]]["short_name"]
                if fixture["label"] == f"{opponent} ({'H' if home else 'A'})":
                    eligible.append(f["id"])
            if len(eligible) != 1:
                rejected.append({"player_id": pid, "event": event, "reason": "Fixture mismatch"})
                continue
            values = {
                name: float(fixture[key])
                for name, key in [
                    ("xg", "proj_xg"),
                    ("xa", "proj_xa"),
                    ("minutes", "expected_minutes"),
                    ("points", "proj_fpl"),
                ]
            }
            cs = fixture.get("proj_cs")
            if (
                not np.isfinite([v for v in values.values() if v is not None]).all()
                or values["xg"] < 0
                or values["xa"] < 0
                or not 0 <= values["minutes"] <= 90
                or (cs is not None and not 0 <= float(cs) <= 1)
            ):
                raise ValueError("FiveStat invalid projection units or values")
            key = pid, eligible[0]
            if key in keys:
                raise ValueError("FiveStat duplicate player-fixture")
            keys.add(key)
            records.append(
                {
                    "player_id": pid,
                    "team_id": player.team,
                    "fixture_id": eligible[0],
                    "event": event,
                    **values,
                    "clean_sheet": None if cs is None else float(cs),
                }
            )
    if not records:
        raise ValueError("No FiveStat projections match the official market")
    updated = soup.select_one(".fpl-freshness")
    return {
        "records": records,
        "rejected": rejected,
        "page_updated_display": updated.get_text(" ", strip=True) if updated else None,
        "coverage": "Partial published player list; absent players are missing, never zero",
        "usage": "Independent diagnostic only; no validated blending weight",
    }


async def collect_fivestat(store):
    players, market = load_market(store)
    if store.list("independent_projections", 1):
        previous = store.get("independent_projections")
        if previous.get("season") == market["season"] and moment(now()) - moment(
            previous["captured_at"]
        ) < timedelta(hours=6):
            return previous
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        html = await PublicTransport(store).get(client, URL, "fivestat", json_body=False, max_bytes=4_000_000)
    result = parse_fivestat(html, players, market)
    snapshot = store.snapshot("fivestat_projections", URL, {"html": html})
    result.update(
        source_url=URL,
        raw_hash=snapshot["hash"],
        captured_at=snapshot["captured_at"],
        market_id=market["id"],
        season=market["season"],
    )
    result["id"] = store.put("independent_projections", result)
    return result


def compare_fivestat(store, forecast):
    if not store.list("independent_projections", 1):
        return {"status": "No independent projection capture"}
    source = store.get("independent_projections")
    cutoff = moment(forecast["information_cutoff"])
    age = cutoff - moment(source["captured_at"])
    if source["season"] != forecast["season"] or age.total_seconds() < 0 or age > timedelta(days=3):
        return {"status": "Capture outside forecast information window", "source_id": source["id"]}
    market = store.get("market", forecast["market_id"])
    teams = {p["id"]: p["team"] for p in market.get("players", [])}
    deadlines = {e["id"]: moment(e["deadline_time"]) for e in market["events"]}
    own = {(r["event"], r["player_id"], r["fixture_id"]): r for r in forecast["fixtures"]}
    records = []
    for row in source["records"]:
        key = row["event"], row["player_id"], row["fixture_id"]
        if (
            key not in own
            or (row.get("team_id") is not None and teams.get(row["player_id"]) != row["team_id"])
            or moment(source["captured_at"]) >= deadlines[row["event"]]
        ):
            continue
        reference = own[key]
        records.append(
            {
                **row,
                "fie_points": reference["mean"],
                "fie_minutes": reference["minutes"],
                "points_difference": reference["mean"] - row["points"],
                "minutes_difference": reference["minutes"] - row["minutes"],
            }
        )
    return {
        "status": "Aligned comparison" if records else "No pre-deadline aligned fixtures",
        "source_id": source["id"],
        "source_url": URL,
        "records": records,
        "scope": "Forecast disagreement, not observed accuracy; source xG/CS are not an attribution of FIE conditional points",
        "mean_absolute_points_difference": float(np.mean([abs(r["points_difference"]) for r in records]))
        if records
        else None,
    }
