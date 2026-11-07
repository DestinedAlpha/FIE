"""Dated free shot/roster observations; conservative fixture-and-club identity joins."""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import httpx

from .decision import canonical_rows
from .ingestion import alias, load_market
from .storage import atomic_json, now
from .transport import PublicTransport

CHANNELS = ("open_play", "set_piece", "penalty")
TEAM_ALIASES = {
    "manchestercity": "mancity",
    "manchesterunited": "manutd",
    "newcastleunited": "newcastle",
    "nottinghamforest": "nottmforest",
    "wolverhamptonwanderers": "wolves",
    "tottenham": "spurs",
    "tottenhamhotspur": "spurs",
    "brightonandhovealbion": "brighton",
    "westhamunited": "westham",
}


def club_key(name):
    key = alias(name)
    return TEAM_ALIASES.get(key, key)


def channel(situation):
    return "penalty" if situation == "Penalty" else "open_play" if situation == "OpenPlay" else "set_piece"


def identify(name, position, candidates):
    norm = alias(name)
    scored = []
    words = name.casefold().split()
    for row in candidates:
        full = str(row.get("name") or "")
        other = full.casefold().split()
        if not other:
            continue
        exact = norm == alias(full)
        tail = bool(words and alias(words[-1]) == alias(other[-1]))
        first = bool(words and alias(words[0]) == alias(other[0]))
        score = (
            1.0
            if exact
            else 0.55 * SequenceMatcher(None, norm, alias(full)).ratio() + 0.3 * tail + 0.15 * first
        )
        expected = 1 if position == "GK" else 2 if position in ("DC", "DL", "DR") else None
        if expected and int(row["position"]) != expected:
            continue
        scored.append((score, int(row["element"]), row))
    scored.sort(key=lambda r: (-r[0], r[1]))
    if scored and scored[0][0] >= 0.68 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.12):
        return scored[0][2]
    return None


def validate_payload(data, *, league=False):
    """Reject soft errors and changed schemas before publishing a source snapshot."""
    try:
        if not isinstance(data, dict):
            raise ValueError("Expected an object")
        if league:
            if not isinstance(data["dates"], list):
                raise ValueError("Expected match dates")
            for match in data["dates"]:
                int(match["id"])
                datetime.fromisoformat(match["datetime"])
                if not isinstance(match["isResult"], bool):
                    raise ValueError("Expected a result flag")
                for side in ("h", "a"):
                    if not isinstance(match[side]["title"], str):
                        raise ValueError("Expected a club name")
        else:
            for side in ("h", "a"):
                roster, shots = data["rosters"][side], data["shots"][side]
                if not isinstance(roster, dict) or not isinstance(shots, list):
                    raise ValueError("Expected rosters and shots")
                for person in roster.values():
                    int(person["player_id"])
                    if not all(isinstance(person[k], str) for k in ("player", "position")):
                        raise ValueError("Expected player identity and role")
                    for key in ("time", "xA", "key_passes", "xGChain", "xGBuildup"):
                        value = float(person[key])
                        if not math.isfinite(value) or value < 0:
                            raise ValueError("Invalid roster statistic")
                    int(person["key_passes"])
                for shot in shots:
                    int(shot["player_id"])
                    if not all(isinstance(shot[k], str) for k in ("situation", "result")):
                        raise ValueError("Expected shot situation and result")
                    if shot["situation"] not in (
                        "OpenPlay",
                        "Penalty",
                        "FromCorner",
                        "SetPiece",
                        "DirectFreekick",
                    ):
                        raise ValueError("Unknown shot situation; refusing to infer an attacking channel")
                    value = float(shot["xG"])
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Invalid shot xG")
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Understat {'league' if league else 'match'} response has an invalid schema: {exc}"
        ) from exc
    return data


async def collect_shots(store, seasons=None):
    from pathlib import Path

    rows, _, _ = canonical_rows(store)
    players, market = load_market(store)
    seasons = seasons or sorted({int(r["season"][:4]) for r in rows})
    groups = defaultdict(list)
    for row in rows:
        if not row.get("name") and row["season"] == market["season"] and int(row["element"]) in players:
            row["name"] = players[int(row["element"])].name
        if row.get("kickoff_time"):
            groups[row["season"], str(row["kickoff_time"])[:10], club_key(row["team"])].append(row)
    transport = PublicTransport(store, 3, 0.3)
    records, mappings, errors = [], [], []
    cutoff = datetime.now(timezone.utc)
    async with httpx.AsyncClient(
        timeout=35,
        headers={"X-Requested-With": "XMLHttpRequest", "User-Agent": "FPL-Intelligence personal research"},
    ) as client:
        for year in seasons:
            season = f"{year}/{str(year + 1)[2:]}"
            url = f"https://understat.com/getLeagueData/EPL/{year}"
            try:
                league = validate_payload(
                    await transport.get(client, url, f"shot_league_{year}"), league=True
                )
                store.snapshot(f"shot_league_{year}", url, league)
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(
                    {
                        "season": season,
                        "error": type(exc).__name__,
                        "detail": str(exc),
                        "fallback": "Last verified league snapshot",
                    }
                )
                try:
                    league, _ = store.latest(f"shot_league_{year}")
                    validate_payload(league, league=True)
                except ValueError as cached:
                    errors.append({"season": season, "fallback_error": str(cached)})
                    continue

            async def one(match):
                if not match.get("isResult"):
                    return
                kickoff = datetime.fromisoformat(match["datetime"]).replace(tzinfo=timezone.utc)
                if kickoff + timedelta(hours=4) >= cutoff:
                    return
                source = f"shot_match_{match['id']}"
                url = f"https://understat.com/getMatchData/{match['id']}"
                try:
                    try:
                        data, _ = store.latest(source)
                        validate_payload(data)
                    except ValueError:
                        data = validate_payload(await transport.get(client, url, source))
                        store.snapshot(source, url, data)
                    for side in ("h", "a"):
                        team = match[side]["title"]
                        candidates = groups.get((season, kickoff.date().isoformat(), club_key(team)), [])
                        roster = data["rosters"][side]
                        totals: dict[str, dict[str, float]] = defaultdict(lambda: {k: 0.0 for k in CHANNELS})
                        goals: dict[str, dict[str, int]] = defaultdict(lambda: {k: 0 for k in CHANNELS})
                        counts: dict[str, dict[str, int]] = defaultdict(lambda: {k: 0 for k in CHANNELS})
                        for shot in data["shots"][side]:
                            pid, kind = str(shot["player_id"]), channel(shot["situation"])
                            totals[pid][kind] += float(shot["xG"])
                            counts[pid][kind] += 1
                            goals[pid][kind] += int(shot["result"] == "Goal")
                        for person in roster.values():
                            row = identify(person["player"], person["position"], candidates)
                            if row is None or not row.get("code"):
                                mappings.append(
                                    {
                                        "source_player": person["player_id"],
                                        "match": match["id"],
                                        "status": "unresolved",
                                    }
                                )
                                continue
                            pid = str(person["player_id"])
                            records.append(
                                {
                                    "season": season,
                                    "player_code": int(row["code"]),
                                    "player_id": int(row["element"]),
                                    "fixture_id": int(row["fixture"]),
                                    "event": int(row["round"]),
                                    "position": int(row["position"]),
                                    "team": row["team"],
                                    "source_player_id": pid,
                                    "kickoff": kickoff.isoformat(),
                                    "known_after": (kickoff + timedelta(hours=4)).isoformat(),
                                    "minutes": float(person["time"]),
                                    "football_role": person["position"],
                                    "xg": totals[pid],
                                    "goals": goals[pid],
                                    "shots": counts[pid],
                                    "xa": float(person["xA"]),
                                    "key_passes": int(person["key_passes"]),
                                    "xg_chain": float(person["xGChain"]),
                                    "xg_buildup": float(person["xGBuildup"]),
                                    "source": source,
                                    "identity": "unique full/partial name, club, fixture-date and compatible role",
                                }
                            )
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    errors.append({"match": match["id"], "error": type(exc).__name__, "detail": str(exc)})

            await asyncio.gather(*(one(match) for match in league["dates"]))
            print("Completed shot season", season, "matched observations", len(records), flush=True)
    keys = [(r["season"], r["player_code"], r["fixture_id"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Ambiguous shot identity would duplicate a player/fixture")
    if not records:
        store.put(
            "shot_refresh",
            {
                "at": now(),
                "errors": errors,
                "status": "No valid new observations; retained last good archive",
            },
        )
        if store.list("shot_archive", 100):
            for item in store.list("shot_archive", 100):
                previous = store.get("shot_archive", item["id"])
                if previous.get("matched", 0) > 0:
                    # Re-publish the retained payload as the active pointer;
                    # its source observations and dates are left unchanged.
                    retained = {k: v for k, v in previous.items() if k != "id"}
                    retained["refresh_attempt"] = {
                        "at": now(),
                        "errors": errors,
                        "retained_archive_id": previous["id"],
                    }
                    retained["id"] = store.put("shot_archive", retained)
                    return retained
        raise ValueError("No usable shot observations and no last good archive")
    # A partial provider recovery must not discard verified seasons or matches
    # absent from this attempt. New validated observations supersede the same key.
    fresh = {(r["season"], r["player_code"], r["fixture_id"]): r for r in records}
    retained_count, retained_id = 0, None
    for item in store.list("shot_archive", 100):
        previous = store.get("shot_archive", item["id"])
        if previous.get("matched", 0) > 0:
            for row in previous["records"]:
                key = row["season"], row["player_code"], row["fixture_id"]
                if key not in fresh:
                    fresh[key] = row
                    retained_count += 1
            retained_id = previous["id"]
            break
    records = [fresh[key] for key in sorted(fresh)]
    report = {
        "at": now(),
        "records": records,
        "matched": len(records),
        "unresolved": len(mappings),
        "errors": errors,
        "retained_observations": retained_count,
        "retained_archive_id": retained_id,
        "source": "https://understat.com",
        "temporal_track": "Dated match observations imported retrospectively; feature queries require completion strictly before deadline",
        "coverage": {
            "current_players": len({r["player_code"] for r in records if r["season"] == market["season"]})
        },
    }
    report["id"] = store.put("shot_archive", report)
    atomic_json(
        Path("artifacts/evaluations/shot_coverage.json"), {k: v for k, v in report.items() if k != "records"}
    )
    return report
