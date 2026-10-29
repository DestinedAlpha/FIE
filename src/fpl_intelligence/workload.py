"""All-competition enrichment with conservative time and identity checks."""

from __future__ import annotations

import asyncio
import csv
import io
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .storage import Store, atomic_json, now

REPO = "olbauday/FPL-Core-Insights"


def time_value(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def valid_fixture(row):
    identity = row.get("match_id", "")
    match = re.search(r"(\d{4}-\d{2}-\d{2})$", identity)
    if not row.get("kickoff_time"):
        return False, "missing cross-checkable kickoff"
    try:
        kickoff = time_value(row["kickoff_time"])
        stated = datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc) if match else kickoff
    except ValueError:
        return False, "invalid kickoff"
    if abs((kickoff.date() - stated.date()).days) > 1:
        return False, "kickoff conflicts with match identity date"
    return True, "UTC upstream timestamp; identifier date cross-check passed"


async def import_workload(store: Store, seasons=("2025-2026", "2026-2027")):
    base = f"https://api.github.com/repos/{REPO}/"
    async with httpx.AsyncClient(timeout=35, follow_redirects=False) as client:
        # A single public tree request; user installation needs no GitHub login.
        response = await client.get(base + "git/trees/main?recursive=1")
        response.raise_for_status()
        tree = response.json()
        if tree.get("truncated"):
            raise ValueError("Workload tree is incomplete")
        commit_response = await client.get(base + "commits/main")
        commit_response.raise_for_status()
        commit = commit_response.json()["sha"]
        paths = [
            r["path"]
            for r in tree["tree"]
            if any(r["path"].startswith("data/" + season + "/") for season in seasons)
            and "/By Tournament/" in r["path"]
            and "/Premier League/" not in r["path"]
            and r["path"].rsplit("/", 1)[-1] in ("fixtures.csv", "playermatchstats.csv")
        ]
        paths += ["data/" + season + "/players.csv" for season in seasons]
        semaphore = asyncio.Semaphore(4)
        bodies = {}
        errors = []

        async def one(path):
            async with semaphore:
                url = f"https://raw.githubusercontent.com/{REPO}/{commit}/" + path.replace(" ", "%20")
                try:
                    r = await client.get(url)
                    r.raise_for_status()
                    if len(r.content) > 8_000_000:
                        raise ValueError("Workload file exceeds byte budget")
                    rows = list(csv.DictReader(io.StringIO(r.text)))
                    bodies[path] = rows
                    store.snapshot(
                        "workload_" + __import__("hashlib").sha256(path.encode()).hexdigest()[:20],
                        url,
                        {"rows": rows, "commit": commit},
                    )
                except (httpx.HTTPError, ValueError) as error:
                    errors.append({"path": path, "error": str(error)[:140]})

        await asyncio.gather(*(one(path) for path in paths))
    codes = {
        season: {
            int(r["player_id"]): int(r["player_code"])
            for r in bodies.get("data/" + season + "/players.csv", [])
            if r.get("player_id") and r.get("player_code")
        }
        for season in seasons
    }
    accepted = []
    rejected = []
    future = []
    for path, rows in bodies.items():
        if not path.endswith("/playermatchstats.csv"):
            continue
        fixtures = {r["match_id"]: r for r in bodies.get(path.rsplit("/", 1)[0] + "/fixtures.csv", [])}
        season = path.split("/")[1]
        competition = path.split("/")[3]
        for row in rows:
            match = fixtures.get(row.get("match_id"))
            if match is None:
                rejected.append({"match": row.get("match_id"), "reason": "unmatched fixture"})
                continue
            valid, reason = valid_fixture(match)
            if not valid:
                rejected.append({"match": row.get("match_id"), "reason": reason})
                continue
            try:
                identity = int(row["player_id"])
                code = codes[season][identity]
                if row.get("minutes_played") in (None, ""):
                    raise ValueError("missing minutes")
                minutes = float(row["minutes_played"])
                if not 0 <= minutes <= 150:
                    raise ValueError("invalid minutes")
                kickoff = time_value(match["kickoff_time"])
            except (ValueError, KeyError):
                rejected.append(
                    {"match": row.get("match_id"), "reason": "unverified player identity or minutes"}
                )
                continue
            record = {
                "season": season[:4] + "/" + season[-2:],
                "player_code": code,
                "player_id": identity,
                "match_id": row["match_id"],
                "competition": competition,
                "kickoff": kickoff.isoformat(),
                "known_after": (kickoff + timedelta(hours=4)).isoformat(),
                "minutes": minutes,
                "extra_time": max(0, minutes - 90),
                "goals": row.get("goals"),
                "xg": row.get("xg"),
                "xa": row.get("xa"),
                "source_commit": commit,
                "temporal_track": "retrospective completed match facts; conservative four-hour completion cutoff",
            }
            if str(match.get("finished", "")).lower() == "true":
                accepted.append(record)
            else:
                future.append(record)
    accepted = list({(r["season"], r["player_code"], r["match_id"]): r for r in accepted}.values())
    result = {
        "at": now(),
        "source": REPO,
        "commit": commit,
        "requested_files": len(paths),
        "retrieved_files": len(bodies),
        "records": accepted,
        "scheduled": future,
        "accepted_player_matches": len(accepted),
        "rejected_player_matches": len(rejected),
        "rejections": rejected,
        "errors": errors,
        "zero_minutes_policy": "Explicit minutes_played=0 is a bench nonappearance; never derived from start/finish timestamps",
        "limits": [
            "Retrospective match enrichment, not original pre-deadline source capture",
            "Conflicting fixture dates are excluded, not guessed",
            "Unobserved competitions, travel and international minutes remain missing",
        ],
    }
    result["id"] = store.put("workload", result)
    atomic_json(
        Path("artifacts/evaluations/workload_coverage.json"),
        {k: v for k, v in result.items() if k not in ("records", "scheduled", "rejections", "errors")},
    )
    return result


def workload_features(records, code, cutoff):
    cutoff = time_value(cutoff)
    eligible = [r for r in records if r["player_code"] == code and time_value(r["known_after"]) < cutoff]
    recent = [r for r in eligible if cutoff - time_value(r["kickoff"]) <= timedelta(days=14)]
    if not eligible:
        return [0.0] * 6 + [0.0]
    seven = [r for r in recent if cutoff - time_value(r["kickoff"]) <= timedelta(days=7)]
    rest = min(21, (cutoff - max(time_value(r["kickoff"]) for r in eligible)).total_seconds() / 86400)
    return [
        sum(r["minutes"] for r in seven) / 240,
        sum(r["minutes"] for r in recent) / 480,
        len(seven) / 3,
        len(recent) / 5,
        sum(r["extra_time"] for r in recent) / 120,
        rest / 21,
        1.0,
    ]
