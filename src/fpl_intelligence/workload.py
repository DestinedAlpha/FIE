"""All-competition enrichment with conservative time and identity checks."""

from __future__ import annotations

import asyncio
import csv
import io
import re
from datetime import datetime, timedelta, timezone

import httpx

from .storage import Store, now

REPO = "olbauday/FPL-Core-Insights"


def records(store: Store, identity: str | None = None):
    """Missing optional input is masked; a missing pinned artifact is an error."""
    if identity is None and not store.list("workload", 1):
        return []
    return store.get("workload", identity)["records"]


def canonical_player_matches(records):
    """Deduplicate identical exports; quarantine every conflicting copy."""
    from collections import defaultdict

    groups = defaultdict(list)
    fixtures = defaultdict(set)
    for row in records:
        groups[row["season"], row["player_code"], row["match_id"]].append(row)
        if row.get("fixture_signature"):
            fixtures[row["season"], row["match_id"]].add(row["fixture_signature"])
    accepted, rejected = [], []
    for copies in groups.values():
        signatures = {
            tuple(
                row.get(k)
                for k in (
                    "competition",
                    "kickoff",
                    "minutes",
                    "extra_time",
                    "goals",
                    "xg",
                    "xa",
                    "fixture_signature",
                )
            )
            for row in copies
        }
        if len(signatures) > 1 or len(fixtures[copies[0]["season"], copies[0]["match_id"]]) > 1:
            rejected.append(
                {
                    "match": copies[0]["match_id"],
                    "player_code": copies[0]["player_code"],
                    "reason": "conflicting duplicate player-match or fixture",
                }
            )
        else:
            accepted.append(copies[0])
    return accepted, rejected


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
    from .transport import PublicTransport

    transport = PublicTransport(store)
    base = f"https://api.github.com/repos/{REPO}/"
    async with httpx.AsyncClient(timeout=35, follow_redirects=False) as client:
        commit = (await transport.get(client, base + "commits/main", "workload_commit"))["sha"]
        tree = await transport.get(client, base + f"git/trees/{commit}?recursive=1", "workload_tree")
        if tree.get("truncated"):
            raise ValueError("Workload tree is incomplete")
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
                    text = await transport.get(
                        client, url, "workload_file", json_body=False, max_bytes=8_000_000
                    )
                    reader = csv.DictReader(io.StringIO(text))
                    required = (
                        {"player_id", "player_code"}
                        if path.endswith("/players.csv")
                        else {"match_id", "kickoff_time", "finished"}
                        if path.endswith("/fixtures.csv")
                        else {"match_id", "player_id", "minutes_played"}
                    )
                    if not required <= set(reader.fieldnames or []):
                        raise ValueError("Workload CSV schema missing required columns")
                    rows = list(reader)
                    bodies[path] = rows
                    store.snapshot(
                        "workload_" + __import__("hashlib").sha256(path.encode()).hexdigest()[:20],
                        url,
                        {"rows": rows, "commit": commit},
                    )
                except (httpx.HTTPError, ValueError) as error:
                    errors.append({"path": path, "error": str(error)[:140]})

        await asyncio.gather(*(one(path) for path in paths))
    if errors or len(bodies) != len(paths):
        report = {
            "at": now(),
            "status": "incomplete; preceding verified workload retained",
            "errors": errors,
            "retrieved_files": len(bodies),
            "requested_files": len(paths),
            "retained_id": store.get("workload")["id"] if store.list("workload", 1) else None,
        }
        report["id"] = store.put("workload_refresh", report)
        return report
    return normalize_workload(store, bodies, commit, seasons, paths)


def normalize_workload(store, bodies, commit, seasons, paths, fetched_at=None):
    """Only complete source batches reach the active aggregate registry."""
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
                "fixture_signature": __import__("hashlib")
                .sha256(
                    str(
                        sorted((k, str(v)) for k, v in match.items() if k not in ("event", "gameweek", "gw"))
                    ).encode()
                )
                .hexdigest(),
                "temporal_track": "retrospective completed match facts; conservative four-hour completion cutoff",
            }
            if str(match.get("finished", "")).lower() == "true":
                accepted.append(record)
            else:
                future.append(record)
    accepted, conflicts = canonical_player_matches(accepted)
    rejected.extend(conflicts)
    if not accepted and not future:
        raise ValueError("Workload batch contains no verified player-match observations or fixtures")
    result = {
        "at": now(),
        "source_fetched_at": fetched_at or now(),
        "source": REPO,
        "commit": commit,
        "requested_files": len(paths),
        "retrieved_files": len(bodies),
        "records": accepted,
        "scheduled": future,
        "accepted_player_matches": len(accepted),
        "rejected_player_matches": len(rejected),
        "rejections": rejected,
        "conflicting_duplicates": len(conflicts),
        "errors": [],
        "schema_version": "workload-registry-v2",
        "zero_minutes_policy": "Explicit minutes_played=0 is a bench nonappearance; never derived from start/finish timestamps",
        "limits": [
            "Retrospective match enrichment, not original pre-deadline source capture",
            "Conflicting fixture dates are excluded, not guessed",
            "Unobserved competitions, travel and international minutes remain missing",
        ],
    }
    result["id"] = store.put("workload", result)
    from .observations import evaluation_report

    evaluation_report(
        store,
        "workload_coverage.json",
        {k: v for k, v in result.items() if k not in ("records", "scheduled", "rejections", "errors")},
    )
    return result


def migrate_cached_workload(store):
    """Reparse legacy caches from their own pinned raw CSVs, without a network call."""
    from urllib.parse import unquote

    if not store.list("workload", 1):
        return None
    previous = store.get("workload")
    if previous.get("schema_version") == "workload-registry-v2" or all(
        r.get("fixture_signature") for r in previous["records"]
    ):
        return None
    prefix = f"https://raw.githubusercontent.com/{REPO}/{previous['commit']}/"
    with store.db() as db:
        sources = [
            r[0] for r in db.execute("SELECT DISTINCT source FROM snapshots WHERE source LIKE 'workload_%'")
        ]
    bodies = {}
    for source in sources:
        body, meta = store.latest(source)
        if meta["url"].startswith(prefix) and body.get("commit") == previous["commit"]:
            bodies[unquote(meta["url"][len(prefix) :])] = body["rows"]
    if len(bodies) != previous["requested_files"]:
        raise ValueError(
            "Legacy workload migration requires its complete retained source batch; refresh sources"
        )
    seasons = sorted({p.split("/")[1] for p in bodies})
    result = normalize_workload(
        store,
        bodies,
        previous["commit"],
        seasons,
        list(bodies),
        previous.get("source_fetched_at", previous["at"]),
    )
    store.put(
        "workload_migration",
        {
            "at": now(),
            "previous_id": previous["id"],
            "workload_id": result["id"],
            "source": "Reparsed retained pinned CSVs; previous archive preserved",
        },
    )
    return result


def refresh_workload_due(store):
    """Daily numerical-input refresh; failed attempts keep the last verified feed."""
    if store.list("workload_refresh_attempt", 1):
        previous = store.get("workload_refresh_attempt")
        if time_value(now()) - time_value(previous["at"]) < timedelta(
            hours=24 if previous.get("success") else 4
        ):
            return previous
    attempt = {"at": now(), "success": False, "status": "started"}
    store.put("workload_refresh_attempt", attempt)
    try:
        result = asyncio.run(import_workload(store))
        attempt.update(
            success="retained_id" not in result,
            status=result.get("status", "updated"),
            artifact_id=result["id"],
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        from .settings import redact

        attempt.update(status="unavailable; preceding workload retained", reason=redact(error)[:180])
    attempt["id"] = store.put("workload_refresh_attempt", attempt)
    return attempt


WORKLOAD_TRANSFORM = "positive-exposure-v2"


def workload_features(records, code, cutoff, transform_version=WORKLOAD_TRANSFORM):
    cutoff = time_value(cutoff)
    eligible = [r for r in records if r["player_code"] == code and time_value(r["known_after"]) < cutoff]
    recent = [r for r in eligible if cutoff - time_value(r["kickoff"]) <= timedelta(days=14)]
    if not eligible:
        return [0.0] * 6 + [0.0]
    if transform_version == WORKLOAD_TRANSFORM:
        # A named but unused substitute supplies coverage, not physical exposure.
        eligible = [r for r in eligible if r["minutes"] > 0]
        recent = [r for r in recent if r["minutes"] > 0]
    elif transform_version != "legacy-v1":
        raise ValueError("Unknown workload transform")
    seven = [r for r in recent if cutoff - time_value(r["kickoff"]) <= timedelta(days=7)]
    rest = (
        min(21, (cutoff - max(time_value(r["kickoff"]) for r in eligible)).total_seconds() / 86400)
        if eligible
        else 21
    )
    return [
        sum(r["minutes"] for r in seven) / 240,
        sum(r["minutes"] for r in recent) / 480,
        len(seven) / 3,
        len(recent) / 5,
        sum(r["extra_time"] for r in recent) / 120,
        rest / 21,
        1.0,
    ]
