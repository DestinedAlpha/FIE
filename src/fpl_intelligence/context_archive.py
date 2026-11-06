"""Hash-verified historical context captured and committed before each deadline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import lzma
from datetime import datetime
from pathlib import Path

import httpx

from .storage import Store, digest, now

MANIFEST = Path(__file__).resolve().parents[2] / "configs" / "context_archive_manifest.json"
FIELDS = (
    "id",
    "code",
    "element_type",
    "team",
    "now_cost",
    "status",
    "news",
    "news_added",
    "chance_of_playing_next_round",
    "selected_by_percent",
    "transfers_in_event",
    "transfers_out_event",
    "penalties_order",
    "direct_freekicks_order",
    "corners_and_indirect_freekicks_order",
    "ep_next",
    "points_per_game",
    "form",
    "starts",
    "minutes",
    "can_select",
    "removed",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "expected_goals_per_90",
    "expected_assists_per_90",
    "expected_goal_involvements_per_90",
    "expected_goals_conceded_per_90",
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
    "defensive_contribution",
)
SCHEMA_VERSION = 4


def verified_cached_bootstrap(record, proof):
    """Reuse complete pinned inputs when widening the reduced context schema."""
    if record is None:
        return None
    original_proof = record.get("bootstrap_proof", {})
    if any(
        original_proof.get(key) != proof.get(key)
        for key in (
            "season",
            "event",
            "deadline",
            "sha256",
            "url",
            "commit",
            "source_captured_at",
            "git_committed_at",
        )
    ):
        return None
    body = record["bootstrap"]
    if digest(body) != record.get("bootstrap_hash"):
        raise ValueError("Cached deadline registry checksum mismatch")
    return body


def check_proof(row):
    deadline = datetime.fromisoformat(row["deadline"].replace("Z", "+00:00"))
    for key in ("source_captured_at", "git_committed_at"):
        timestamp = datetime.fromisoformat(row[key].replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp >= deadline:
            raise ValueError("Context file was not demonstrably available before its deadline")
    if not row["url"].startswith(
        "https://raw.githubusercontent.com/Randdalf/fplcache/" + row["commit"] + "/cache/"
    ):
        raise ValueError("Context archive is not pinned to the verified upstream commit")


async def import_context(store: Store):
    manifest = json.loads(MANIFEST.read_text())
    semaphore = asyncio.Semaphore(4)
    entries = []
    errors = []
    cached = {}
    for item in reversed(store.list("replay_inputs", 1000)):
        record = store.get("replay_inputs", item["id"])
        cached[record["season"], record["event"]] = record
    async with httpx.AsyncClient(timeout=35, follow_redirects=False) as client:

        async def one(row):
            check_proof(row)
            source = f"context_{row['season'].replace('/', '-')}_{row['event']}"
            try:
                body, capture_meta = store.latest(source)
                if (
                    body.get("compressed_sha256") == row["sha256"]
                    and body.get("schema_version") == SCHEMA_VERSION
                ):
                    return {
                        **row,
                        "source": source,
                        "hash": capture_meta["hash"],
                        "players": len(body["elements"]),
                    }
            except ValueError:
                pass
            async with semaphore:
                original = verified_cached_bootstrap(cached.get((row["season"], row["event"])), row)
                if original is None:
                    response = await client.get(row["url"])
                    response.raise_for_status()
                    if (
                        len(response.content) > 2_000_000
                        or hashlib.sha256(response.content).hexdigest() != row["sha256"]
                    ):
                        raise ValueError("Context archive checksum/size verification failed")
                    decoder = lzma.LZMADecompressor()
                    raw = decoder.decompress(response.content, max_length=16_000_000)
                    if not decoder.eof:
                        raise ValueError("Context archive decompression exceeds budget")
                    original = json.loads(raw)
                actual = next(e for e in original["events"] if e["id"] == row["event"])
                if actual["deadline_time"] != row["deadline"]:
                    raise ValueError("Archived and final deadline disagree; context needs a separate audit")
                elements = [{k: player.get(k) for k in FIELDS} for player in original["elements"]]
                body = {
                    "schema_version": SCHEMA_VERSION,
                    "elements": elements,
                    "season": row["season"],
                    "event": row["event"],
                    "deadline": row["deadline"],
                    "source_captured_at": row["source_captured_at"],
                    "git_committed_at": row["git_committed_at"],
                    "commit": row["commit"],
                    "compressed_sha256": row["sha256"],
                    "our_known_at": now(),
                }
                meta = store.snapshot(source, row["url"], body, row["source_captured_at"])
                return {**row, "source": source, "hash": meta["hash"], "players": len(elements)}

        async def guarded(row):
            try:
                result = await one(row)
                entries.append(result)
            except (ValueError, httpx.HTTPError, KeyError, lzma.LZMAError) as error:
                errors.append({"season": row["season"], "event": row["event"], "error": str(error)[:180]})

        await asyncio.gather(*(guarded(row) for row in manifest["snapshots"]))
    archived_count = len(entries)
    entries = add_local_context(store, entries)
    result = {
        "at": now(),
        "entries": entries,
        "errors": errors,
        "retrieved_deadlines": len(entries),
        "requested_deadlines": len(manifest["snapshots"]) + len(entries) - archived_count,
        "retrieved_archived_deadlines": archived_count,
        "locally_captured_deadlines": len(entries) - archived_count,
        "temporal_track": "upstream capture and containing Git commit both precede deadline; local historical import is later",
    }
    result["id"] = store.put("context_archive", result)
    return result


def load_context(store: Store, archive_id=None):
    report = store.get("context_archive", archive_id)
    result = {}
    for row in report["entries"]:
        body, _ = store.latest(row["source"], content_hash=row.get("hash"))
        for player in body["elements"]:
            if player.get("code"):
                result[body["season"], body["event"], int(player["code"])] = player
    return result


def capture_live_context(store: Store, bootstrap):
    """Freeze next-deadline context now, so future retraining can use our own proof."""
    from .ingestion import season_of

    event = next((e for e in bootstrap["events"] if e.get("is_next")), None)
    if event is None:
        return None
    captured = now()
    deadline = event["deadline_time"]
    if datetime.fromisoformat(captured) >= datetime.fromisoformat(deadline.replace("Z", "+00:00")):
        return None
    season = season_of(bootstrap)
    source = f"context_live_{season.replace('/', '-')}_{event['id']}"
    body = {
        "schema_version": SCHEMA_VERSION,
        "elements": [{k: p.get(k) for k in FIELDS} for p in bootstrap["elements"]],
        "season": season,
        "event": event["id"],
        "deadline": deadline,
        "our_known_at": captured,
        "source_captured_at": captured,
        "temporal_track": "Locally captured before target deadline",
    }
    return store.snapshot(source, "https://fantasy.premierleague.com/api/bootstrap-static/", body, captured)


def add_local_context(store, entries):
    indexed = {(r["season"], r["event"]): r for r in entries}
    with store.db() as db:
        rows = db.execute(
            "SELECT DISTINCT source FROM snapshots WHERE source LIKE 'context_live_%'"
        ).fetchall()
    for row in rows:
        latest, _ = store.latest(row["source"])
        key = (latest["season"], latest["event"])
        # Preserve the final locally observed registry before the deadline,
        # including if the current installation has already moved to next GW.
        body, meta = store.latest(row["source"], cutoff=latest["deadline"])
        if time_before(meta["captured_at"], body["deadline"]) and key not in indexed:
            source = row["source"]
            captured = meta["captured_at"]
            if body.get("schema_version") != SCHEMA_VERSION:
                # Old local news packets dropped fields which their original
                # full bootstrap still contains. Reconstruct from an ACTUALLY
                # captured earlier source; never borrow today's later season
                # totals or pretend the derived packet was captured then.
                try:
                    bootstrap, original = store.latest("bootstrap", cutoff=body["deadline"])
                    from .ingestion import season_of

                    target: dict = next((e for e in bootstrap["events"] if e.get("is_next")), {})
                    if (
                        season_of(bootstrap) == key[0]
                        and target.get("id") == key[1]
                        and time_before(original["captured_at"], body["deadline"])
                    ):
                        source = f"context_reconstructed_{key[0].replace('/', '-')}_{key[1]}"
                        try:
                            reconstructed, derived = store.latest(source)
                        except ValueError:
                            reconstructed, derived = {}, {}
                        if (
                            reconstructed.get("schema_version") != SCHEMA_VERSION
                            or reconstructed.get("origin_bootstrap_hash") != original["hash"]
                        ):
                            reconstructed = {
                                "schema_version": SCHEMA_VERSION,
                                "season": key[0],
                                "event": key[1],
                                "deadline": body["deadline"],
                                "elements": [
                                    {field: p.get(field) for field in FIELDS} for p in bootstrap["elements"]
                                ],
                                "origin_bootstrap_hash": original["hash"],
                                "source_captured_at": original["captured_at"],
                                "derived_at": now(),
                                "temporal_track": "Later transformation of an immutable locally captured pre-deadline bootstrap",
                            }
                            derived = store.snapshot(
                                source, original["url"], reconstructed, original["captured_at"]
                            )
                        body, meta, captured = reconstructed, derived, original["captured_at"]
                except (ValueError, KeyError):
                    pass
            indexed[key] = {
                "season": body["season"],
                "event": body["event"],
                "deadline": body["deadline"],
                "source": source,
                "hash": meta["hash"],
                "source_captured_at": captured,
                "players": len(body["elements"]),
                "temporal_track": body.get("temporal_track", "Locally observed pre-deadline context"),
            }
    return sorted(indexed.values(), key=lambda r: (r["season"], r["event"]))


def time_before(captured, deadline):
    return datetime.fromisoformat(captured.replace("Z", "+00:00")) < datetime.fromisoformat(
        deadline.replace("Z", "+00:00")
    )
