"""Immutable completed-match views for the normal production engine in replay.

The backing archive can contain later labels. The only reader exposed to the
forecaster projects out every match not completed before its information cutoff.
Import time is retained separately; historical decisions are never prospective.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from .availability import moment
from .storage import atomic_json, now


def information_time(store):
    if store.list("replay_context", 1):
        return store.get("replay_context")["information_cutoff"]
    return now()


def finalization_clocks(rows):
    """Conservative historical finalization; live ingestion uses data_checked.

    Since 2026/27 a whole gameweek locks at 09:00 UK on the next day.
    Earlier seasons retain the four-hour post-kickoff reconstruction delay.
    This is not a substitute for a contemporaneous official finalized flag.
    """
    latest: dict[tuple[str, int], datetime] = {}
    for row in rows:
        if row.get("kickoff_time"):
            key = (row["season"], int(row["round"]))
            kickoff = moment(row["kickoff_time"])
            latest[key] = max(latest.get(key, kickoff), kickoff)
    london = ZoneInfo("Europe/London")
    return {
        key: datetime.combine(
            kickoff.astimezone(london).date() + timedelta(days=1), time(9), london
        ).astimezone(timezone.utc)
        for key, kickoff in latest.items()
        if key[0] >= "2026/27"
    }


def label_available_at(row, clocks):
    return clocks.get((row["season"], int(row["round"])), moment(row["kickoff_time"]) + timedelta(hours=4))


def completed_before(rows, cutoff, group):
    boundary = moment(cutoff)
    clocks = finalization_clocks(rows)
    return [
        row
        for row in rows
        if row.get("kickoff_time")
        and label_available_at(row, clocks) < boundary
        and f"{row['season']}-GW{int(row['round']):02d}" < group
    ]


def observation_view(store):
    if not store.list("observation_frame", 1):
        return None
    spec = store.get("observation_frame")
    context = store.get("replay_context")
    if spec["cutoff"] != context["information_cutoff"] or spec["group"] != context["group"]:
        raise ValueError("Observation view and replay clock disagree")
    path = store.artifact_path(spec["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("Historical observation archive checksum mismatch")
    return completed_before(pl.read_parquet(path).to_dicts(), spec["cutoff"], spec["group"])


def evaluation_report(store, filename, report):
    path = (
        store.root / "reports" / filename
        if store.list("replay_context", 1)
        or store.root.resolve() != Path(os.getenv("FPL_DATA_DIR") or "data").resolve()
        else Path("artifacts/evaluations") / filename
    )
    atomic_json(path, report)
