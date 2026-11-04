"""Immutable completed-match views for the normal production engine in replay.

The backing archive can contain later labels. The only reader exposed to the
forecaster projects out every match not completed before its information cutoff.
Import time is retained separately; historical decisions are never prospective.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

import polars as pl

from .availability import moment
from .storage import atomic_json, now


def information_time(store):
    if store.list("replay_context", 1):
        return store.get("replay_context")["information_cutoff"]
    return now()


def completed_before(rows, cutoff, group):
    boundary = moment(cutoff)
    return [
        row
        for row in rows
        if row.get("kickoff_time")
        and moment(row["kickoff_time"]) + timedelta(hours=4) < boundary
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
        else Path("artifacts/evaluations") / filename
    )
    atomic_json(path, report)
