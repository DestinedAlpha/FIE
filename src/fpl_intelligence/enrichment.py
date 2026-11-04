"""Identical historical/live shot, tactical-role and disciplinary feature queries."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime

import numpy as np

from .attack import FEATURES as ATTACK_FEATURES
from .attack import attack_vector

FEATURES = (
    *ATTACK_FEATURES,
    "report_age_days",
    "injury_reported",
    "season_cautions",
    "recent_dismissals",
    "cautions_to_next_threshold",
    "season_fixture_fraction",
    "role_centre_back",
    "role_full_back",
    "role_central_midfield",
    "role_attacking_midfield",
    "role_forward",
    "shots_observed",
    "history_observed",
)


def epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def role_vector(role):
    role = (role or "").upper()
    if role == "DC":
        return [1, 0, 0, 0, 0]
    if role in ("DL", "DR"):
        return [0, 1, 0, 0, 0]
    if role in ("MC", "ML", "MR", "DMC", "DML", "DMR"):
        return [0, 0, 1, 0, 0]
    if role.startswith("AM"):
        return [0, 0, 0, 1, 0]
    if role.startswith("F"):
        return [0, 0, 0, 0, 1]
    return [0, 0, 0, 0, 0]


class EnrichmentIndex:
    def __init__(self, store, official_rows=(), temporal_model_id=None):
        self.shots = defaultdict(list)
        self.official = defaultdict(list)
        self.cache = {}
        self.lineage = {}
        self.params = {"decay": 0.94, "finishing_strength": None}
        if store.list("shot_archive", 1):
            archive = store.get("shot_archive")
            self.lineage["shot_archive"] = archive["id"]
            for row in archive["records"]:
                self.shots[row["player_code"]].append(row)
        if store.list("temporal_model", 1):
            temporal = store.get("temporal_model", temporal_model_id)
            self.params = temporal["adopted"]
            self.lineage["temporal_model"] = temporal["id"]
        for values in self.shots.values():
            values.sort(key=lambda r: epoch(r["known_after"]))
        self.shot_dates = {
            code: [epoch(r["known_after"]) for r in values] for code, values in self.shots.items()
        }
        for row in official_rows:
            if row.get("code") and row.get("kickoff_time"):
                self.official[int(row["code"])].append(row)
        for values in self.official.values():
            values.sort(key=lambda r: epoch(r["kickoff_time"]) + 4 * 3600)
        self.official_dates = {
            code: [epoch(r["kickoff_time"]) + 4 * 3600 for r in values]
            for code, values in self.official.items()
        }

    def vector(self, code, position, cutoff, season, packet=None):
        packet = packet or {}
        at = epoch(cutoff)
        code = int(code or 0)
        key = (code, int(position), cutoff, season, packet.get("news_added"), packet.get("status"))
        if key in self.cache:
            return self.cache[key]
        count = bisect_left(self.shot_dates.get(code, []), at)
        recent = self.shots.get(code, [])[:count][-24:]
        official_count = bisect_left(self.official_dates.get(code, []), at)
        official = self.official.get(code, [])[:official_count]
        current = [r for r in official if r["season"] == season]
        yellows = sum(float(r.get("yellow_cards") or 0) for r in current)
        reds = sum(float(r.get("red_cards") or 0) for r in official[-8:])
        games = len({r["fixture"] for r in current})
        threshold = 5 if games < 19 else 10 if games < 32 else 15
        added = packet.get("news_added")
        age = max(0, (at - epoch(added)) / 86400) if added and epoch(added) <= at else 0
        roles = (
            np.mean([role_vector(r.get("football_role")) for r in recent[-12:]], axis=0)
            if recent
            else np.zeros(5)
        )
        result = np.concatenate(
            [
                attack_vector(recent, position, **self.params),
                [
                    min(age, 365),
                    float(packet.get("status") in ("i", "d")),
                    yellows,
                    reds,
                    max(0, threshold - yellows),
                    min(games, 38) / 38,
                ],
                roles,
                [float(bool(recent)), float(bool(official))],
            ]
        ).astype(np.float32)
        self.cache[key] = result
        return result
