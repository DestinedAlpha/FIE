"""Exposure-shrunk, lag-only features; retrospective data is labeled as such."""

from __future__ import annotations

import math
from datetime import datetime

import duckdb
import numpy as np
import polars as pl

from .domain import Player
from .storage import Store, digest, now

PRIOR = {
    1: {"goals": 0.002, "assists": 0.008, "actions": 0, "saves": 3, "yellow": 0.04},
    2: {"goals": 0.045, "assists": 0.065, "actions": 8, "saves": 0, "yellow": 0.15},
    3: {"goals": 0.18, "assists": 0.14, "actions": 8, "saves": 0, "yellow": 0.13},
    4: {"goals": 0.35, "assists": 0.12, "actions": 5, "saves": 0, "yellow": 0.10},
}
FEATURE_NAMES = [
    "position",
    "home",
    "p_start",
    "p_cameo",
    "start_minutes",
    "goals90",
    "assists90",
    "actions90",
    "saves90",
    "yellow90",
    "points_mean",
    "exposure",
    "rest_days",
]


def number(row: dict, key: str, default=0.0) -> float:
    value = row.get(key)
    return default if value is None or value == "" else float(value)


def histories(store: Store, players: dict[int, Player], cutoff: str | None = None) -> dict[int, list[dict]]:
    result = {}
    for identity in players:
        try:
            body, _ = store.latest(f"history_{identity}", cutoff=cutoff)
            result[identity] = sorted(
                body["history"], key=lambda r: (r.get("kickoff_time") or "", r["fixture"])
            )
        except ValueError:
            result[identity] = []
    return result


def profile(position: int, rows: list[dict]) -> dict:
    rows = rows[-24:]
    weights = np.array([0.94**i for i in range(len(rows) - 1, -1, -1)])
    minutes = np.array([number(r, "minutes") for r in rows])
    exposure = float(np.dot(weights, minutes) / 90) if rows else 0
    prior = PRIOR[position]

    def rate(key, prior_value, strength=5):
        valid = [(r, w) for r, w in zip(rows, weights) if r.get(key) is not None]
        total = sum(number(r, key) * w for r, w in valid)
        exp = sum(number(r, "minutes") * w / 90 for r, w in valid)
        return (total + strength * prior_value) / (exp + strength)

    starts = np.array([number(r, "starts", float(number(r, "minutes") >= 60)) for r in rows])
    start_p = (float(np.dot(weights, starts)) + 1.2) / (float(weights.sum()) + 2)
    bench_exp = float(np.dot(weights, 1 - starts)) if rows else 0
    cameo_p = (sum(w for w, s, m in zip(weights, starts, minutes) if not s and m > 0) + 0.3) / (
        bench_exp + 1.5
    )
    starter_rows = [(number(r, "minutes"), w) for r, w in zip(rows, weights) if number(r, "starts") > 0]
    starter_minutes = (sum(m * w for m, w in starter_rows) + 2 * 78) / (sum(w for _, w in starter_rows) + 2)
    starter_long = (sum(w for m, w in starter_rows if m >= 60) + 2 * 0.9) / (
        sum(w for _, w in starter_rows) + 2
    )
    # Missing defensive-action history remains a typed measurement gap.
    actions_rows = [
        r
        for r in rows
        if r.get("clearances_blocks_interceptions") is not None
        and r.get("tackles") is not None
        and (position == 2 or r.get("recoveries") is not None)
    ]
    action_exp = sum(number(r, "minutes") / 90 for r in actions_rows)
    action_sum = sum(
        number(r, "clearances_blocks_interceptions")
        + number(r, "tackles")
        + (number(r, "recoveries") if position in (3, 4) else 0)
        for r in actions_rows
    )
    return {
        "p_start": float(np.clip(start_p, 0.015, 0.995)),
        "p_cameo": float(cameo_p),
        "start_minutes": float(np.clip(starter_minutes, 45, 90)),
        "p_long_start": float(starter_long),
        "goals90": rate("expected_goals", prior["goals"]),
        "assists90": rate("expected_assists", prior["assists"]) * 1.12,
        "actions90": (action_sum + 5 * prior["actions"]) / (action_exp + 5),
        "saves90": rate("saves", prior["saves"]),
        "yellow90": rate("yellow_cards", prior["yellow"]),
        "points_mean": (sum(number(r, "total_points") * w for r, w in zip(rows, weights)) + 3 * 2)
        / (float(weights.sum()) + 3),
        "exposure": exposure,
        "action_coverage": len(actions_rows) / max(1, len(rows)),
        "history_count": len(rows),
        "last_kickoff": rows[-1].get("kickoff_time") if rows else None,
    }


def vector(position: int, home: bool, p: dict, kickoff: str | None) -> list[float]:
    rest = 7.0
    if kickoff and p["last_kickoff"]:
        rest = min(
            30,
            max(
                0,
                (
                    datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
                    - datetime.fromisoformat(p["last_kickoff"].replace("Z", "+00:00"))
                ).total_seconds()
                / 86400,
            ),
        )
    return [float(position), float(home), *[p[k] for k in FEATURE_NAMES[2:-1]], rest]


def build_features(store: Store, players: dict[int, Player], market: dict) -> dict:
    all_rows = histories(store, players)
    records = []
    missing = []
    for identity, player in players.items():
        rows = all_rows[identity]
        if not rows:
            missing.append(identity)
        for row in rows:
            # Group strictly by decision deadline, so the first match of a DGW
            # cannot become a feature of the second match of that same deadline.
            event = int(row["round"])
            past = [r for r in rows if int(r["round"]) < event]
            p = profile(player.position, past)
            x = vector(player.position, bool(row["was_home"]), p, row.get("kickoff_time"))
            records.append(
                dict(zip(FEATURE_NAMES, x))
                | {
                    "player_id": identity,
                    "event": event,
                    "season": market["season"],
                    "deadline_group": f"{market['season']}-GW{event:02d}",
                    "fixture_id": row["fixture"],
                    "target": number(row, "total_points"),
                    "minutes_target": number(row, "minutes"),
                    "start_target": number(row, "starts"),
                    "baseline": p["points_mean"],
                    "known_at": market["known_at"],
                }
            )
    from collections import defaultdict

    from .history import archived_rows

    archive_groups = defaultdict(list)
    for row in archived_rows(store):
        archive_groups[row["season"], row["element"]].append(row)
    for (season, identity), rows in archive_groups.items():
        rows = sorted(rows, key=lambda r: (r["round"], r["fixture"]))
        for row in rows:
            event = int(row["round"])
            past = [r for r in rows if r["round"] < event]
            p = profile(row["position"], past)
            x = vector(row["position"], row["was_home"], p, row.get("kickoff_time"))
            records.append(
                dict(zip(FEATURE_NAMES, x))
                | {
                    "player_id": identity,
                    "event": event,
                    "season": season,
                    "deadline_group": f"{season}-GW{event:02d}",
                    "fixture_id": row["fixture"],
                    "target": number(row, "total_points"),
                    "minutes_target": number(row, "minutes"),
                    "start_target": number(row, "starts"),
                    "baseline": p["points_mean"],
                    "known_at": market["known_at"],
                }
            )
    if not records:
        raise ValueError("No historical rows; ingest with histories enabled")
    path = store.root / "features" / "lagged.parquet"
    path.parent.mkdir(exist_ok=True)
    pl.DataFrame(records).write_parquet(path)
    # SQL is the temporal feature inspection boundary; no current injury/price
    # columns are present in a historical training table.
    with duckdb.connect() as db:
        summary = db.execute(
            "SELECT event, count(*), avg(target) FROM read_parquet(?) GROUP BY event ORDER BY event",
            [str(path)],
        ).fetchall()
    result = {
        "created_at": now(),
        "rows": len(records),
        "features": FEATURE_NAMES,
        "path": str(path),
        "hash": digest(records),
        "events": summary,
        "missing_histories": missing,
        "track": "historical_statistical_replay",
        "point_in_time_news_available": False,
        "registration_caveat": "Current season registration; same-season position changes are not reconstructed",
    }
    result["id"] = store.put("features", result)
    return result


def team_rates(fixtures: list[dict], teams: list[dict]) -> dict[int, dict]:
    complete = [f for f in fixtures if f.get("finished") and f.get("team_h_score") is not None]
    mean = (sum(f["team_h_score"] + f["team_a_score"] for f in complete) + 60 * 1.4) / (
        2 * len(complete) + 60
    )
    output = {}
    for team in teams:
        rows = [(f, f["team_h"] == team["id"]) for f in complete if team["id"] in (f["team_h"], f["team_a"])]
        scored = sum(f["team_h_score"] if home else f["team_a_score"] for f, home in rows)
        conceded = sum(f["team_a_score"] if home else f["team_h_score"] for f, home in rows)
        output[team["id"]] = {
            "attack": (scored + 8 * mean) / (len(rows) + 8) / mean,
            "defence": (conceded + 8 * mean) / (len(rows) + 8) / mean,
            "mean": mean,
        }
    return output


def intensity(home: int, away: int, rates: dict[int, dict], horizon: int) -> tuple[float, float]:
    # Team-parameter mean reversion, not duplicated copies of today's xPts.
    shrink = math.exp(-horizon / 45)
    h, a = rates[home], rates[away]
    first = h["mean"] * (1 + (h["attack"] - 1) * shrink) * (1 + (a["defence"] - 1) * shrink) * 1.12
    second = a["mean"] * (1 + (a["attack"] - 1) * shrink) * (1 + (h["defence"] - 1) * shrink) * 0.88
    return float(np.clip(first, 0.25, 4.5)), float(np.clip(second, 0.25, 4.5))
