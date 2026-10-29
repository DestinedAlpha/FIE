"""Shared match worlds: coherent lineups, goal attribution and exposure scoring.

This is a transparent component baseline. BPS and action-count processes are
estimated proxies and are explicitly recorded, not claimed as official events.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .domain import Player
from .features import histories, intensity, profile, team_rates
from .ingestion import load_market
from .rules import Rules, scoring_ids
from .storage import Store, digest, now

MODEL_VERSION = "shared-components-v1"


def weighted_choice(rng, weights):
    total = weights.sum(axis=1)
    safe = np.where(total[:, None] > 0, weights, np.ones_like(weights))
    cdf = np.cumsum(safe, axis=1)
    draw = rng.random(len(weights)) * cdf[:, -1]
    return np.minimum((cdf < draw[:, None]).sum(axis=1), weights.shape[1] - 1)


def sample_lineup(
    rng, members: list[Player], profiles: dict, worlds: int, horizon: int, health: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(members)
    positions = np.array([p.position for p in members])
    chances = np.array(
        [p.chance if p.chance is not None else (1 if p.status == "a" else 0.5) for p in members], dtype=float
    )
    chances += (1 - chances) * (1 - np.exp(-horizon / 5))
    available = health < chances[None, :]
    raw = np.array([profiles[p.id]["p_start"] for p in members])
    # Shrink very long-horizon role certainty towards the club's finite XI places.
    p_start = raw * np.exp(-horizon / 50) + (11 / n) * (1 - np.exp(-horizon / 50))
    odds = p_start / np.clip(1 - p_start, 0.02, 1)
    scores = np.log(odds)[None, :] + rng.gumbel(size=(worlds, n))
    scores[~available] = -1e9
    selected = np.zeros((worlds, n), dtype=bool)
    for mask, count in [(positions == 1, 1), (positions != 1, 10)]:
        columns = np.where(mask)[0]
        if len(columns) < count:
            raise ValueError("Incomplete club registration for coherent real lineups")
        chosen = np.argpartition(scores[:, columns], -count, axis=1)[:, -count:]
        selected[np.arange(worlds)[:, None], columns[chosen]] = True
    selected &= available
    starting_counts = selected.sum(axis=1)
    start = np.zeros((worlds, n), dtype=np.int16)
    end = np.zeros_like(start)
    long_p = np.array([profiles[p.id]["p_long_start"] for p in members])
    duration = np.array([profiles[p.id]["start_minutes"] for p in members])
    long = rng.random((worlds, n)) < long_p[None, :]
    sampled = np.where(
        long,
        np.clip(np.rint(rng.normal(duration, 9, (worlds, n))), 60, 90),
        rng.integers(20, 60, (worlds, n)),
    ).astype(np.int16)
    sampled[:, positions == 1] = 90
    end[selected] = sampled[selected]
    # At most five substitutions, paired to an actual exiting player. GK and
    # outfield replacements are distinct; no starter can occupy two roles.
    for slot in range(5):
        departing_weights = ((end > 0) & (end < 90) & selected).astype(float)
        departing = weighted_choice(rng, departing_weights)
        has_departing = departing_weights.sum(axis=1) > 0
        departure_time = end[np.arange(worlds), departing]
        wanted_gk = positions[departing] == 1
        cameo = np.array([profiles[p.id]["p_cameo"] for p in members])
        eligible = available & ~selected & (end == 0)
        eligible &= (positions == 1)[None, :] == wanted_gk[:, None]
        weights = eligible * cameo[None, :]
        substitute = weighted_choice(rng, weights)
        use = has_departing & (weights.sum(axis=1) > 0)
        indices = np.where(use)[0]
        start[indices, substitute[use]] = departure_time[use]
        end[indices, substitute[use]] = 90
        # Mark the departed starter as dealt with, leaving their actual interval.
        selected[indices, departing[use]] = False
    short_starters = (start == 0) & (end > 0) & (end < 90) & selected
    # If the player is not replaced within five changes, assume they finish the
    # match; the simulator does not silently leave ten footballers on the pitch.
    end[short_starters] = 90
    return start, end, {"short_lineup_worlds": int((starting_counts < 11).sum())}


def fixture_worlds(
    rng,
    fixture: dict,
    players: dict[int, Player],
    profiles: dict,
    rates: dict,
    rules: Rules,
    worlds: int,
    horizon: int,
    health: dict,
):
    members = [
        p for p in players.values() if p.team in (fixture["team_h"], fixture["team_a"]) and p.selectable
    ]
    ids = [p.id for p in members]
    n = len(ids)
    pos = np.array([p.position for p in members])
    team = np.array([p.team for p in members])
    starts = np.zeros((worlds, n), dtype=np.int16)
    ends = np.zeros_like(starts)
    quality = []
    for club in (fixture["team_h"], fixture["team_a"]):
        columns = np.where(team == club)[0]
        a, b, q = sample_lineup(rng, [members[i] for i in columns], profiles, worlds, horizon, health[club])
        starts[:, columns], ends[:, columns] = a, b
        quality.append(q)
    played = ends > starts
    yellows = (
        rng.random((worlds, n))
        < np.array([profiles[p]["yellow90"] for p in ids])[None, :] * (ends - starts) / 90
    ) & played
    red = (rng.random((worlds, n)) < 0.004 * (ends - starts) / 90) & played
    red_time = rng.integers(1, 91, (worlds, n))
    ends = np.where(red, np.maximum(starts + 1, np.minimum(ends, red_time)), ends)
    minutes = ends - starts
    goals = np.zeros((worlds, n), dtype=np.int16)
    assists = np.zeros_like(goals)
    own = np.zeros_like(goals)
    conceded = np.zeros_like(goals)
    saves = np.zeros_like(goals)
    home_lambda, away_lambda = intensity(fixture["team_h"], fixture["team_a"], rates, horizon)
    goal_rates = np.array([profiles[p]["goals90"] for p in ids])
    assist_rates = np.array([profiles[p]["assists90"] for p in ids])
    # Shared parameter uncertainty is distinct from Poisson event randomness.
    uncertainty = 0.12 + 0.012 * horizon
    shape = 1 / uncertainty**2
    match_tempo = rng.gamma(shape, 1 / shape, worlds)
    totals = {}
    for club, other, goal_lambda in [
        (fixture["team_h"], fixture["team_a"], home_lambda),
        (fixture["team_a"], fixture["team_h"], away_lambda),
    ]:
        totals[club] = rng.poisson(goal_lambda * match_tempo)
        for goal_index in range(int(totals[club].max(initial=0))):
            occurred = totals[club] > goal_index
            moment = rng.integers(1, 91, worlds)
            on_pitch = (starts < moment[:, None]) & (ends >= moment[:, None])
            eligible = on_pitch & (team == club)[None, :]
            scorer = weighted_choice(rng, eligible * goal_rates[None, :])
            valid = occurred & eligible.any(axis=1)
            is_own = (rng.random(worlds) < 0.02) & valid
            scoring = np.where(valid & ~is_own)[0]
            goals[scoring, scorer[scoring]] += 1
            own_eligible = on_pitch & (team == other)[None, :]
            own_scorer = weighted_choice(rng, own_eligible * (np.where(pos <= 2, 1, 0.15))[None, :])
            own_indices = np.where(is_own & own_eligible.any(axis=1))[0]
            own[own_indices, own_scorer[own_indices]] += 1
            assist_eligible = eligible.copy()
            assist_eligible[np.arange(worlds), scorer] = False
            assistant = weighted_choice(rng, assist_eligible * assist_rates[None, :])
            given_assist = valid & ~is_own & (rng.random(worlds) < 0.74) & assist_eligible.any(axis=1)
            indices = np.where(given_assist)[0]
            assists[indices, assistant[indices]] += 1
            # Dismissed defenders remain liable for later conceded goals.
            liable = (team == other)[None, :] & (starts < moment[:, None]) & ((ends >= moment[:, None]) | red)
            conceded += (occurred[:, None] & liable).astype(np.int16)
        # Saved shots and goals jointly define shots on target. They are not
        # sampled independently for multiple keepers in the same team.
        saved_shots = rng.poisson((2.6 + 0.35 * goal_lambda) * match_tempo)
        for shot in range(int(saved_shots.max(initial=0))):
            moment = rng.integers(1, 91, worlds)
            keeper = (
                (team == other)[None, :]
                & (pos == 1)[None, :]
                & (starts < moment[:, None])
                & (ends >= moment[:, None])
            )
            saves += (keeper & (saved_shots > shot)[:, None]).astype(np.int16)
    p60 = minutes >= 60
    appeared = minutes > 0
    appearance = np.where(p60, rules.reward("long_play", 1, 2), appeared * rules.reward("short_play", 1, 1))
    goals_reward = np.array(
        [rules.reward("goals_scored", int(p), {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}) for p in pos]
    )
    clean_reward = np.array(
        [rules.reward("clean_sheets", int(p), {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}) for p in pos]
    )
    concede_reward = np.array(
        [rules.reward("goals_conceded", int(p), {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0}) for p in pos]
    )
    actions = rng.poisson(
        np.maximum(0, np.array([profiles[p]["actions90"] for p in ids])[None, :] * minutes / 90)
    )
    threshold = np.array([rules.dc_threshold[int(p)] for p in pos])
    dc_reward = np.array(
        [
            rules.reward("defensive_contribution", int(p), {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2})
            for p in pos
        ]
    )
    components = {
        "appearance": appearance,
        "goals": goals * goals_reward[None, :],
        "assists": assists * rules.reward("assists", 1, 3),
        "clean_sheet": (p60 & (conceded == 0)) * clean_reward[None, :],
        "conceded": (conceded // 2) * concede_reward[None, :],
        "saves": (saves // 3) * rules.reward("saves", 1, 1),
        "defcon": (actions >= threshold[None, :]) * dc_reward[None, :],
        "discipline": yellows * rules.reward("yellow_cards", 1, -1) + red * rules.reward("red_cards", 1, -3),
        "own_goals": own * rules.reward("own_goals", 1, -2),
    }
    bps = (
        appeared * 3
        + p60 * 3
        + goals * np.where(pos == 4, 24, np.where(pos == 3, 18, 12))[None, :]
        + assists * 9
        + (p60 & (conceded == 0)) * (pos <= 2)[None, :] * 12
        + saves * 2
        + actions * 0.5
        - yellows * 3
        - red * 9
        + np.where(appeared, rng.normal(0, 4, (worlds, n)), 0)
    )
    bps[~appeared] = -1e6
    bonus = np.zeros((worlds, n), dtype=np.int16)
    for column in range(n):
        # Bonus is contested by both teams in the SAME match, with official ties.
        rank = (bps > bps[:, column, None]).sum(axis=1)
        bonus[:, column] = np.where(appeared[:, column], np.maximum(0, 3 - rank), 0)
    components["bonus"] = bonus
    points = sum(components.values()).astype(np.int16)
    audit = {
        "fixture_id": fixture["id"],
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "lineups": quality,
        "scored_goals": int(goals.sum()),
        "own_goals": int(own.sum()),
        "team_goals": int(sum(value.sum() for value in totals.values())),
    }
    return ids, points, minutes, starts, components, audit


def forecast(
    store: Store,
    start_event: int | None = None,
    end_event: int = 38,
    worlds: int = 2000,
    seed: int = 11,
    use_beliefs: bool = False,
) -> dict:
    started = time.monotonic()
    cutoff = now()
    players, market = load_market(store)
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    if start_event is None:
        start_event = next((e["id"] for e in market["events"] if e.get("is_next")), 38)
    if not 1 <= start_event <= end_event <= 38 or not 100 <= worlds <= 20000:
        raise ValueError("Invalid forecast horizon or scenario budget (100–20000)")
    histories_ = histories(store, players, cutoff=cutoff)
    from collections import defaultdict

    from .history import archived_rows

    archive = defaultdict(list)
    for row in archived_rows(store, cutoff=cutoff):
        if row.get("code"):
            archive[row["code"]].append(row)
    for identity, player in players.items():
        previous = sorted(archive[player.code], key=lambda r: (r["season"], r.get("kickoff_time") or ""))[
            -12:
        ]
        # The earliest current-season fixtures should not erase established
        # playing-time priors; stable official codes bridge seasonal IDs.
        histories_[identity] = previous + histories_[identity]
    profiles = {p.id: profile(p.position, histories_[p.id]) for p in players.values()}
    beliefs = []
    if use_beliefs:
        from .research import active_beliefs

        beliefs = active_beliefs(store, cutoff=cutoff)
        for belief in beliefs:
            if belief["player_id"] in profiles:
                profiles[belief["player_id"]][belief["parameter"]] = belief["value"]
    rates = team_rates(market["fixtures"], market["teams"])
    ids = sorted(players)
    index = {identity: i for i, identity in enumerate(ids)}
    rng = np.random.default_rng(seed)
    health = {
        team["id"]: rng.random((worlds, sum(p.team == team["id"] and p.selectable for p in players.values())))
        for team in market["teams"]
    }
    records, fixture_records, audits = [], [], []
    arrays: dict[str, Any] = {}
    for event in range(start_event, end_event + 1):
        points = np.zeros((worlds, len(ids)), dtype=np.int16)
        minutes = np.zeros_like(points)
        start_count = np.zeros_like(points)
        components: dict[str, Any] = {}
        fixtures = [f for f in market["fixtures"] if f.get("event") == event and not f.get("finished")]
        for fixture in fixtures:
            f_ids, p, m, starts, comp, audit = fixture_worlds(
                rng, fixture, players, profiles, rates, rules, worlds, event - start_event, health
            )
            columns = [index[i] for i in f_ids]
            points[:, columns] += p
            minutes[:, columns] += m
            start_count[:, columns] += (starts == 0) & (m > 0)
            audits.append(audit)
            for key, value in comp.items():
                components.setdefault(key, np.zeros(len(ids)))[columns] += value.mean(axis=0)
            for col, identity in enumerate(f_ids):
                fixture_records.append(
                    {
                        "player_id": identity,
                        "fixture_id": fixture["id"],
                        "event": event,
                        "mean": float(p[:, col].mean()),
                        "minutes": float(m[:, col].mean()),
                    }
                )
        for column, identity in enumerate(ids):
            p, m = points[:, column], minutes[:, column]
            records.append(
                {
                    "player_id": identity,
                    "event": event,
                    "mean": round(float(p.mean()), 4),
                    "sd": round(float(p.std()), 4),
                    "point_quantiles": np.quantile(p, [0.1, 0.5, 0.9]).tolist(),
                    "minute_quantiles": np.quantile(m, [0.1, 0.5, 0.9]).tolist(),
                    "minutes_mean": round(float(m.mean()), 2),
                    "p_zero": float((m == 0).mean()),
                    "p_start": float((start_count[:, column] > 0).mean()),
                    "p_60": float((m >= 60).mean()),
                    "p_haul": float((p >= 10).mean()),
                    "components": {k: round(float(v[column]), 4) for k, v in components.items()},
                    "history_count": profiles[identity]["history_count"],
                    "action_coverage": profiles[identity]["action_coverage"],
                    "fixture_count": sum(
                        players[identity].team in (f["team_h"], f["team_a"]) for f in fixtures
                    ),
                }
            )
        arrays[f"points_{event}"] = points
        arrays[f"minutes_{event}"] = minutes
    issued = now()
    with store.db() as db:
        sources = [
            r[0]
            for r in db.execute(
                "SELECT DISTINCT source FROM snapshots WHERE captured_at<=? AND "
                "(source LIKE 'history_%' OR source LIKE 'archive_%')",
                (cutoff,),
            )
        ]
    manifest = [
        {
            k: v
            for k, v in store.latest(source, cutoff=cutoff)[1].items()
            if k in ("source", "hash", "captured_at")
        }
        for source in sorted(sources)
    ]
    body = {
        "issued_at": issued,
        "cutoff": cutoff,
        "season": market["season"],
        "start_event": start_event,
        "end_event": end_event,
        "worlds": worlds,
        "seed": seed,
        "model_version": MODEL_VERSION,
        "rules_hash": rules.hash,
        "market_id": market["id"],
        "input_hashes": [market["source_hash"], market["fixture_hash"], digest(manifest)],
        "input_manifest": manifest,
        "players": records,
        "belief_scenario": [b["id"] for b in beliefs],
        "fixtures": fixture_records,
        "match_audit": audits,
        "player_ids": ids,
        "unassigned_fixtures": [f["id"] for f in market["fixtures"] if f.get("event") is None],
        "assumptions": [
            "Exposure-shrunk current-season component baseline, not a validated elite forecast",
            "Shared match tempo, finite XI, timed goals and on-pitch attribution",
            "BPS proxy and Poisson defensive actions require calibration",
            "Injury recovery is a horizon sensitivity prior; future prices held constant",
            "Rare penalty saves/misses are not separately parameterized",
        ],
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    artifact_hash = digest(body)
    path = store.root / "scenarios" / f"{artifact_hash}.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **arrays)
    body["scenario_path"] = str(path)
    body["id"] = store.put("forecast", body)
    return body


def lineup_worlds(lineup, points, minutes, ids, players, rules, chip=None):
    index = {p: i for i, p in enumerate(ids)}
    relevant = lineup.xi + lineup.bench
    patterns, inverse = np.unique(minutes[:, [index[p] for p in relevant]] > 0, axis=0, return_inverse=True)
    result = np.zeros(len(points))
    for group, pattern in enumerate(patterns):
        selected = scoring_ids(lineup, dict(zip(relevant, pattern)), players, rules, chip == "bboost")
        mask = inverse == group
        result[mask] = points[mask][:, [index[p] for p in selected]].sum(axis=1)
    captain, vice = index[lineup.captain], index[lineup.vice]
    extra = np.where(
        minutes[:, captain] > 0, points[:, captain], np.where(minutes[:, vice] > 0, points[:, vice], 0)
    )
    return result + extra * (2 if chip == "3xc" else 1)


def score_plan(store: Store, plan: dict, forecast_: dict) -> np.ndarray:
    players, market = load_market(store, forecast_["market_id"])
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    from .domain import Lineup

    worlds = np.zeros(forecast_["worlds"])
    with np.load(store.artifact_path(forecast_["scenario_path"]), allow_pickle=False) as samples:
        for row in plan["ledger"]:
            event = row["event"]
            scores = lineup_worlds(
                Lineup.model_validate(row["lineup"]),
                samples[f"points_{event}"],
                samples[f"minutes_{event}"],
                forecast_["player_ids"],
                players,
                rules,
                row["chip"],
            )
            worlds += scores - row["hit"]
            row["simulated_net_mean"] = round(float(scores.mean() - row["hit"]), 3)
            row["net_quantiles"] = np.quantile(scores - row["hit"], [0.1, 0.5, 0.9]).tolist()
    plan["simulation"] = {
        "worlds": len(worlds),
        "seed": forecast_["seed"],
        "mean": float(worlds.mean()),
        "sd": float(worlds.std()),
        "quantiles": np.quantile(worlds, [0.1, 0.5, 0.9]).tolist(),
        "method": "common match worlds with autosubs and captain fallback",
    }
    return worlds
