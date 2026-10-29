from itertools import combinations

import numpy as np
import pytest

from fpl_intelligence.domain import Lineup
from fpl_intelligence.optimization import select_squad
from fpl_intelligence.rules import best_lineup, scoring_ids
from fpl_intelligence.simulation import fixture_worlds, lineup_worlds


def test_actual_squad_solver_against_enumeration(market, rules, state):
    tiny = {p: market[p] for p in range(1, 17)}
    means = {p: float((p * 11) % 17) for p in tiny}
    selected = select_squad(state, tiny, rules, means, seconds=5)

    def value(squad):
        lineup = best_lineup(squad, means, tiny, rules)
        return sum(means[p] for p in lineup.xi) + means[lineup.captain] + 0.015 * sum(means[p] for p in squad)

    oracle = max(
        value(list(squad))
        for squad in combinations(tiny, 15)
        if sum(tiny[p].position == 1 for p in squad) == 2
    )
    assert value(selected) == pytest.approx(oracle)


def test_bench_formation_priority(market, rules, state):
    squad = [lot.player_id for lot in state.lots]
    lineup = Lineup(xi=[1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14], bench=[2, 15, 6, 7], captain=13, vice=14)
    appeared = {p: True for p in squad}
    appeared[3] = False
    chosen = scoring_ids(lineup, appeared, market, rules)
    assert 6 in chosen and 15 not in chosen
    appeared[3] = True  # Any cameo blocks the reserve.
    assert scoring_ids(lineup, appeared, market, rules) == lineup.xi


def test_common_player_is_identical_in_two_managers_worlds(market, rules, state):
    ids = [lot.player_id for lot in state.lots]
    lineup = best_lineup(ids, {p: p % 7 for p in ids}, market, rules)
    rng = np.random.default_rng(1)
    points = rng.integers(-2, 16, size=(100, len(ids)))
    minutes = rng.choice([0, 1, 59, 90], size=points.shape)
    first = lineup_worlds(lineup, points, minutes, ids, market, rules)
    second = lineup_worlds(lineup, points, minutes, ids, market, rules)
    assert np.array_equal(first, second)


def test_coherent_fixture_goal_conservation_and_bonus(market, rules):
    # Build two clubs each with one keeper and ten outfield players. Player
    # categories do not impose an FPL 3-5-2 on the real football lineup.
    chosen = {}
    for team in (1, 2):
        for index in range(11):
            identity = 100 * team + index
            chosen[identity] = market[1 if index == 0 else 3].model_copy(
                update={"id": identity, "team": team}
            )
    from fpl_intelligence.features import profile

    profiles = {
        p: profile(player.position, [{"minutes": 90, "starts": 1}] * 8) for p, player in chosen.items()
    }
    rates = {t: {"attack": 1, "defence": 1, "mean": 1.4} for t in (1, 2)}
    rng = np.random.default_rng(41)
    health = {t: np.zeros((100, 11)) for t in (1, 2)}
    ids, points, minutes, starts, components, audit = fixture_worlds(
        rng, {"id": 1, "team_h": 1, "team_a": 2}, chosen, profiles, rates, rules, 100, 0, health
    )
    assert audit["scored_goals"] + audit["own_goals"] == audit["team_goals"]
    assert np.all((minutes >= 0) & (minutes <= 90))
    assert np.array_equal(points, sum(components.values()).astype(np.int16))
    assert np.all(components["bonus"].sum(axis=1) >= 6)


def test_player_absence_over_double_requires_zero_total_minutes(market, rules, state):
    ids = [lot.player_id for lot in state.lots]
    lineup = best_lineup(ids, {p: p % 7 for p in ids}, market, rules)
    points = np.ones((2, len(ids)), dtype=int) * 2
    minutes = np.ones_like(points) * 90
    cap = ids.index(lineup.captain)
    minutes[0, cap] = 0
    points[0, cap] = 0
    minutes[1, cap] = 1
    points[1, cap] = 1
    result = lineup_worlds(lineup, points, minutes, ids, market, rules, "3xc")
    # Zero minutes -> substitute + triple vice, cameo -> one captain point x3.
    assert result[0] == 26
    assert result[1] == 23
