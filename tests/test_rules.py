from itertools import combinations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fpl_intelligence.domain import Action
from fpl_intelligence.rules import (
    best_lineup,
    bonus_awards,
    quote,
    score_fixture,
    score_gameweek,
    selling_price,
    transition,
    validate_lineup,
    validate_squad,
)


@given(st.integers(35, 180), st.integers(35, 180))
def test_gain_sharing_rounding(purchase, market):
    sell = selling_price(purchase, market)
    assert sell <= market
    assert sell == market if market <= purchase else (market - sell in ((market - purchase + 1) // 2,))


def test_unknown_lot_needs_new_quote(state):
    lot = state.lots[0].model_copy(update={"purchase_price": None})
    assert quote(lot, 50) == 49
    with pytest.raises(ValueError, match="fresh sell quote"):
        quote(lot, 51)


def test_exact_imported_quote_has_current_priority(state):
    lot = state.lots[0].model_copy(update={"sell_price": 50})
    assert quote(lot, 50) == 50
    assert quote(lot, 52) == 50


def test_unknown_player_is_an_actionable_validation_error(market, rules, state):
    with pytest.raises(ValueError, match="unknown season-scoped"):
        transition(state, Action(sell=[1], buy=[999999]), market, rules)


def test_boundaries(rules):
    assert score_fixture({"minutes": 59}, 2, rules) == 1
    assert score_fixture({"minutes": 60}, 2, rules) == 6
    assert score_fixture({"minutes": 90, "goals_scored": 1}, 1, rules) == 16
    assert score_fixture({"minutes": 90, "goals_conceded": 3, "saves": 5}, 1, rules) == 2
    assert score_fixture({"minutes": 90, "defensive_actions": 9}, 2, rules) == 6
    assert score_fixture({"minutes": 90, "defensive_actions": 10}, 2, rules) == 8
    assert score_fixture({"minutes": 90, "defensive_actions": 25}, 2, rules) == 8
    assert score_fixture({"minutes": 90, "defensive_actions": 11}, 3, rules) == 3
    assert score_fixture({"minutes": 90, "defensive_actions": 12}, 3, rules) == 5
    assert score_fixture({"minutes": 0, "bonus": 3}, 2, rules) == 0


def test_bonus_ties():
    assert bonus_awards({1: 30, 2: 30, 3: 20, 4: 10}) == {1: 3, 2: 3, 3: 1, 4: 0}
    assert bonus_awards({1: 30, 2: 20, 3: 20, 4: 10}) == {1: 3, 2: 2, 3: 2, 4: 0}
    assert bonus_awards({1: 30, 2: 30, 3: 30, 4: 30}) == {1: 3, 2: 3, 3: 3, 4: 3}


@given(ft=st.integers(0, 5), count=st.integers(0, 5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_rollover(ft, count, market, rules, state):
    state = state.model_copy(update={"free_transfers": ft, "bank": 100})
    # Select legal same-position replacements from second squad.
    sells = list(range(1, count + 1))
    buys = [p + 15 for p in sells]
    result, ledger = transition(state, Action(sell=sells, buy=buys), market, rules)
    assert result.free_transfers == min(5, max(0, ft - count) + 1)
    assert ledger["hit"] == max(0, count - ft) * 4
    assert result.bank == 100 + 49 * count - 51 * count


def test_freehit_restore_and_no_consecutive(market, rules, state):
    next_state, row = transition(state, Action(sell=[1], buy=[16], chip="freehit"), market, rules)
    assert next_state.lots == state.lots
    assert next_state.bank == state.bank
    assert next_state.free_transfers == state.free_transfers
    assert 16 in row["playing_squad"] and 1 in row["permanent_squad"]
    with pytest.raises(ValueError, match="already used|Consecutive"):
        transition(next_state, Action(chip="freehit"), market, rules)
    # The restriction also crosses the first/second-half chip boundary.
    from fpl_intelligence.domain import ChipUse

    boundary = state.model_copy(update={"event": 20, "used_chips": [ChipUse(chip="freehit", event=19)]})
    with pytest.raises(ValueError, match="Consecutive"):
        transition(boundary, Action(chip="freehit"), market, rules)


def test_fh_after_prior_moves_restores_prior_deadline_and_refunds(market, rules, state):
    current = state.model_copy(update={"transfers_made": 3, "hit_paid": 4})
    with pytest.raises(ValueError, match="prior deadline"):
        transition(current, Action(chip="freehit"), market, rules)
    current = current.model_copy(update={"deadline_lots": state.lots, "deadline_bank": 31})
    following, ledger = transition(current, Action(chip="freehit"), market, rules)
    assert following.bank == 31 and ledger["hit"] == -4


def test_wc_preserves_ft_and_cancels_hits(market, rules, state):
    state = state.model_copy(update={"free_transfers": 5, "transfers_made": 6, "hit_paid": 4})
    following, ledger = transition(state, Action(chip="wildcard"), market, rules)
    assert following.free_transfers == 5 and ledger["hit"] == -4


def test_affordability_and_duplicate_transfers(market, rules, state):
    with pytest.raises(ValueError, match="unaffordable"):
        transition(state.model_copy(update={"bank": 0}), Action(sell=[1], buy=[16]), market, rules)
    with pytest.raises(ValueError, match="Duplicate"):
        Action(sell=[1, 1], buy=[16, 17])


def test_captain_vice_cameo_double_and_bench_boost(market, rules, state):
    squad = [lot.player_id for lot in state.lots]
    means = {p: float(20 - p) for p in squad}
    lineup = best_lineup(squad, means, market, rules)
    validate_lineup(lineup, squad, market, rules)
    points = {p: 2 for p in squad}
    minutes = {p: 90 for p in squad}
    points[lineup.captain] = 0
    minutes[lineup.captain] = 0
    assert score_gameweek(points, minutes, lineup, market, rules) == 24
    assert score_gameweek(points, minutes, lineup, market, rules, "3xc") == 26
    minutes[lineup.captain] = 1
    assert score_gameweek(points, minutes, lineup, market, rules) == 20
    # Minutes aggregated over both DGW matches prevent vice fallback.
    minutes[lineup.captain] = 90
    points[lineup.captain] = 2
    assert score_gameweek(points, minutes, lineup, market, rules, "bboost") == 32


def test_formation_oracle(market, rules, state):
    squad = [lot.player_id for lot in state.lots]
    means = {p: (p * 7) % 13 for p in squad}
    result = best_lineup(squad, means, market, rules)
    chosen = sum(means[p] for p in result.xi) + means[result.captain]
    oracle = -1
    for xi in combinations(squad, 11):
        from collections import Counter

        counts = Counter(market[p].position for p in xi)
        if all(rules.minimum[pos] <= counts[pos] <= rules.maximum[pos] for pos in rules.composition):
            oracle = max(oracle, sum(means[p] for p in xi) + max(means[p] for p in xi))
    assert chosen == oracle


def test_grandfathering_resolved_by_transfer(market, rules, state):
    market = {p: v.model_copy(update={"team": 1 if p <= 4 else v.team}) for p, v in market.items()}
    state = state.model_copy(update={"grandfathered_clubs": [1]})
    validate_squad(state, market, rules)
    with pytest.raises(ValueError, match="More than"):
        transition(state, Action(sell=[15], buy=[30]), market, rules)
