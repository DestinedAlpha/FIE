from fpl_intelligence.optimization import relaxed_horizon_bound


def test_transfer_free_bound_matches_attainable_two_chip_score_for_fixed_squad(market, rules, state):
    # One legal squad makes the best relaxed policy independently countable:
    # GW10: 20 XI points + 10 captain + 10 TC; GW11: 11 + 1 captain + 4 BB.
    players = {p: market[p] for p in range(1, 16)}
    state = state.model_copy(update={"event": 10})
    means = {(event, p): 10.0 if event == 10 and p == 8 else 1.0 for event in (10, 11) for p in players}
    continuation = {"ft": 0.0, "bank": 0.0, "bank_max": 0, "chip_use": {}}
    allowed = {(event, chip): True for event in (10, 11) for chip in ("bboost", "3xc")}
    bound = relaxed_horizon_bound(players, rules, means, state, [10, 11], {}, continuation, allowed)
    assert bound is not None and abs(bound - 56) < 1e-6


def test_bounds_count_separate_second_half_entitlements(market, rules, state):
    players = {p: market[p] for p in range(1, 16)}
    state = state.model_copy(update={"event": 19})
    means = {(event, p): 1.0 for event in (19, 20) for p in players}
    continuation = {"ft": 0.0, "bank": 0.0, "bank_max": 0, "chip_use": {}}
    allowed = {(event, "bboost"): True for event in (19, 20)}
    assert relaxed_horizon_bound(players, rules, means, state, [19, 20], {}, continuation, allowed) == 32
