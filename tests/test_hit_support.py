import numpy as np
import pytest

from fpl_intelligence.objectives import choose_recommendation, paid_transfer_evidence
from fpl_intelligence.storage import Store


def policy(identity, mean, hit=0, bound=610, objective=600):
    return {
        "id": identity,
        "state_id": "same-state",
        "forecast_id": "same-forecast",
        "end_event": 11,
        "first_week_hit_budget": None if hit else 0,
        "ledger": [
            {"event": 10, "hit": hit, "chip": None, "buy": [], "simulated_net_mean": 50 - hit},
            {"event": 11, "hit": 0, "chip": None, "buy": [], "simulated_net_mean": mean - (50 - hit)},
        ],
        "solver": {"objective": objective, "best_bound": bound},
        "simulation": {"mean": mean},
        "validation": {"legal": True},
    }


def test_precise_monte_carlo_cannot_establish_a_paid_hit_with_unfinished_search():
    paid, free = policy("paid", 601.73, 12), policy("free", 600)
    e = paid_transfer_evidence(paid, free, np.full(20000, 1.73), free)
    assert e["hit_supported_beyond_mc_error"]
    assert not e["hit_supported_beyond_search_error"]
    assert e["no_hit_search_allowance"] == 10
    assert e["weekly_tradeoff"][0]["cumulative_net_advantage"] == -12
    assert e["weekly_tradeoff"][-1]["cumulative_net_advantage"] == pytest.approx(1.73)


def test_material_hit_can_pass_and_unknown_or_conditional_bounds_cannot():
    paid, free = policy("paid", 615, 4), policy("free", 600, bound=600.2)
    assert paid_transfer_evidence(paid, free, np.full(1000, 15), free)["hit_supported_beyond_search_error"]
    for solver in [
        {"objective": 600, "best_bound": None},
        {"objective": 600, "best_bound": 599},
        {"objective": 600, "best_bound": 600, "continuation_plan_id": "fixed-first-action"},
    ]:
        search = {**free, "solver": solver}
        assert not paid_transfer_evidence(paid, free, np.full(1000, 15), search)[
            "hit_supported_beyond_search_error"
        ]
    with pytest.raises(ValueError, match="frozen inputs"):
        paid_transfer_evidence(paid, {**free, "forecast_id": "different"}, np.ones(100), free)


def test_recommendation_retains_free_policy_despite_precise_small_paid_advantage(tmp_path, monkeypatch):
    store = Store(tmp_path)
    paid, free, hold = policy("paid", 601.73, 12), policy("free", 600), policy("hold", 590)
    for plan in (paid, free, hold):
        plan["id"] = store.put("plan", {k: v for k, v in plan.items() if k != "id"})
    monkeypatch.setattr("fpl_intelligence.optimization.optimize", lambda *a, **kw: free)
    monkeypatch.setattr("fpl_intelligence.optimization.hold_plan", lambda *a: hold)
    monkeypatch.setattr(
        "fpl_intelligence.simulation.score_plans",
        lambda s, plans, f: [np.full(100, p["simulation"]["mean"]) for p in plans],
    )
    selected, report = choose_recommendation(store, [paid], {"id": "same-forecast"})
    assert selected["id"] == free["id"]
    assert report["selection"]["actual_ev_sacrifice"] == pytest.approx(1.73)
    assert report["immediate_hit_evidence"]["no_hit_search_allowance"] == 10


def test_stale_paid_receipts_cannot_survive_an_upgrade_or_invalidate_other_objectives():
    from fpl_intelligence.objectives import SELECTION_VERSION, stale_paid_evidence

    p = policy("paid", 615, 4)
    old = {
        "selection": {"objective": "expected_points", "selected_plan_id": "paid"},
        "immediate_hit_evidence": None,
    }
    assert stale_paid_evidence(p, old)
    current = {
        **old,
        "selection_version": SELECTION_VERSION,
        "immediate_hit_evidence": {"hit_supported_beyond_search_error": True},
    }
    assert not stale_paid_evidence(p, current)
    assert not stale_paid_evidence(p, {"selection": {"objective": "cvar", "selected_plan_id": "paid"}})
    assert not stale_paid_evidence(
        p, {"selection": {"objective": "expected_points", "selected_plan_id": "different"}}
    )


def test_forecast_review_distinguishes_fitness_from_starts_and_keeps_missing_ep(
    tmp_path, market, monkeypatch
):
    from fpl_intelligence.objectives import source_forecast_review

    store = Store(tmp_path)
    store.put("market", {})
    market[3] = market[3].model_copy(update={"chance": 1.0, "raw": {"ep_next": "5.5", "starts": 8}})
    market[18] = market[18].model_copy(update={"raw": {"ep_next": None, "starts": 9}})
    frozen = {
        "teams": [{"id": p.team, "short_name": str(p.team)} for p in market.values()],
        "fixtures": [{"event": 10, "team_h": market[3].team, "team_a": market[18].team, "finished": False}],
    }
    monkeypatch.setattr("fpl_intelligence.ingestion.load_market", lambda *a: (market, frozen))
    p = policy("paid", 600, 4)
    p["ledger"][0].update(sell=[{"player_id": 3}], buy=[{"player_id": 18}])
    f = {
        "start_event": 10,
        "players": [
            {"player_id": pid, "event": 10, "mean": 3.1, "minutes_mean": 59, "p_start": 0.74}
            for pid in (3, 18)
        ],
    }
    rows = source_forecast_review(store, p, f)
    assert rows[0]["official_availability"] == 1 and rows[0]["model_start_probability"] == 0.74
    assert rows[0]["official_next_points"] == 5.5 and rows[1]["official_next_points"] is None


@pytest.mark.parametrize("gain,chip_used", [(1.2, False), (20, True)])
def test_current_chip_requires_support_beyond_ordinary_transfer_search(
    tmp_path, monkeypatch, rules, gain, chip_used
):
    from fpl_intelligence.rules import Rules

    store = Store(tmp_path)
    sid = store.put("state", {"active_chip": None})
    chip, ordinary = policy("chip", 600 + gain), policy("ordinary", 600)
    chip["ledger"][0]["chip"] = "freehit"
    for plan in (chip, ordinary):
        plan["state_id"] = sid
        plan["id"] = store.put("plan", {k: v for k, v in plan.items() if k != "id"})
    monkeypatch.setattr(store, "latest", lambda *a: ({}, {}))
    monkeypatch.setattr(Rules, "from_bootstrap", classmethod(lambda *a: rules))

    def control(*a, **kw):
        assert set(kw["blocked_chips"][10]) == {"wildcard", "freehit", "bboost", "3xc"}
        assert kw["first_week_hit_budget"] == 0
        return ordinary

    monkeypatch.setattr("fpl_intelligence.optimization.optimize", control)
    monkeypatch.setattr("fpl_intelligence.optimization.hold_plan", lambda *a: ordinary)
    monkeypatch.setattr(
        "fpl_intelligence.simulation.score_plans",
        lambda s, ps, f: [np.full(100, p["simulation"]["mean"]) for p in ps],
    )
    selected, receipt = choose_recommendation(store, [chip], {"id": "same-forecast"})
    assert bool(selected["ledger"][0]["chip"]) == chip_used
    assert receipt["immediate_chip_evidence"]["supported_beyond_search_error"] == chip_used


def test_committed_pinned_and_expiring_chips_keep_their_entitlement_rules(rules):
    from fpl_intelligence.objectives import chip_has_future_value

    assert chip_has_future_value({}, rules, 10, "freehit")
    assert not chip_has_future_value({"active_chip": "freehit"}, rules, 10, "freehit")
    assert not chip_has_future_value({}, rules, 10, "freehit", {10: "freehit"})
    assert not chip_has_future_value({}, rules, 19, "freehit")


def test_chip_comparison_includes_ordinary_actions_at_the_same_hit_budget():
    from fpl_intelligence.objectives import chip_use_evidence

    boosted, ordinary = policy("boosted", 601, hit=4), policy("ordinary", 600, hit=4)
    boosted["ledger"][0]["chip"] = "bboost"
    ordinary["first_week_hit_budget"] = 4
    evidence = chip_use_evidence(boosted, ordinary, np.ones(100), ordinary)
    assert not evidence["supported_beyond_search_error"]
    stronger = policy("stronger", 615, hit=4)
    stronger["ledger"][0]["chip"] = "bboost"
    assert chip_use_evidence(stronger, ordinary, np.full(100, 15), ordinary)["supported_beyond_search_error"]
    with pytest.raises(ValueError, match="charges a new hit"):
        paid_transfer_evidence(boosted, ordinary, np.ones(100), ordinary)
