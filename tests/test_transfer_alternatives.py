from fpl_intelligence.domain import Action
from fpl_intelligence.search_control import planning_budget
from fpl_intelligence.storage import Store
from fpl_intelligence.transfer_alternatives import alternative_actions, compare_cheaper_actions


def test_cheaper_candidates_use_model_and_independent_ep_and_obey_account_rules(
    tmp_path, monkeypatch, market, rules, state
):
    from fpl_intelligence import transfer_alternatives as module

    store = Store(tmp_path)
    state_id = store.put("state", state.model_dump())
    for identity, player in market.items():
        market[identity] = player.model_copy(
            update={"selectable": identity <= 15 or identity in (18, 19, 20)}
        )
    market[18] = market[18].model_copy(update={"price": 60})
    market[19] = market[19].model_copy(update={"price": 50, "raw": {"ep_next": "4.0"}})
    market[20] = market[20].model_copy(update={"price": 49, "raw": {"ep_next": "9.0"}})
    monkeypatch.setattr(module, "load_market", lambda *_: (market, {"source_hash": "frozen"}))
    monkeypatch.setattr(store, "latest", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(module.Rules, "from_bootstrap", lambda _: rules)
    reference = {
        "state_id": state_id,
        "end_event": 11,
        "ledger": [{"buy": [{"player_id": 18}], "sell": [{"player_id": 3}], "hit": 0, "chip": None}],
    }
    forecast = {
        "market_id": "market",
        "players": [
            {"event": e, "player_id": p, "mean": 5 if p == 19 else 2} for e in (9, 10, 11) for p in market
        ],
    }
    choices = alternative_actions(store, reference, forecast)
    assert [a.buy for a, _ in choices] == [[19], [20]]
    assert all(a.sell == [3] for a, _ in choices)
    market[20] = market[20].model_copy(update={"selectable": False})
    assert [a.buy for a, _ in alternative_actions(store, reference, forecast)] == [[19]]
    compared = []

    def continuation(*args, **kwargs):
        compared.append(args[4].buy)
        return {"id": "legal-improving-action"}

    monkeypatch.setattr("fpl_intelligence.decision_runtime.conditional_plan", continuation)
    chip_incumbent = {**reference, "ledger": [{**reference["ledger"][0], "chip": "bboost"}]}
    forecast["id"] = "frozen-forecast"
    assert compare_cheaper_actions(store, chip_incumbent, forecast, max_transfers=1)
    assert [19] in compared


def test_comparison_preserves_later_pinned_chips_and_transfer_limits(monkeypatch):
    from fpl_intelligence import transfer_alternatives as module

    monkeypatch.setattr(
        module, "alternative_actions", lambda *_: [(Action(sell=[3], buy=[19]), "Independent candidate")]
    )
    calls = []

    def continuation(*args, **kwargs):
        calls.append(kwargs)
        return {"id": "scored-candidate"}

    monkeypatch.setattr("fpl_intelligence.decision_runtime.conditional_plan", continuation)
    reference = {"state_id": "state", "end_event": 15, "ledger": [{"event": 11}]}
    with planning_budget(30):
        result = compare_cheaper_actions(
            None, reference, {"id": "forecast"}, fixed_chips={14: "bboost"}, max_transfers=1
        )
    assert result == [{"id": "scored-candidate"}]
    assert calls[0]["continuation_options"] == {"chips": {14: "bboost"}, "max_transfers": 1}
    with planning_budget(0.01):
        assert compare_cheaper_actions(None, reference, {"id": "forecast"}) == []
    assert len(calls) == 1


def test_comparison_cannot_replace_an_explicit_current_chip(monkeypatch):
    from fpl_intelligence import transfer_alternatives as module

    def forbidden(*_):
        raise AssertionError("A pinned current chip must not generate ordinary alternatives")

    monkeypatch.setattr(module, "alternative_actions", forbidden)
    reference = {"ledger": [{"event": 11}]}
    assert compare_cheaper_actions(None, reference, {}, fixed_chips={11: "3xc"}) == []
