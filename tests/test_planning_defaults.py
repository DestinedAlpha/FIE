from fpl_intelligence.planning_defaults import chips_assessed, defaults
from fpl_intelligence.storage import Store


def install(tmp_path, monkeypatch, rules, state):
    store = Store(tmp_path)
    identity = store.put("state", state.model_dump())
    market = {"source_hash": "frozen"}
    market["id"] = store.put("market", market)
    bootstrap = {"game_settings": {}, "element_types": [], "events": [], "chips": []}
    monkeypatch.setattr(store, "latest", lambda *a, **kw: (bootstrap, {}))
    monkeypatch.setattr("fpl_intelligence.rules.Rules.from_bootstrap", lambda _: rules)
    return store, identity, market


def test_default_horizon_can_value_all_remaining_current_window_chips(tmp_path, monkeypatch, rules, state):
    store, _, market = install(tmp_path, monkeypatch, rules, state)
    result = defaults(store, state.model_dump(), market)
    assert result["end_event"] == 19 and result["search_seconds"] == 180
    used = [{"chip": chip, "event": 8} for chip in ("wildcard", "freehit", "bboost", "3xc")]
    result = defaults(store, state.model_dump() | {"used_chips": used}, market)
    assert result["end_event"] == state.event + 7


def test_short_horizon_preserves_unassessed_chips_but_respects_explicit_commitment(
    tmp_path, monkeypatch, rules, state
):
    store, identity, market = install(tmp_path, monkeypatch, rules, state)
    plan = {
        "state_id": identity,
        "market_id": market["id"],
        "end_event": 12,
        "ledger": [{"event": 10, "chip": "freehit"}],
    }
    assert not chips_assessed(store, plan)
    assert chips_assessed(store, plan, {10: "freehit"})
    assert chips_assessed(store, plan | {"end_event": 19})
    assert chips_assessed(store, plan | {"chip_constraints": {"pinned": {"10": "freehit"}}})
    current = store.put("state", state.model_dump() | {"active_chip": "freehit"})
    assert chips_assessed(
        store, plan | {"state_id": current, "ledger": [{"event": state.event, "chip": "freehit"}]}
    )
