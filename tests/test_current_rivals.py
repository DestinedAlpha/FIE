import pytest

from fpl_intelligence.rivals import current_comparison
from fpl_intelligence.storage import Store


def test_current_rival_comparison_accepts_later_gameweeks_and_uses_frozen_inputs(tmp_path, monkeypatch):
    store = Store(tmp_path)
    state = store.put("state", {"event": 11, "points": 500, "entry_id": 123})
    forecast = store.put("forecast", {"start_event": 11})
    league = store.put("league", {"members": []})
    plan = store.put("plan", {"end_event": 19})
    store.put("selected_plan", {"plan_id": plan})
    monkeypatch.setattr("fpl_intelligence.readiness.selection_is_current", lambda *_: True)
    monkeypatch.setattr(
        "fpl_intelligence.rivals.hold_plan", lambda s, st, f, end: {"id": "hold", "end_event": end}
    )
    seen = []

    def compare(s, plans, fid, lid, points, entry):
        seen.append((plans, fid, lid, points, entry))
        return {"id": "comparison"}

    monkeypatch.setattr("fpl_intelligence.rivals.compare", compare)
    assert current_comparison(store)["id"] == "comparison"
    assert seen[0][1:] == (forecast, league, 500, 123)
    assert [p["label"] for p in seen[0][0]] == ["Current recommendation", "Hold current squad"]
    assert store.get("state")["id"] == state
    monkeypatch.setattr("fpl_intelligence.readiness.selection_is_current", lambda *_: False)
    with pytest.raises(ValueError, match="Refresh the current recommendation"):
        current_comparison(store)


def test_companion_migration_requires_refresh_even_when_core_model_is_unchanged(tmp_path):
    from fpl_intelligence.readiness import forecast_dependencies_current

    store = Store(tmp_path)
    model = store.put("decision_model", {})
    previous = store.put("manager_policy", {"player_features": ["retired"]})
    forecast = {"decision_model_id": model, "manager_policy_id": previous}
    assert forecast_dependencies_current(store, forecast)
    store.put("manager_policy", {"player_features": ["supported"]})
    assert not forecast_dependencies_current(store, forecast)
    assert store.get("decision_model")["id"] == model
