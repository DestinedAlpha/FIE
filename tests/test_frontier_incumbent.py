from fpl_intelligence.decision_runtime import refine_frontier
from fpl_intelligence.storage import Store


def test_frontier_keeps_a_stronger_existing_policy(tmp_path, state, monkeypatch):
    store = Store(tmp_path / "data")
    state_id = store.put("state", state.model_dump())
    forecast_id = store.put("forecast", {"version": "fixture"})
    base = {"state_id": state_id, "forecast_id": forecast_id, "end_event": 19, "quality": 100}
    plan_id = store.put("plan", base)
    store.put("selected_plan", {"plan_id": plan_id})
    store.put("league", {})
    analysis = {
        "id": "analysis",
        "selected_candidate": 0,
        "promotion_gate_passed": False,
        "candidates": [{"action": {}, "reference_plus_continuation": 1, "tree": 1}],
    }
    monkeypatch.setattr("fpl_intelligence.decision_runtime.analyze_decisions", lambda *_: analysis)
    monkeypatch.setattr("fpl_intelligence.optimization.optimize", lambda *a, **kw: {**base, "quality": 90})
    monkeypatch.setattr("fpl_intelligence.optimization.hold_plan", lambda *a, **kw: {**base, "quality": 0})
    monkeypatch.setattr(
        "fpl_intelligence.decision_runtime.conditional_plan", lambda *a, **kw: {**base, "quality": 80}
    )

    def compare(saved, plans, *args):
        assert max(p["quality"] for p in plans) == 100
        return {"selected_plan_id": max(plans, key=lambda p: p["quality"])["id"]}

    monkeypatch.setattr("fpl_intelligence.rivals.compare", compare)
    result = refine_frontier(store)
    assert store.get("plan", result["selected_plan_id"])["quality"] == 100
