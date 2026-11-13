import pytest

from fpl_intelligence.future_training import (
    activate_expert,
    bind_immediate_audits,
    refit_expert,
    train_future_component,
)
from fpl_intelligence.storage import Store


def test_failed_future_policy_cannot_replace_the_live_model(tmp_path):
    store = Store(tmp_path / "data")
    original = store.put("decision_model", {"accepted_reference": True})
    store.put("market", {"season": "2026/27"})
    store.put("future_policy_replay", {"policy_gate_passed": False, "season": "2025/26"})
    with pytest.raises(ValueError, match="policy gate failed"):
        activate_expert(store)
    with pytest.raises(ValueError, match="accepted earlier-season"):
        refit_expert(store)
    assert store.get("decision_model")["id"] == original


def test_complete_training_keeps_reference_when_future_acceptance_fails(tmp_path):
    store = Store(tmp_path / "data")
    original = store.put("decision_model", {"accepted_reference": True})
    store.put("horizon_trial", {"completed": True})
    rejected = store.put("horizon_refinement", {"forecast_gate_passed": False})
    assert train_future_component(store)["id"] == rejected
    accepted_forecast = store.put("horizon_refinement", {"forecast_gate_passed": True})
    store.put("pipeline_replay", {"completed": True})
    rejected_policy = store.put(
        "future_policy_replay",
        {"policy_gate_passed": False, "protocol": {"refinement_id": accepted_forecast}},
    )
    assert train_future_component(store)["id"] == rejected_policy
    assert store.get("decision_model")["id"] == original


def test_partial_or_current_season_evidence_cannot_authorize_production_refits(tmp_path):
    store = Store(tmp_path / "data")
    store.put("market", {"season": "2026/27"})
    store.put("future_policy_replay", {"policy_gate_passed": True, "season": "2026/27"})
    with pytest.raises(ValueError, match="earlier-season"):
        refit_expert(store)
    store.put(
        "future_policy_replay",
        {
            "policy_gate_passed": True,
            "season": "2025/26",
            "weeks": [{}],
            "verification": {"sealed_deadlines": 1},
        },
    )
    with pytest.raises(ValueError, match="complete independently checked"):
        refit_expert(store)


def test_immediate_audits_reuse_executed_results_only_for_an_identical_head(tmp_path):
    store = Store(tmp_path / "data")
    original = {"version": "reference", "model_hashes": {"minutes": "pinned"}, "at": "fit-time"}
    old_id = store.put("decision_model", original)
    report_id = store.put(
        "decision_audit", {"decision_model_id": old_id, "at": "actual-test-time", "error": 1.2}
    )
    current = original | {"future_horizon_model_id": "accepted-expert"}
    current["id"] = store.put("decision_model", current)
    assert bind_immediate_audits(store, current)
    reused = store.get("decision_audit")
    assert (
        reused["reused_from"] == report_id and reused["at"] == "actual-test-time" and reused["error"] == 1.2
    )
    changed = current | {"model_hashes": {"minutes": "different"}}
    changed["id"] = store.put("decision_model", {k: v for k, v in changed.items() if k != "id"})
    assert not bind_immediate_audits(store, changed)


@pytest.mark.parametrize("cancel", [False, True])
def test_scoring_contract_repair_is_atomic_preserves_immediate_head_and_runs_once(
    tmp_path, monkeypatch, cancel
):
    from concurrent.futures import CancelledError

    from fpl_intelligence import future_training, model_refresh, search_control
    from fpl_intelligence.fixture_horizon import DATA_CONTRACT

    store = Store(tmp_path / "data")
    old_future = store.put("future_horizon_model", {"acceptance_id": "accepted-family"})
    body = {"promoted": True, "future_horizon_model_id": old_future, "model_hashes": {"points": "retained"}}
    original = store.put("decision_model", body)
    calls = []

    def fit(staged, acceptance):
        assert acceptance == "accepted-family"
        calls.append(acceptance)
        repaired = {"data_contract": DATA_CONTRACT}
        return repaired | {"id": staged.put("future_horizon_model", repaired)}

    def cancelled():
        if cancel and calls:
            raise CancelledError("Cancelled after fitting and before publication")

    monkeypatch.setattr(future_training, "refit_expert", fit)
    monkeypatch.setattr(
        model_refresh, "observation_signature", lambda _: {"signature": "same-finalized-inputs"}
    )
    monkeypatch.setattr(search_control, "check_cancelled", cancelled)
    if cancel:
        with pytest.raises(CancelledError):
            future_training.repair_data_contract(store)
        assert store.get("decision_model")["id"] == original
        assert store.get("future_horizon_model")["id"] == old_future
        assert not store.list("future_data_repair", 1)
    else:
        result = future_training.repair_data_contract(store)
        model = store.get("decision_model")
        assert result["previous_decision_model_id"] == original
        assert model["model_hashes"] == body["model_hashes"]
        assert model["future_horizon_model_id"] != old_future
        assert store.get("model_training_state")["decision_model_id"] == model["id"]
        assert future_training.repair_data_contract(store) is None
        assert calls == ["accepted-family"]
