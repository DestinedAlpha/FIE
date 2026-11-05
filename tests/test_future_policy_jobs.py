from fpl_intelligence.future_policy_jobs import run_future_suite
from fpl_intelligence.storage import Store


def test_changed_replay_budget_rebuilds_reference_without_reusing_old_worlds(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    accepted = {
        "season": "2025/26",
        "policy_gate_passed": True,
        "weeks": [{"event": n} for n in range(20, 39)],
        "candidate_points": 1202,
        "baseline_points": 1098,
    }
    acceptance_id = store.put("future_policy_replay", accepted)
    expert_id = store.put(
        "future_horizon_model", {"acceptance_id": acceptance_id, "refinement_id": "refinement"}
    )
    model_id = store.put("decision_model", {"future_horizon_model_id": expert_id})
    store.put("market", {"season": "2026/27", "events": [{"id": 1, "data_checked": True}]})
    store.put(
        "pipeline_replay",
        {
            "protocol_id": "old-protocol",
            "protocol": {"worlds": 4000, "seconds": 60},
            "weeks": [{"season": "2026/27", "event": 1}],
        },
    )
    store.put(
        "future_policy_diagnostic",
        {
            "season": "2026/27",
            "weeks": [{"event": 1}],
            "protocol": {"refinement_id": "refinement", "worlds": 2000, "seconds": 30},
            "candidate_points": 47,
            "baseline_points": 47,
        },
    )
    monkeypatch.setattr("fpl_intelligence.future_policy_jobs.verify_future_policy", lambda *_: {})

    async def prepare(*args):
        return []

    calls = []

    def reference(saved, **options):
        calls.append(options)
        return {"protocol": {"worlds": 2000, "seconds": 30}}

    monkeypatch.setattr("fpl_intelligence.consensus.prepare_inputs", prepare)
    monkeypatch.setattr("fpl_intelligence.pipeline_replay.run_pipeline_replay", reference)
    result = run_future_suite(store, worlds=2000, seconds=30)
    assert calls[0]["resume_protocol_id"] is None and calls[0]["reference_family_only"]
    assert result["decision_model_id"] == model_id and store.get("decision_model")["id"] == model_id
