import asyncio

import pytest

from fpl_intelligence.replay import capture_replay_inputs, extend_replay
from fpl_intelligence.storage import Store


def test_replay_extension_retains_frozen_earlier_decisions(tmp_path, monkeypatch):
    import fpl_intelligence.replay as module

    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "data")
    feature = store.put("decision_features", {"hash": "frozen"})
    store.put("market", {"events": [{"id": n, "data_checked": True} for n in (1, 2)]})
    previous_id = store.put(
        "weekly_replay",
        {
            "feature_id": feature,
            "worlds": 20000,
            "weeks": [{"event": 1, "actual_points": 45, "decision_id": "old"}],
            "total_points": 45,
        },
    )

    def run(store, events, worlds, seconds, publish):
        assert events == [2] and worlds == 20000 and not publish
        assert store.get("weekly_replay")["id"] == previous_id
        return {
            "id": "segment",
            "feature_id": feature,
            "worlds": worlds,
            "weeks": [{"event": 2, "actual_points": 55, "decision_id": "new"}],
        }

    monkeypatch.setattr(module, "run_replay", run)
    result = extend_replay(store, 2)
    assert result["total_points"] == 100
    assert [w["decision_id"] for w in result["weeks"]] == ["old", "new"]
    assert result["parent_replay_id"] == previous_id
    assert extend_replay(store, 2)["id"] == result["id"]
    store.put("decision_features", {"hash": "changed"})
    with pytest.raises(ValueError, match="lineage changed"):
        extend_replay(store, 3)


def test_local_deadline_capture_rejects_future_snapshots(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    deadline = "2026-10-31T11:00:00Z"
    when = {"stamp": "2026-10-30T10:00:00+00:00"}

    def latest(source, **kwargs):
        if source.startswith("context_live"):
            return {"deadline": deadline}, {}
        body = {"events": [{"id": 9, "deadline_time": deadline}]} if source == "bootstrap" else []
        return body, {"hash": source, "captured_at": when["stamp"]}

    monkeypatch.setattr(store, "latest", latest)
    captured = asyncio.run(capture_replay_inputs(store, "2026/27", [9]))[0]
    assert captured["bootstrap_hash"] == "bootstrap"
    assert captured["fixture_proof"]["source_captured_at"] == when["stamp"]
    when["stamp"] = "2026-10-31T11:00:00+00:00"
    with pytest.raises(ValueError, match="do not precede"):
        asyncio.run(capture_replay_inputs(store, "2026/27", [9]))


def test_live_refresh_preserves_the_registry_for_next_weeks_training(tmp_path, monkeypatch):
    import fpl_intelligence.replay as module
    import fpl_intelligence.storage as storage

    store = Store(tmp_path / "data")
    when = "2026-11-05T10:00:00+00:00"
    monkeypatch.setattr(module, "now", lambda: when)
    monkeypatch.setattr(storage, "now", lambda: when)
    bootstrap = {
        "events": [{"id": 10, "is_next": True, "deadline_time": "2026-11-06T18:30:00Z"}],
        "elements": [{"id": 1, "now_cost": 61}],
    }
    fixtures = [{"id": 90, "event": 10, "team_h": 1, "team_a": 2}]
    store.snapshot("bootstrap", "test://bootstrap", bootstrap)
    store.snapshot("fixtures", "test://fixtures", fixtures)
    result = module.capture_observed_inputs(store, bootstrap, fixtures)
    assert result["event"] == 10 and result["bootstrap"]["elements"][0]["now_cost"] == 61
    assert result["fixture_proof"]["source_captured_at"] == when
    when = "2026-11-06T18:30:00+00:00"
    assert module.capture_observed_inputs(store, bootstrap, fixtures) is None
    assert store.get("replay_inputs")["id"] == result["id"]
