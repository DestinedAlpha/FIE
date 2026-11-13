import numpy as np
import pytest

from fpl_intelligence import future_expert
from fpl_intelligence.fixture_horizon import FEATURES
from fpl_intelligence.storage import Store


def inputs(tmp_path, market, monkeypatch, *, availability=0.5, base=0.5):
    store = Store(tmp_path / "data")
    store.put("replay_context", {"information_cutoff": "2026-10-24T11:00:00Z"})
    identity = store.put(
        "future_horizon_model",
        {
            "forecast_gate_passed": True,
            "policy_gate_passed": False,
            "weight": 0.5,
            "max_lead": 18,
            "curves": {"start": {"x": [0, 1], "y": [0, 1]}},
        },
    )

    class Starter:
        def predict(self, x, **kwargs):
            return np.full(len(x), 0.7)

    monkeypatch.setattr(
        future_expert,
        "load_expert",
        lambda *_: {"starter": Starter(), "curves": {"start": {"x": [0, 1], "y": [0, 1]}}},
    )
    monkeypatch.setattr(future_expert, "feature_vector", lambda *_: [0.0] * len(FEATURES))
    monkeypatch.setattr(
        future_expert,
        "predict",
        lambda _, frame: (
            np.full(len(frame), 4.0),
            np.tile([0.2, 0, 0, 0.8], (len(frame), 1)),
            np.tile([0.0, 0, 0, 5.0], (len(frame), 1)),
        ),
    )
    rows = [
        {
            "event": event,
            "player_id": 1,
            "fixture_id": event,
            "dc_probability": [0.0] * 4,
            "availability": availability,
            "base_availability": base,
        }
        for event in (9, 10, 28)
    ]
    p, c, s = np.tile([0.2, 0, 0, 0.8], (3, 1)), np.tile([0.0, 0, 0, 4.0], (3, 1)), np.full(3, 0.7)
    args = (
        store,
        {"future_horizon_model_id": identity},
        rows,
        [{"id": r["event"]} for r in rows],
        {"teams": []},
        market,
        p,
        c,
        s,
        9,
        "2026-10-24T11:00:00Z",
    )
    return args


def test_only_future_rows_change_and_news_availability_is_applied_once(tmp_path, market, monkeypatch):
    args = inputs(tmp_path, market, monkeypatch)
    before = [x.copy() for x in args[6:9]]
    p, c, s, included = future_expert.combine(*args)
    assert included.tolist() == [False, True, False]
    for actual, original in zip((p, c, s), before, strict=True):
        assert np.array_equal(actual[[0, 2]], original[[0, 2]])
    assert np.isclose(1 - p[1, 0], 0.6)
    assert np.isclose(s[1], 0.525)
    assert np.allclose(p.sum(1), 1)
    assert np.isclose((p[1] * c[1]).sum(), 0.5 * (0.4 * 4 + 0.8 * 5))
    assert all(np.array_equal(x, original) for x, original in zip(args[6:9], before, strict=True))


def test_explicit_absence_overrides_both_models(tmp_path, market, monkeypatch):
    args = inputs(tmp_path, market, monkeypatch, availability=0.0)
    p, _, starts, included = future_expert.combine(*args)
    assert np.array_equal(p[1], [1, 0, 0, 0]) and starts[1] == 0 and included[1]


def test_unaccepted_future_expert_cannot_be_enabled_on_live_account(tmp_path, market, monkeypatch):
    args = inputs(tmp_path, market, monkeypatch)
    with args[0].db() as db:
        db.execute("DELETE FROM objects WHERE kind='replay_context'")
    with pytest.raises(ValueError, match="full policy gate"):
        future_expert.combine(*args)


def test_explicit_recovery_does_not_apply_the_old_injury_prior_again(tmp_path, market, monkeypatch):
    args = inputs(tmp_path, market, monkeypatch, availability=1.0)
    p, _, starts, _ = future_expert.combine(*args)
    assert np.isclose(1 - p[1, 0], 0.8) and np.isclose(starts[1], 0.7)


def test_live_forecasting_cannot_reuse_unrepaired_scoring_weights(tmp_path, market, monkeypatch):
    args = list(inputs(tmp_path, market, monkeypatch))
    store = args[0]
    with store.db() as db:
        db.execute("DELETE FROM objects WHERE kind='replay_context'")
    proof = store.put("future_distribution_validation", {"passed": True})
    spec = store.get("future_horizon_model")
    body = {k: v for k, v in spec.items() if k != "id"}
    body.update(policy_gate_passed=True, distribution_validation_id=proof)
    args[1] = {"future_horizon_model_id": store.put("future_horizon_model", body)}
    with pytest.raises(ValueError, match="one-time model repair"):
        future_expert.combine(*args)


def test_refinement_replay_uses_the_selected_content_addressed_model(tmp_path):
    import json

    store = Store(tmp_path / "data")
    folder = store.root / "models" / "research" / "2025-26-GW01-uniform-selected-hash"
    folder.mkdir(parents=True)
    (folder / "model.json").write_text(
        json.dumps(
            {
                "hashes": {"minutes": "selected-weight-hash"},
                "curves": {},
                "fit_max_label": "2024/25-GW30",
                "cal_max_label": "2024/25-GW38",
                "cut": "2024/25-GW31",
                "training_rows": 100,
                "calibration_rows": 20,
                "data_contract": "repaired-target",
            }
        )
    )
    refinement = {
        "id": "selected-refinement",
        "forecast_gate_passed": True,
        "selected": "uniform_0.5",
        "model_paths": {"2025/26-GW01": str(folder)},
    }
    frozen = future_expert.frozen_spec(store, refinement, "2025/26-GW01")
    assert frozen["models"]["minutes"]["path"] == str(folder / "minutes.txt")
    assert frozen["data_contract"] == "repaired-target"
