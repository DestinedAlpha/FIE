import numpy as np
import pytest

from fpl_intelligence.replay import current_weekly_results, weekly_worlds
from fpl_intelligence.storage import Store


def test_unfinished_gameweeks_cannot_be_reported_as_zero_point_backtests(tmp_path):
    from fpl_intelligence.replay import run_replay

    store = Store(tmp_path)
    store.put(
        "market",
        {"season": "2026/27", "events": [{"id": 1, "data_checked": True}, {"id": 2, "data_checked": False}]},
    )
    with pytest.raises(ValueError, match="finalized target"):
        run_replay(store, events=[2])


def test_current_scorecard_cannot_silently_quote_the_legacy_reference(tmp_path):
    store = Store(tmp_path)
    old = store.put("weekly_replay", {"total_points": 551})
    store.put("decision_model", {"role_minutes_model_id": "accepted-role"})
    with pytest.raises(ValueError):
        current_weekly_results(store)
    store.put("role_weekly_replay", {"participation_configuration_id": "different-role", "total_points": 600})
    with pytest.raises(ValueError, match="current participation"):
        current_weekly_results(store)
    store.put("decision_model", {"fusion_model_id": "different-accepted-family"})
    with pytest.raises(ValueError, match="separate policy"):
        current_weekly_results(store)
    store.put("decision_model", {"role_minutes_model_id": "accepted-role", "restored": True})
    latest = store.put(
        "role_weekly_replay", {"participation_configuration_id": "accepted-role", "total_points": 560}
    )
    assert current_weekly_results(store)["id"] == latest
    assert current_weekly_results(store, baseline=True)["id"] == old


def test_historical_scorecard_matches_configuration_after_a_final_weight_refit(tmp_path):
    from fpl_intelligence.replay import replay_configuration_hash

    store = Store(tmp_path)
    config = {"version": "role-minutes-v2", "weight": 1, "workload_transform": "positive-exposure-v2"}
    model = store.put("role_minutes_model", config | {"fit_as_of": "later-completed-data"})
    store.put("decision_model", {"role_minutes_model_id": model})
    scorecard = store.put(
        "role_weekly_replay",
        {
            "configuration_hash": replay_configuration_hash(config),
            "participation_configuration_id": "earlier-final-weights",
            "total_points": 569,
        },
    )
    assert current_weekly_results(store)["id"] == scorecard
    changed = store.put("role_minutes_model", config | {"weight": 0.5})
    store.put("decision_model", {"role_minutes_model_id": changed})
    with pytest.raises(ValueError, match="current participation"):
        current_weekly_results(store)


def test_replay_role_probabilities_do_not_apply_availability_twice(monkeypatch, market):
    captured = []
    monkeypatch.setattr(
        "fpl_intelligence.replay.predict_models",
        lambda *a: (None, np.array([[0.2, 0.2, 0.3, 0.3]]), np.zeros((1, 4))),
    )

    def sample(rng, p, conditional, dc, pools, position, uniform, residual):
        captured.append(p.copy())
        return np.zeros(4, dtype=np.int16), np.zeros(4, dtype=int), np.zeros(4)

    monkeypatch.setattr("fpl_intelligence.replay.sample_point_outcomes", sample)
    row = {
        "player_id": 1,
        "fixture_id": 9,
        "team": 1,
        "x": [0] * 26,
        "dc_probability": [0] * 4,
        "availability": 0.5,
    }
    role = np.array([[0.1, 0.2, 0.3, 0.4]])
    weekly_worlds(
        [row],
        {},
        {},
        {"defence": 0.1},
        {str(k): [90] * 101 for k in (1, 2, 3)},
        {1: market[1]},
        4,
        11,
        role,
        1,
    )
    assert np.allclose(captured[-1], role[0])
    weekly_worlds(
        [row | {"availability": 0}],
        {},
        {},
        {"defence": 0.1},
        {str(k): [90] * 101 for k in (1, 2, 3)},
        {1: market[1]},
        4,
        11,
        role,
        1,
    )
    assert captured[-1].tolist() == [1, 0, 0, 0]
