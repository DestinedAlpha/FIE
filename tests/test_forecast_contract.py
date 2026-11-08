from types import SimpleNamespace

import numpy as np
import pytest

from fpl_intelligence import decision
from fpl_intelligence.forecast_validation import validate_forecast
from fpl_intelligence.storage import Store


def test_double_cameo_is_not_a_60_minute_fixture(tmp_path, market, rules, monkeypatch):
    store = Store(tmp_path)
    player = market[1]
    boot = store.snapshot("bootstrap", "https://example.test/", {})
    spec = {
        "promoted": True,
        "minute_quantiles": {str(k): [m, m] for k, m in enumerate([0, 30, 74, 90])},
        "dependence": {"defence": 0, "attack": 0},
        "residuals": {},
        "feature_id": "frozen",
        "seasons": [],
    }
    store.put("decision_model", spec)
    context = {
        "id": "market",
        "season": "2026/27",
        "source_hash": boot["hash"],
        "fixture_hash": "fixtures",
        "events": [{"id": 9, "is_next": True}],
        "fixtures": [],
    }
    rows = [
        {
            "player_id": player.id,
            "fixture_id": f,
            "event": 9,
            "team": player.team,
            "x": [0] * 26,
            "availability": 1,
            "minutes_cap": 90,
            "dc_probability": [0] * 4,
            "profile": {"history_count": 5, "action_coverage": 1},
        }
        for f in (1, 2)
    ]
    monkeypatch.setattr(decision, "load_market", lambda *_: ({player.id: player}, context))
    monkeypatch.setattr(decision, "future_inputs", lambda *_: (rows, []))
    monkeypatch.setattr(
        decision, "load_bundle", lambda *_: {"starter": SimpleNamespace(predict=lambda *_, **__: [0, 0])}
    )
    monkeypatch.setattr(decision, "curve_apply", lambda p, _: p)
    store.put("decision_model", {**spec, "curves": {"start": {}}})
    monkeypatch.setattr(
        decision, "predict_models", lambda *_: (None, np.array([[0, 1, 0, 0]] * 2), np.ones((2, 4)))
    )
    monkeypatch.setattr("fpl_intelligence.rules.Rules.from_bootstrap", lambda *_: rules)
    monkeypatch.setattr(
        decision,
        "sample_point_outcomes",
        lambda rng, probs, cond, dc, residuals, pos, u, ru: (
            np.ones(len(u), dtype=np.int16),
            np.ones(len(u), dtype=int),
            np.zeros(len(u)),
        ),
    )
    report = decision.learned_forecast(store, 9, 9, 100, 11)
    row = report["players"][0]
    assert row["minutes_mean"] == 60 and row["p_60"] == 0 and row["mean"] == 2
    assert row["p_60_semantics"] == "at_least_one_fixture_with_60_minutes"
    assert report["validation"]["passed"]
    with np.load(store.artifact_path(report["scenario_path"])) as arrays:
        report["players"][0]["mean"] += 1
        with pytest.raises(ValueError, match="summary differs"):
            validate_forecast(report, arrays)


def test_zero_minute_disciplinary_penalties_are_valid_worlds():
    forecast = {
        "player_ids": [1],
        "start_event": 1,
        "end_event": 1,
        "worlds": 2,
        "fixtures": [
            {"event": 1, "player_id": 1, "fixture_id": 1, "mean": -1, "minutes": 0, "p_zero": 1, "p_60": 0}
        ],
        "players": [
            {
                "event": 1,
                "player_id": 1,
                "mean": -1,
                "sd": 0,
                "minutes_mean": 0,
                "p_zero": 1,
                "p_start": 0,
                "p_60": 0,
                "p_haul": 0,
                "fixture_count": 1,
                "point_quantiles": [-1, -1, -1],
                "minute_quantiles": [0, 0, 0],
                "components": {"discipline": -1},
            }
        ],
    }
    arrays = {"points_1": np.full((2, 1), -1), "minutes_1": np.zeros((2, 1))}
    assert validate_forecast(forecast, arrays)["passed"]
    arrays["points_1"][:] = 1
    with pytest.raises(ValueError, match="participation"):
        validate_forecast(forecast, arrays)


@pytest.mark.parametrize("mutation", ["quantiles", "haul", "count", "fixture_probability", "fixture_total"])
def test_displayed_distribution_and_fixture_totals_match_optimizer_worlds(mutation):
    forecast = {
        "player_ids": [1],
        "start_event": 1,
        "end_event": 1,
        "worlds": 2,
        "fixtures": [
            {"event": 1, "player_id": 1, "fixture_id": 1, "mean": 5, "minutes": 90, "p_zero": 0, "p_60": 1}
        ],
        "players": [
            {
                "event": 1,
                "player_id": 1,
                "mean": 5,
                "sd": 5,
                "minutes_mean": 90,
                "p_zero": 0,
                "p_start": 1,
                "p_60": 1,
                "p_haul": 0.5,
                "fixture_count": 1,
                "point_quantiles": [1, 5, 9],
                "minute_quantiles": [90, 90, 90],
                "components": {"total": 5},
            }
        ],
    }
    arrays = {"points_1": np.array([[0], [10]]), "minutes_1": np.full((2, 1), 90)}
    assert validate_forecast(forecast, arrays)["passed"]
    if mutation == "quantiles":
        forecast["players"][0]["point_quantiles"] = [2, 5, 8]
    elif mutation == "haul":
        forecast["players"][0]["p_haul"] = 0.1
    elif mutation == "count":
        forecast["players"][0]["fixture_count"] = 2
    elif mutation == "fixture_probability":
        forecast["fixtures"][0]["p_zero"] = -1
    else:
        forecast["fixtures"][0]["mean"] = 6
    with pytest.raises(ValueError):
        validate_forecast(forecast, arrays)
