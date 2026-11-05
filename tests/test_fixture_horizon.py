import numpy as np
import polars as pl

from fpl_intelligence.decision import NAMES
from fpl_intelligence.fixture_horizon import FEATURES, feature_vector, partitions, player_bootstrap, predict


def test_published_weekly_forecast_is_not_cloned_into_future_fixtures(market):
    player = market[1].model_copy(update={"raw": {"ep_next": "8.0", "minutes": "90"}})
    fixtures = [
        {"id": i, "team_h": 1, "team_a": 2, "event": e, "kickoff_time": f"2024-10-{d:02d}T15:00:00Z"}
        for i, e, d in ((1, 9, 25), (2, 9, 28), (3, 10, 31))
    ]
    teams = {1: {}, 2: {}}
    row = {"x": [0.0] * len(NAMES), "target": 99999}
    first = feature_vector(player, row, fixtures[0], fixtures, teams, 9, "2024-10-24T11:00:00Z")
    future = feature_vector(player, row, fixtures[2], fixtures, teams, 9, "2024-10-24T11:00:00Z")
    assert first[FEATURES.index("ep_next")] == 4 and first[FEATURES.index("ep_available")] == 1
    assert np.isnan(future[FEATURES.index("ep_next")]) and future[FEATURES.index("ep_available")] == 0
    assert np.isnan(first[FEATURES.index("expected_goals")])
    assert first[FEATURES.index("observed_expected_goals")] == 0
    assert not {"target", "base_target", "minutes_target", "start_target", "label_known"} & set(FEATURES)


def test_future_label_windows_and_late_matches_are_purged_before_calibration():
    rows = [
        {
            "source_group": f"2023/24-GW{e:02d}",
            "source_deadline": float(e),
            "label_end": f"2023/24-GW{e:02d}",
            "label_known": e + 0.1,
            "id": e,
        }
        for e in range(1, 13)
    ]
    rows.extend(
        [
            {
                "source_group": "2023/24-GW01",
                "source_deadline": 1.0,
                "label_end": "2023/24-GW01",
                "label_known": 8.5,
                "id": 999,
            },
            {
                "source_group": "2023/24-GW12",
                "source_deadline": 12.0,
                "label_end": "2023/24-GW12",
                "label_known": 21.0,
                "id": 998,
            },
            {
                "source_group": "2024/25-GW01",
                "source_deadline": 20.0,
                "label_end": "2024/25-GW01",
                "label_known": 20.1,
                "id": 997,
            },
        ]
    )
    train, cal, cut = partitions(pl.DataFrame(rows), "2024/25-GW01")
    assert cut == "2023/24-GW05"
    assert set(train["id"]) == {1, 2, 3, 4} and set(cal["id"]) == set(range(5, 13))
    assert train["label_known"].max() < 5 and cal["label_known"].max() < 20
    import pytest

    with pytest.raises(ValueError, match="explicit information cutoff"):
        partitions(pl.DataFrame(rows), "2024/25-GW02")
    live_train, live_cal, live_cut = partitions(pl.DataFrame(rows), "2024/25-GW02", 20.0)
    assert live_train.equals(train) and live_cal.equals(cal) and live_cut == cut


def test_assistant_manager_is_not_an_ordinary_player_and_original_is_immutable():
    original = {
        "elements": [{"id": 1, "element_type": 1}, {"id": 2, "element_type": 5}],
        "element_types": [{"id": i} for i in range(1, 6)],
    }
    ordinary = player_bootstrap(original)
    assert ordinary["elements"] == [original["elements"][0]] and len(original["elements"]) == 2
    assert [p["id"] for p in ordinary["element_types"]] == [1, 2, 3, 4]
    assert ordinary["chips"] == []


def test_calibration_cannot_destroy_probability_mass_or_revive_known_absence():
    class Minute:
        def predict_proba(self, matrix):
            return np.tile([0.2, 0.1, 0.2, 0.5], (len(matrix), 1))

    class Point:
        def predict(self, matrix):
            return np.full(len(matrix), 4.0)

    rows = [{name: 0.0 for name in FEATURES} for _ in range(2)]
    for row in rows:
        row.update(selectable=1.0, chance_of_playing_next_round=100.0)
    rows[1]["chance_of_playing_next_round"] = 0.0
    frame = pl.DataFrame(rows).with_columns(pl.Series("dc_prior", [[0.0] * 4] * 2))
    curves = {str(k): {"x": [0.0, 1.0], "y": [0.0, 0.0]} for k in range(4)}
    mean, p, _ = predict({"minutes": Minute(), "points": Point(), "curves": curves}, frame)
    assert np.allclose(p.sum(1), 1) and np.isfinite(mean).all()
    assert np.isclose(mean[0], 3.2) and mean[1] == 0 and p[1, 0] == 1


def test_future_expert_selection_excludes_later_targets_from_early_forecasts():
    from fpl_intelligence.horizon_refinement import future_mask

    frame = pl.DataFrame(
        {
            "season": ["2024/25"] * 4,
            "event": [1, 1, 18, 20],
            "lead_weeks": [0, 18, 3, 1],
            "label_end": ["2024/25-GW01", "2024/25-GW19", "2024/25-GW21", "2024/25-GW21"],
        }
    )
    assert future_mask(frame, "2024/25", 1, 19).tolist() == [False, True, False, False]
