import numpy as np
import polars as pl
import pytest

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


def test_unavailable_team_strength_is_missing_and_positive_ratings_are_preserved(market):
    player = market[1]
    fixture = {"id": 1, "team_h": 1, "team_a": 2, "event": 9}
    row = {"x": [0.0] * len(NAMES)}
    teams = {1: {"strength_attack_home": 0}, 2: {"strength_defence_away": 0}}
    values = feature_vector(player, row, fixture, [fixture], teams, 9, "2026-10-20T11:00:00Z")
    assert np.isnan(values[FEATURES.index("official_own_attack")])
    assert np.isnan(values[FEATURES.index("official_opponent_defence")])
    teams[1]["strength_attack_home"], teams[2]["strength_defence_away"] = 1250, 1170
    values = feature_vector(player, row, fixture, [fixture], teams, 9, "2026-10-20T11:00:00Z")
    assert values[FEATURES.index("official_own_attack")] == 1250
    assert values[FEATURES.index("official_opponent_defence")] == 1170


@pytest.mark.parametrize("target,base", [(8.0, 6.0), (6.0, 10.0)])
def test_horizon_dataset_uses_canonical_scoring_target_once(tmp_path, monkeypatch, market, target, base):
    from fpl_intelligence import fixture_horizon, observations
    from fpl_intelligence.availability import moment
    from fpl_intelligence.storage import Store

    store = Store(tmp_path / "data")
    path = store.root / "features" / "labels.parquet"
    path.parent.mkdir(exist_ok=True)
    label = {
        "season": "2026/27",
        "fixture_id": 1,
        "player_id": 1,
        "target": target,
        "base_target": base,
        "minutes_target": 90.0,
        "start_target": 1.0,
        "deadline_group": "2026/27-GW02",
    }
    pl.DataFrame([label]).write_parquet(path)
    store.put("decision_features", {"path": str(path), "hash": "canonical-labels"})
    fixture = {"id": 1, "event": 2, "team_h": 1, "team_a": 2, "kickoff_time": "2026-08-29T15:00:00Z"}
    packet = {
        "id": "frozen-packet",
        "deadline": "2026-08-29T11:00:00Z",
        "fixture_proof": {"fixtures": [fixture]},
        "bootstrap": {"teams": [{"id": 1}, {"id": 2}], "elements": [], "element_types": []},
    }
    original = {"season": "2026/27", "fixture": 1, "round": 2, "kickoff_time": fixture["kickoff_time"]}
    monkeypatch.setattr(fixture_horizon, "replay_packets", lambda _: {("2026/27", 2): packet})
    monkeypatch.setattr(fixture_horizon, "canonical_rows", lambda _: ([original], {}, {}))
    monkeypatch.setattr(observations, "finalization_clocks", lambda _: {})
    monkeypatch.setattr(observations, "label_available_at", lambda *_: moment("2026-08-30T09:00:00Z"))

    def deadline_rows(*_):
        return {1: market[1]}, [
            {
                "player_id": 1,
                "fixture_id": 1,
                "x": [0.0] * len(NAMES),
                "dc_probability": [0.0, 0.1, 0.5, 0.8],
                "availability": 1.0,
            }
        ]

    monkeypatch.setattr(fixture_horizon, "deadline_rows", deadline_rows)
    result = pl.read_parquet(fixture_horizon.build_data(store))
    assert result["target"][0] == target
    assert result["base_target"][0] == base
    assert result["dc_prior"].to_list() == [[0.0, 0.1, 0.5, 0.8]]


def test_exact_legacy_cache_repair_preserves_labels_and_rejects_different_packets(tmp_path, monkeypatch):
    import hashlib

    from fpl_intelligence import fixture_horizon
    from fpl_intelligence.storage import Store, atomic_json, digest

    store = Store(tmp_path / "data")
    directory = store.root / "features"
    directory.mkdir(exist_ok=True)
    old_path, labels = directory / "old.parquet", directory / "labels.parquet"
    keys = {"season": "2026/27", "fixture_id": 1, "player_id": 3}
    old = pl.DataFrame(
        [
            keys
            | {
                "target": 8.0,
                "base_target": 8.0,
                "official_own_attack": 0.0,
                "official_opponent_defence": 1250.0,
            }
        ]
    )
    old.write_parquet(old_path)
    pl.DataFrame([keys | {"base_target": 6.0}]).write_parquet(labels)
    source = {"hash": "same-labels", "path": str(labels)}
    teams = [
        {
            k: 0
            for k in (
                "strength_attack_home",
                "strength_attack_away",
                "strength_defence_home",
                "strength_defence_away",
            )
        }
    ]
    packets = {("2026/27", 1): {"id": "same-packet", "bootstrap": {"teams": teams}}}
    legacy = {"data_version": 5}
    monkeypatch.setattr(fixture_horizon, "LEGACY_V5_TRANSFORM", digest(legacy))
    old_signature = digest({"packets": ["same-packet"], "labels": source["hash"], "transform": legacy})
    atomic_json(
        old_path.parent / "lineage.json",
        {"transform": legacy, "signature": old_signature, "label_hash": source["hash"]},
    )
    store.put(
        "future_horizon_model",
        {"data_path": str(old_path), "data_hash": hashlib.sha256(old_path.read_bytes()).hexdigest()},
    )
    destination = directory / "new" / "repaired.parquet"
    destination.parent.mkdir()
    assert fixture_horizon.repair_cached_data(store, destination, source, packets, {"data_version": 6}, "new")
    fixed = pl.read_parquet(destination)
    assert fixed["target"][0] == 8 and fixed["base_target"][0] == 6
    assert np.isnan(fixed["official_own_attack"][0])
    assert fixed["official_opponent_defence"][0] == 1250
    assert pl.read_parquet(old_path).equals(old)
    packets["2026/27", 1]["id"] = "different-information"
    assert not fixture_horizon.repair_cached_data(
        store, store.root / "unsafe.parquet", source, packets, {}, "other"
    )
