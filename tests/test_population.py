import numpy as np
import pytest

from fpl_intelligence.football_challenger import (
    _past_matches,
    allocation,
    match_metrics,
    rate_vector,
    score_matrix,
)
from fpl_intelligence.forecast_population import binary_metrics, gameweek_rows


def test_bivariate_poisson_preserves_marginal_goals_and_clean_sheets():
    independent = score_matrix(1.7, 0.8)
    shared = score_matrix(1.7, 0.8, 0.15)
    grid = np.arange(len(shared))
    assert shared.sum() == pytest.approx(1)
    assert shared.sum(1) @ grid == pytest.approx(1.7)
    assert shared.sum(0) @ grid == pytest.approx(0.8)
    assert shared[:, 0].sum() == pytest.approx(np.exp(-0.8))
    assert shared[0, :].sum() == pytest.approx(np.exp(-1.7))
    h, a = np.indices(shared.shape)
    covariance = np.sum(shared * (h - 1.7) * (a - 0.8))
    assert covariance == pytest.approx(0.15)
    assert np.trace(shared) > np.trace(independent)
    metrics = match_metrics(shared, 2, 0)
    assert metrics["home_mean"] == pytest.approx(1.7)
    assert 0 <= metrics["rps"] <= 1


@pytest.mark.parametrize("values", [(1, 1, 2), (float("nan"), 1, 0), (-1, 1, 0)])
def test_invalid_score_rates_are_rejected(values):
    with pytest.raises(ValueError):
        score_matrix(*values)


def test_grid_tail_is_not_silently_renormalized():
    with pytest.raises(ValueError, match="truncates"):
        score_matrix(12, 12, 0, maximum=4)


def test_team_opportunity_allocation_conserves_goals_and_respects_minutes():
    allocated = allocation([0.4, 0.2, 0.1], [0, 90, 45], 2.5)
    assert allocated.sum() == pytest.approx(2.5)
    assert allocated[0] == 0
    assert allocated[1] / allocated[2] == pytest.approx(4)


def test_same_deadline_and_incomplete_matches_never_update_team_strength():
    fixtures = [
        {"group": "2025/26-GW01", "kickoff": "2025-08-10T12:00:00Z"},
        {"group": "2025/26-GW02", "kickoff": "2025-08-11T12:00:00Z"},
        {"group": "2025/26-GW01", "kickoff": "2025-08-12T12:00:00Z"},
    ]
    assert _past_matches(fixtures, "2025/26-GW02", "2025-08-12T14:00:00Z") == fixtures[:1]


def test_missing_advanced_stats_do_not_become_zero_observations():
    prior = np.array([0.3, 0.2, 0.4, 0.1, 0.01, 3.0, 0.01, 0.01, 0.01])
    missing = [{"season": "2025/26", "minutes": 90}]
    assert rate_vector(missing, "2025/26", prior, 0.3) == pytest.approx(prior)


def test_calibration_includes_probability_one_and_gameweeks_sum_both_fixtures():
    result = binary_metrics([0, 1, 1], [0, 0.5, 1])
    assert result["n"] == sum(b["n"] for b in result["bins"]) == 3
    assert result["brier"] == pytest.approx(0.25 / 3)
    weekly = gameweek_rows(
        [
            {"deadline_group": "S-GW01", "player_id": 1, "actual": 2, "reference": 3},
            {"deadline_group": "S-GW01", "player_id": 1, "actual": 6, "reference": 4},
        ]
    )
    assert weekly[0]["actual"] == 8 and weekly[0]["reference"] == 7 and weekly[0]["fixture_count"] == 2


def test_component_goals_reconcile_to_team_and_fpl_assists_keep_distinct_units(monkeypatch):
    from fpl_intelligence import decision
    from fpl_intelligence.football_challenger import component_predictions

    originals = []
    rows = []
    for pid, pos in ((1, 2), (2, 4)):
        common = {"element": pid, "code": pid, "position": pos, "team": "Arsenal", "was_home": True}
        originals.append(
            {
                **common,
                "season": "2024/25",
                "round": 30,
                "fixture": 10,
                "kickoff_time": "2025-04-01T12:00:00Z",
                "minutes": 90,
                "expected_goals": 0.5,
                "expected_assists": 0.1,
                "goals_scored": 1,
                "assists": 1,
            }
        )
        originals.append(
            {
                **common,
                "season": "2025/26",
                "round": 5,
                "fixture": 99,
                "kickoff_time": "2025-10-02T12:00:00Z",
                "minutes": 90,
                "expected_goals": 99,
                "expected_assists": 99,
                "goals_scored": 99,
                "assists": 99,
            }
        )
        row = {
            "season": "2025/26",
            "event": 5,
            "fixture_id": 99,
            "player_id": pid,
            "position": pos,
            "team": "Arsenal",
            "deadline_group": "2025/26-GW05",
            "deadline": "2025-10-01T11:00:00Z",
            "expected_minutes": 67.5,
        }
        row.update({f"p{k}": v for k, v in enumerate((0.1, 0.1, 0.3, 0.5))})
        row.update({f"m{k}": v for k, v in enumerate((0, 15, 70, 90))})
        row.update({f"dc{k}": 0 for k in range(4)})
        rows.append(row)
    monkeypatch.setattr(decision, "canonical_rows", lambda store: (originals, {}, {}))
    teams = [
        {
            "season": "2025/26",
            "fixture_id": 99,
            "alpha": 0,
            "covariance": 0,
            "home": "arsenal",
            "away": "liverpool",
            "home_mean": 1.5,
            "away_mean": 0.8,
        }
    ]
    output = component_predictions(None, rows, teams, {}, 0, 0.3)
    assert sum(r["xg"] for r in output) == pytest.approx(1.5)
    assert sum(r["component_goals"] / (6 if r["position"] == 2 else 4) for r in output) == pytest.approx(1.5)
    assert sum(r["component_assists"] / 3 for r in output) == pytest.approx(
        sum(r["expected_fpl_assists"] for r in output)
    )
    assert sum(r["xa"] for r in output) == pytest.approx(0.3)
    assert sum(r["expected_fpl_assists"] for r in output) == pytest.approx(1.5)
    # Changing held-out labels cannot change any forecast component.
    originals[-1]["expected_goals"] = 0.01
    originals[-1]["assists"] = 0
    again = component_predictions(None, rows, teams, {}, 0, 0.3)
    assert [r["component"] for r in again] == pytest.approx([r["component"] for r in output])


def test_bundled_diagnostics_work_for_existing_installs_without_importing_state(tmp_path, monkeypatch):
    import json

    from fpl_intelligence import forecast_population
    from fpl_intelligence.storage import Store

    s = Store(tmp_path / "data")
    monkeypatch.setattr(forecast_population, "EVIDENCE_ROOT", tmp_path)
    (tmp_path / "evidence.json").write_text(json.dumps({"id": "bundled", "provenance": "historical"}))
    assert forecast_population.saved_evidence(s, "population_diagnostics", "evidence.json")["id"] == "bundled"
    assert not s.list("population_diagnostics", 1) and not s.list("state", 1)
    live = s.put("population_diagnostics", {"provenance": "this installation"})
    assert forecast_population.saved_evidence(s, "population_diagnostics", "evidence.json")["id"] == live


def test_research_predictions_are_immutable_when_candidates_change(tmp_path):
    from fpl_intelligence.football_challenger import seal_predictions

    first, checksum = seal_predictions(tmp_path, "experiment", [{"prediction": 3.0}])
    old = first.read_bytes()
    second, _ = seal_predictions(tmp_path, "experiment", [{"prediction": 4.0}])
    assert second != first and first.read_bytes() == old
    assert seal_predictions(tmp_path, "experiment", [{"prediction": 3.0}]) == (first, checksum)
    first.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        seal_predictions(tmp_path, "experiment", [{"prediction": 3.0}])
