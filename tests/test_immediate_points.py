import numpy as np
import pytest

from fpl_intelligence.context_archive import FIELDS, verified_cached_bootstrap
from fpl_intelligence.fusion_data import SEQUENCE_FIELDS, sequence
from fpl_intelligence.immediate_points import TEAM_NAMES, TeamForm, conditional, matrix, names
from fpl_intelligence.storage import digest


def inputs(past):
    seq, mask = sequence(past)
    return {
        "tab": np.zeros((1, 26)),
        "sequence": seq[None],
        "sequence_mask": mask[None],
        "enrichment": np.zeros((1, 31)),
        "team_form": np.zeros((1, len(TEAM_NAMES))),
    }


def test_recent_underlying_statistics_and_missingness_reach_points_independently_of_price():
    past = [
        {"minutes": 90, "starts": 1, "expected_goals": 0.1, "expected_assists": 0.2, "total_points": 2}
    ] * 4
    earlier = matrix(inputs(past), [{"minutes": 360, "expected_goals": 0.4}], "advanced")
    improved = past + [
        {"minutes": 90, "starts": 1, "expected_goals": 0.8, "expected_assists": 0.4, "total_points": 2}
    ]
    later = matrix(inputs(improved), [{"minutes": 450, "expected_goals": 1.2}], "advanced")
    columns = names("advanced")
    assert "price_lag" not in columns and earlier.shape[1] == len(columns)
    assert later[0, columns.index("last3_expected_goals")] > earlier[0, columns.index("last3_expected_goals")]
    assert later[0, columns.index("last1_total_points")] == earlier[0, columns.index("last1_total_points")]
    assert np.isnan(later[0, columns.index("season_creativity90")])
    assert later[0, columns.index("known_creativity")] == 0
    known_zero = matrix(inputs(improved), [{"minutes": 450, "creativity": 0}], "advanced")
    assert known_zero[0, columns.index("known_creativity")] == 1
    assert known_zero[0, columns.index("season_creativity90")] == 0


def test_nonappearance_is_in_recent_points_and_double_fixture_history_is_not_mutated():
    past = [{"minutes": 90, "starts": 1, "total_points": 10}, {"minutes": 0, "starts": 0, "total_points": 0}]
    data = inputs(past)
    before = data["sequence"].copy()
    result = matrix(data, [{}], "form")
    assert result[0, names("form").index("last1_total_points")] == 0
    assert result[0, names("form").index("last3_total_points")] == 5
    assert np.array_equal(before, data["sequence"])
    assert result[0, names("form").index("last1_known_expected_goals")] == 0
    assert len(SEQUENCE_FIELDS) == 16


def test_advanced_archive_upgrade_requires_the_same_predeadline_proof_and_payload():
    proof = {
        k: str(i)
        for i, k in enumerate(
            (
                "season",
                "event",
                "deadline",
                "sha256",
                "url",
                "commit",
                "source_captured_at",
                "git_committed_at",
            )
        )
    }
    body = {"elements": [{"expected_goals": "1.2", "threat": "140.0"}]}
    record = {"bootstrap": body, "bootstrap_hash": digest(body), "bootstrap_proof": proof}
    assert verified_cached_bootstrap(record, proof) is body
    assert verified_cached_bootstrap(record, proof | {"sha256": "other"}) is None
    with pytest.raises(ValueError, match="checksum"):
        verified_cached_bootstrap(record | {"bootstrap_hash": "bad"}, proof)
    assert {
        "expected_goals",
        "expected_assists",
        "threat",
        "creativity",
        "bps",
        "defensive_contribution",
    } <= set(FIELDS)


def test_conditional_scoring_marginalizes_bands_and_does_not_relabel_absence():
    class Head:
        def predict(self, values):
            return values[:, -1] * 2

    result = conditional(Head(), np.zeros((2, 3)), {"1": -0.5, "2": 0.1, "3": 0.2})
    assert np.all(result[:, 0] == 0)
    assert np.allclose(result[0], [0, 1.5, 4.1, 6.2])


def test_team_form_excludes_future_labels_and_preserves_missing_xg():
    def match(day, fixture, goals, xg):
        return [
            {
                "season": "2025/26",
                "round": fixture,
                "fixture": fixture,
                "was_home": home,
                "team": "A" if home else "B",
                "minutes": 90,
                "kickoff_time": f"2025-08-{day:02d}T15:00:00Z",
                "team_h_score": goals,
                "team_a_score": 0,
                "expected_goals": xg if home else None,
            }
            for home in (True, False)
        ]

    rows = match(1, 1, 2, 1.1) + match(8, 2, 0, 0.2) + match(15, 3, 10, 8.0)
    before = TeamForm(rows).vector("A", "B", "2025-08-15T12:00:00Z", "2025/26-GW03")
    assert before[0] == 1 and np.isnan(before[2]) and before[4] == 0
    for r in rows:
        if r["fixture"] == 3:
            r["team_h_score"] = 1000
            r["expected_goals"] = 1000
    after = TeamForm(rows).vector("A", "B", "2025-08-15T12:00:00Z", "2025/26-GW03")
    np.testing.assert_equal(before, after)


def test_local_context_upgrade_uses_the_original_full_bootstrap_not_later_totals(tmp_path):
    from fpl_intelligence.context_archive import SCHEMA_VERSION, add_local_context
    from fpl_intelligence.storage import Store

    store = Store(tmp_path)
    deadline = "2026-10-30T18:30:00Z"
    source = "context_live_2026-27_9"
    store.snapshot(
        source,
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        {
            "season": "2026/27",
            "event": 9,
            "deadline": deadline,
            "elements": [{"code": 777, "id": 7}],
        },
    )
    bootstrap = {
        "events": [
            {"id": 1, "deadline_time": "2026-08-21T17:30:00Z", "is_next": False},
            {"id": 9, "deadline_time": deadline, "is_next": True},
        ],
        "elements": [{"code": 777, "id": 7, "starts": 7, "minutes": 630, "expected_goals": "1.5"}],
    }
    raw = store.snapshot("bootstrap", "https://fantasy.premierleague.com/api/bootstrap-static/", bootstrap)
    with store.db() as db:
        db.execute("UPDATE snapshots SET captured_at='2026-10-29T12:00:00+00:00'")
    later = {**bootstrap, "elements": [{"code": 777, "id": 7, "starts": 20, "minutes": 1800}]}
    store.snapshot("bootstrap", "https://fantasy.premierleague.com/api/bootstrap-static/", later)
    restored = add_local_context(store, [])[0]
    body, _ = store.latest(restored["source"], content_hash=restored["hash"])
    assert body["schema_version"] == SCHEMA_VERSION
    assert body["elements"][0]["starts"] == 7 and body["elements"][0]["expected_goals"] == "1.5"
    assert body["origin_bootstrap_hash"] == raw["hash"]
    assert restored["source_captured_at"] == "2026-10-29T12:00:00+00:00"
    assert add_local_context(store, [])[0]["hash"] == restored["hash"]


def test_online_bias_uses_only_completed_earlier_forecast_errors():
    from fpl_intelligence.online_calibration import evaluate

    rows = []
    for event, day in enumerate((1, 8, 15, 22), 1):
        for code, residual in ((777, -3.0), (999, 0.0)):
            rows.append(
                {
                    "deadline_group": f"2025/26-GW{event:02d}",
                    "deadline": f"2025-08-{day:02d}T12:00:00Z",
                    "known_after": __import__("datetime")
                    .datetime(2025, 8, day + 1, tzinfo=__import__("datetime").timezone.utc)
                    .timestamp(),
                    "code": code,
                    "position": 3,
                    "minutes": 90,
                    "reference": 5.0,
                    "exposure": 1.0,
                    "actual": 5 + residual,
                    "residual90": residual,
                }
            )
    before = evaluate(rows)
    changed = [dict(r) for r in rows]
    changed[4].update(actual=12.0, residual90=7.0)
    after = evaluate(changed)
    for a, b in zip(before[:6], after[:6]):
        assert a["bias_0.8_4"] == b["bias_0.8_4"]
    assert before[6]["bias_0.8_4"] != after[6]["bias_0.8_4"]
