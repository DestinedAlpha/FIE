import hashlib

import polars as pl
import pytest

from fpl_intelligence.observations import completed_before, evaluation_report, observation_view
from fpl_intelligence.storage import Store


def test_observation_cutoff_excludes_target_future_and_unfinished_matches():
    def row(season, gameweek, kickoff, value):
        return {"season": season, "round": gameweek, "kickoff_time": kickoff, "total_points": value}

    rows = [
        row("2026/27", 7, "2026-10-24T15:00:00Z", 2),
        row("2026/27", 8, "2026-10-31T07:00:00Z", 3),
        row("2026/27", 9, "2026-10-24T15:00:00Z", 100),
        row("2026/27", 10, "2026-10-24T15:00:00Z", 1000),
    ]
    assert completed_before(rows, "2026-10-31T11:00:00Z", "2026/27-GW09") == rows[:1]
    for r in rows[1:]:
        r["total_points"] = 999999
    assert completed_before(rows, "2026-10-31T11:00:00Z", "2026/27-GW09") == rows[:1]


def test_normal_forecaster_reads_only_the_verified_historical_view(tmp_path, market, monkeypatch):
    from fpl_intelligence.decision import canonical_rows

    store = Store(tmp_path / "data")
    path = store.root / "features" / "history.parquet"
    path.parent.mkdir()
    rows = [
        {
            "season": "2026/27",
            "round": week,
            "element": p,
            "code": p,
            "fixture": week,
            "kickoff_time": "2026-10-24T15:00:00Z" if week == 8 else "2026-10-31T15:00:00Z",
            "team": "Home" if p == 1 else "Away",
            "was_home": p == 1,
            "starts": 1,
            "minutes": 90,
            "team_h_score": 1,
            "team_a_score": 0,
            "total_points": week,
        }
        for week in (8, 9)
        for p in (1, 2)
    ]
    pl.DataFrame(rows).write_parquet(path)
    store.put("replay_context", {"information_cutoff": "2026-10-31T11:00:00Z", "group": "2026/27-GW09"})
    spec = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cutoff": "2026-10-31T11:00:00Z",
        "group": "2026/27-GW09",
    }
    store.put("observation_frame", spec)
    monkeypatch.setattr(
        "fpl_intelligence.decision.load_market",
        lambda *_: (market, {"teams": [], "fixtures": [], "events": []}),
    )

    def no_live_fallback(*args, **kwargs):
        raise AssertionError("Historical forecasts must not read today's history snapshots")

    monkeypatch.setattr("fpl_intelligence.decision.histories", no_live_fallback)
    actual, fixtures, _ = canonical_rows(store)
    assert len(actual) == 2 and {r["round"] for r in actual} == {8}
    assert set(fixtures) == {("2026/27", 8)}
    store.put("observation_frame", spec | {"group": "2026/27-GW10"})
    with pytest.raises(ValueError, match="clock disagree"):
        observation_view(store)
    store.put("observation_frame", spec | {"sha256": "tampered"})
    with pytest.raises(ValueError, match="checksum"):
        observation_view(store)


def test_replay_reports_do_not_overwrite_live_model_cards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "data")
    store.put("replay_context", {"information_cutoff": "2026-10-31T11:00:00Z"})
    evaluation_report(store, "availability.json", {"historical": True})
    assert (store.root / "reports" / "availability.json").exists()
    assert not (tmp_path / "artifacts/evaluations/availability.json").exists()


def test_replay_writer_lock_rejects_concurrent_work_and_releases_after_error(tmp_path):
    from fpl_intelligence.pipeline_jobs import replay_lock

    path = tmp_path / "replay.lock"
    with pytest.raises(RuntimeError):
        with replay_lock(path):
            with pytest.raises(ValueError, match="already running"):
                with replay_lock(path):
                    raise AssertionError("Second writer should never enter")
            raise RuntimeError("Worker failed")
    with replay_lock(path):
        assert path.exists()


def test_isolated_live_store_cannot_overwrite_production_research_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main = Store("data")
    isolated = Store(tmp_path / "verification" / "data")
    evaluation_report(main, "coverage.json", {"production": True})
    evaluation_report(isolated, "coverage.json", {"verification": True})
    import json

    assert json.loads((tmp_path / "artifacts/evaluations/coverage.json").read_text()) == {"production": True}
    assert json.loads((isolated.root / "reports/coverage.json").read_text()) == {"verification": True}
