import asyncio
from datetime import timedelta

import pytest

from fpl_intelligence.readiness import selection_is_current, workload_is_current
from fpl_intelligence.storage import Store
from fpl_intelligence.transport import PublicTransport
from fpl_intelligence.workload import (
    REPO,
    import_workload,
    migrate_cached_workload,
    refresh_workload_due,
    time_value,
)


@pytest.mark.parametrize("retained", [True, False])
def test_partial_workload_fetch_does_not_replace_complete_cache(tmp_path, monkeypatch, retained):
    store = Store(tmp_path)
    previous = (
        store.put("workload", {"records": [{"player_code": 7}], "sentinel": "previous"}) if retained else None
    )
    prefix = "data/2026-2027/By Tournament/Cup/GW1/"
    paths = [prefix + "fixtures.csv", prefix + "playermatchstats.csv"]

    async def get(self, client, url, source, **kwargs):
        if url.endswith("commits/main"):
            return {"sha": "a" * 40}
        if "/git/trees/" in url:
            return {"tree": [{"path": p} for p in paths], "truncated": False}
        if url.endswith("players.csv"):
            return "player_id,player_code\n1,7\n"
        if url.endswith("fixtures.csv"):
            return "match_id,kickoff_time,finished\nmatch,2026-10-01T18:00:00Z,true\n"
        return "<html>upstream maintenance</html>"

    monkeypatch.setattr(PublicTransport, "get", get)
    result = asyncio.run(import_workload(store, ("2026-2027",)))
    assert result["retained_id"] == previous
    assert not result.get("records") and result["errors"]
    assert (store.get("workload")["id"] if retained else None) == previous
    if not retained:
        assert not store.list("workload", 1)


def test_legacy_workload_is_reparsed_from_pinned_raw_and_original_is_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "data")
    commit = "b" * 40
    season = "2026-2027"
    prefix = f"data/{season}/By Tournament/Cup/"
    bodies = {f"data/{season}/players.csv": [{"player_id": "1", "player_code": "7"}]}
    for week, score in [(1, 1), (2, 2)]:
        base = prefix + f"GW{week}/"
        bad = {
            "match_id": "bad-2026-10-01",
            "kickoff_time": "2026-10-01T18:00:00Z",
            "finished": "true",
            "score": score,
        }
        good = {"match_id": "good-2026-10-02", "kickoff_time": "2026-10-02T18:00:00Z", "finished": "true"}
        bodies[base + "fixtures.csv"] = [bad] + ([good] if week == 1 else [])
        bodies[base + "playermatchstats.csv"] = [
            {"match_id": bad["match_id"], "player_id": "1", "minutes_played": "90"}
        ]
        if week == 1:
            bodies[base + "playermatchstats.csv"].append(
                {"match_id": good["match_id"], "player_id": "1", "minutes_played": "0"}
            )
    for path, rows in bodies.items():
        store.snapshot(
            "workload_" + path.replace("/", "_"),
            f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}",
            {"commit": commit, "rows": rows},
        )
    previous = store.put(
        "workload",
        {
            "at": "2026-10-03T00:00:00Z",
            "commit": commit,
            "requested_files": len(bodies),
            "records": [{"player_code": 7}],
        },
    )
    state = store.put("state", {"bank": 3})
    result = migrate_cached_workload(store)
    assert result["conflicting_duplicates"] == 1 and result["accepted_player_matches"] == 1
    assert result["records"][0]["minutes"] == 0
    assert result["source_fetched_at"] == "2026-10-03T00:00:00Z"
    assert store.get("workload", previous)["records"] == [{"player_code": 7}]
    assert store.get("state")["id"] == state
    assert migrate_cached_workload(store) is None


def test_changed_workload_invalidates_a_plan_without_changing_account_or_model(tmp_path):
    store = Store(tmp_path)
    source = store.put("workload", {"records": []})
    model = {
        "id": store.put("decision_model", {"role_minutes_model_id": "accepted-role"}),
        "role_minutes_model_id": "accepted-role",
    }
    state = store.put("state", {"event": 11})
    market = store.put("market", {"events": []})
    fid = store.put(
        "forecast",
        {
            "decision_model_id": model["id"],
            "market_id": market,
            "input_manifest": {"live_workload_id": source},
        },
    )
    forecast = store.get("forecast")
    plan = {"state_id": state, "market_id": market, "forecast_id": fid}
    assert workload_is_current(store, forecast, model) and selection_is_current(store, plan)
    store.put("workload", {"records": [], "new_capture": True})
    assert not workload_is_current(store, forecast, model) and not selection_is_current(store, plan)
    assert store.get("state")["id"] == state and store.get("decision_model")["id"] == model["id"]


def test_daily_workload_refresh_and_failed_attempt_backoff(tmp_path, monkeypatch):
    store = Store(tmp_path)
    clock = ["2026-11-07T12:00:00Z"]
    calls = []
    monkeypatch.setattr("fpl_intelligence.workload.now", lambda: clock[0])

    async def refresh(_):
        calls.append(clock[0])
        raise ValueError("temporary failure")

    monkeypatch.setattr("fpl_intelligence.workload.import_workload", refresh)
    assert not refresh_workload_due(store)["success"]
    refresh_workload_due(store)
    assert len(calls) == 1
    clock[0] = (time_value(clock[0]) + timedelta(hours=5)).isoformat()
    refresh_workload_due(store)
    assert len(calls) == 2
