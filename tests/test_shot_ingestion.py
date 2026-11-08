import asyncio

import httpx
import pytest

from fpl_intelligence.shot_data import collect_shots, validate_payload
from fpl_intelligence.storage import Store
from fpl_intelligence.transport import PublicTransport


def match_data():
    person = {
        "player_id": "77",
        "player": "Test Player",
        "position": "MC",
        "time": "90",
        "xA": "0.1",
        "key_passes": "1",
        "xGChain": "0.2",
        "xGBuildup": "0.1",
    }
    return {
        "rosters": {"h": {"77": person}, "a": {}},
        "shots": {
            "h": [{"player_id": "77", "situation": "OpenPlay", "result": "Goal", "xG": "0.2"}],
            "a": [],
        },
    }


def test_http_200_html_is_diagnosed_without_retry_or_snapshot(tmp_path):
    store = Store(tmp_path)
    transport = PublicTransport(store)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text="<!DOCTYPE html><title>Error</title>")
            )
        ) as client:
            with pytest.raises(ValueError, match="HTML instead of JSON"):
                await transport.get(
                    client, "https://understat.com/getLeagueData/EPL/2025", "shot_league_2025"
                )

    asyncio.run(run())
    assert len(store.list("source_request", 10)) == 1
    assert store.get("source_request")["status"] == 200
    with pytest.raises(ValueError, match="No stored snapshot"):
        store.latest("shot_league_2025")


@pytest.mark.parametrize(
    "invalid", [{}, {"dates": {}}, {"dates": [None]}, {"dates": [{"id": 1, "datetime": "bad"}]}]
)
def test_changed_league_schema_is_rejected(invalid):
    with pytest.raises(ValueError, match="league response has an invalid schema"):
        validate_payload(invalid, league=True)


@pytest.mark.parametrize("field,value", [("xA", "nan"), ("time", -1), ("key_passes", "not a number")])
def test_invalid_roster_statistics_are_not_cached(field, value):
    data = match_data()
    data["rosters"]["h"]["77"][field] = value
    with pytest.raises(ValueError, match="match response has an invalid schema"):
        validate_payload(data)


def test_unknown_shot_situation_is_not_silently_a_set_piece():
    data = match_data()
    data["shots"]["h"][0]["situation"] = "NewUnknownCategory"
    with pytest.raises(ValueError, match="Unknown shot situation"):
        validate_payload(data)


def test_partial_refresh_retains_verified_matches_and_rejects_bad_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path)
    old = {"season": "2024/25", "player_code": 42, "fixture_id": 3, "sentinel": "retained"}
    store.put("shot_archive", {"matched": 1, "records": [old]})
    store.snapshot("shot_league_2024", "https://understat.com/", {"error": "bad cached shape"})
    store.snapshot("shot_match_10", "https://understat.com/", {"rosters": []})
    monkeypatch.setattr(
        "fpl_intelligence.shot_data.canonical_rows",
        lambda *_: (
            [
                {
                    "season": "2025/26",
                    "element": 1,
                    "code": 88,
                    "name": "Test Player",
                    "position": 3,
                    "fixture": 9,
                    "round": 1,
                    "team": "Arsenal",
                    "kickoff_time": "2025-08-15T15:00:00Z",
                }
            ],
            {},
            {},
        ),
    )
    monkeypatch.setattr("fpl_intelligence.shot_data.load_market", lambda *_: ({}, {"season": "2025/26"}))

    async def fetch(self, client, url, source, **kwargs):
        if source == "shot_league_2024":
            return {"message": "temporarily unavailable"}
        if source == "shot_league_2025":
            return {
                "dates": [
                    {
                        "id": 10,
                        "datetime": "2025-08-15 15:00:00",
                        "isResult": True,
                        "h": {"title": "Arsenal"},
                        "a": {"title": "Chelsea"},
                    }
                ]
            }
        return match_data()

    monkeypatch.setattr(PublicTransport, "get", fetch)
    result = asyncio.run(collect_shots(store, [2024, 2025]))
    assert result["matched"] == 2 and result["retained_observations"] == 1
    assert result["records"][0] == old
    assert result["records"][1]["xg"]["open_play"] == 0.2
    assert result["errors"][0]["detail"].startswith("Understat league response")
    assert store.latest("shot_league_2024")[0] == {"error": "bad cached shape"}
    assert store.latest("shot_match_10")[0] == match_data()


def test_tactical_outfield_role_and_complete_surname_prefix_are_valid_identities():
    from fpl_intelligence.shot_data import identify

    candidates = [{"name": "Alex Example Extra", "element": 1, "position": 3}]
    assert identify("Alex Example", "DL", candidates) == candidates[0]
    assert identify("Alex Example", "GK", candidates) is None
    keeper = [{"name": "Alex Example", "element": 5, "position": 1}]
    assert identify("Alex Example", "Sub", keeper) == keeper[0]
    assert identify("Alex", "DL", candidates) is None
    assert (
        identify(
            "Alex Example", "DL", candidates + [{"name": "Alex Example Other", "element": 2, "position": 2}]
        )
        is None
    )


def test_offline_rejoin_cannot_fetch_an_unavailable_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "data")
    store.put(
        "shot_archive", {"matched": 1, "records": [{"season": "2025/26", "player_code": 42, "fixture_id": 9}]}
    )
    monkeypatch.setattr("fpl_intelligence.shot_data.canonical_rows", lambda *_: ([], {}, {}))
    monkeypatch.setattr("fpl_intelligence.shot_data.load_market", lambda *_: ({}, {"season": "2025/26"}))

    async def forbidden(*args, **kwargs):
        raise AssertionError("Offline rejoin attempted a network request")

    monkeypatch.setattr(PublicTransport, "get", forbidden)
    result = asyncio.run(collect_shots(store, [2025], offline=True))
    assert result["matched"] == 1 and result["records"][0]["player_code"] == 42


def test_offline_rejoin_job_is_integrated_and_returns_a_bounded_summary(tmp_path, monkeypatch):
    import time

    from fastapi.testclient import TestClient

    from fpl_intelligence.api import create_app

    calls = []

    async def rejoin(store, seasons=None, *, offline=False):
        calls.append(offline)
        return {
            "id": "checked",
            "matched": 1,
            "records": [{"sensitive_large_corpus": "kept in immutable store"}],
        }

    monkeypatch.setattr("fpl_intelligence.shot_data.collect_shots", rejoin)
    with TestClient(create_app(tmp_path / "data"), base_url="http://127.0.0.1") as client:
        reply = client.post("/api/football/shots/rejoin", json={})
        assert reply.status_code == 200
        for _ in range(100):
            job = client.get("/api/jobs/" + reply.json()["job_id"]).json()
            if job["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.01)
        assert job["status"] == "completed" and calls == [True]
        assert job["result"]["matched"] == 1 and "records" not in job["result"]


def test_official_short_alias_is_accepted_only_with_unique_fixture_identity():
    from fpl_intelligence.shot_data import identify

    keeper = {
        "name": "Firstname Middle Lastname",
        "identity_aliases": ["OfficialNickname"],
        "element": 1,
        "position": 1,
    }
    assert identify("OfficialNickname", "GK", [keeper]) == keeper
    other = {**keeper, "element": 2}
    assert identify("OfficialNickname", "GK", [keeper, other]) is None


def test_stable_provider_links_require_bijection_multiple_fixtures_and_current_club():
    from fpl_intelligence.shot_data import club_key, identify_stable, stable_id_links

    history = [
        {"source_player_id": "7", "player_code": 88, "season": "2024/25", "fixture_id": i} for i in range(3)
    ]
    links = stable_id_links(history)
    assert links == {"7": 88}
    candidate = {"code": 88, "position": 3, "element": 42}
    assert identify_stable("7", "DL", [candidate], links) == candidate
    assert identify_stable("7", "GK", [candidate], links) is None
    assert identify_stable("7", "DL", [], links) is None
    assert stable_id_links(history[:2]) == {}
    assert (
        stable_id_links(
            history + [{"source_player_id": "8", "player_code": 88, "season": "2024/25", "fixture_id": 9}]
        )
        == {}
    )
    assert (
        stable_id_links(
            history + [{"source_player_id": "7", "player_code": 99, "season": "2024/25", "fixture_id": 9}]
        )
        == {}
    )
    assert club_key("Sheffield United") == club_key("Sheffield Utd")
    assert club_key("Ipswich Town") == club_key("Ipswich")


def test_cancelled_cache_rejoin_preserves_the_published_archive(tmp_path, monkeypatch):
    from concurrent.futures import CancelledError
    from threading import Event

    from fpl_intelligence.search_control import search_session

    store = Store(tmp_path / "data")
    previous = store.put("shot_archive", {"matched": 1, "records": []})
    event = Event()

    def cancelled_setup(*args):
        event.set()
        return ({}, {"season": "2025/26"})

    monkeypatch.setattr("fpl_intelligence.shot_data.canonical_rows", lambda *_: ([], {}, {}))
    monkeypatch.setattr("fpl_intelligence.shot_data.load_market", cancelled_setup)
    with search_session(cancel=event):
        with pytest.raises(CancelledError):
            asyncio.run(collect_shots(store, [2025], offline=True))
    assert store.get("shot_archive")["id"] == previous
