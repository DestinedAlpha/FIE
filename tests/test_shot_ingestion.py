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
