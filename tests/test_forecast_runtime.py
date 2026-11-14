import asyncio

import httpx
import numpy as np
from fastapi.testclient import TestClient

from fpl_intelligence.api import create_app
from fpl_intelligence.decision import NAMES, context_vector, future_inputs, strength
from fpl_intelligence.domain import Action
from fpl_intelligence.optimization import refine_freehit_lineup
from fpl_intelligence.rules import best_lineup, transition
from fpl_intelligence.simulation import lineup_worlds
from fpl_intelligence.storage import Store
from fpl_intelligence.transport import PublicTransport, SourceDeferred


def test_rate_limit_survives_a_new_refresh_client_without_more_requests(tmp_path):
    store = Store(tmp_path)
    calls = []

    def limited(request):
        calls.append(request.url)
        return httpx.Response(429, headers={"Retry-After": "180"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(limited)) as client:
            for path in ("FantasyPL", "fplAnalytics"):
                try:
                    await PublicTransport(store).get(client, f"https://www.reddit.com/r/{path}/.rss", path)
                except SourceDeferred as error:
                    assert error.host == "www.reddit.com" and error.retry_at
                else:
                    raise AssertionError("Rate-limited community source must be deferred")

    asyncio.run(run())
    assert len(calls) == 1
    assert store.get("source_request")["status"] == 429


def test_job_worker_timestamp_does_not_overwrite_a_later_forecast_phase(tmp_path):
    store = Store(tmp_path)
    with store.db() as db:
        db.execute(
            "INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", ("job", "running", "start", "start", "{}", None, None)
        )
    store.put("job_worker", {"job_id": "job", "pid": 17600, "at": "2026-11-14T13:44:30Z"})
    store.put("job_phase", {"job_id": "job", "phase": "Simulating GW12", "at": "2026-11-14T13:46:00Z"})
    response = TestClient(create_app(tmp_path), base_url="http://127.0.0.1").get("/api/jobs/job").json()
    assert response["progress"]["at"] == response["progress"]["phase_at"] == "2026-11-14T13:46:00Z"
    assert response["progress"]["worker_started_at"] == "2026-11-14T13:44:30Z"


def test_cached_live_player_profiles_match_the_canonical_historical_transform(tmp_path, market, monkeypatch):
    store = Store(tmp_path)
    player = market[3]
    history = [
        {
            "code": player.code,
            "minutes": 90,
            "starts": 1,
            "total_points": 6,
            "kickoff_time": "2026-10-01T14:00:00Z",
        }
    ]
    meta = {("2026/27", 1): {"home": "Own", "away": "Opp", "event": 1, "home_goals": 2, "away_goals": 1}}
    monkeypatch.setattr("fpl_intelligence.decision.canonical_rows", lambda *_: (history, meta, {}))
    fixture = {
        "id": 100,
        "event": 10,
        "team_h": player.team,
        "team_a": 20,
        "kickoff_time": "2026-11-01T14:00:00Z",
    }
    market_data = {
        "teams": [{"id": player.team, "name": "Own"}, {"id": 20, "name": "Opp"}],
        "fixtures": [fixture],
    }
    rows, _ = future_inputs(store, market_data, {player.id: player}, 10, 10)
    expected, profile = context_vector(
        player.position, True, history, fixture["kickoff_time"], strength([(2, 1)]), strength([(1, 2)]), 10
    )
    assert len(rows) == 1 and len(expected) == len(NAMES)
    np.testing.assert_array_equal(rows[0]["x"], expected)
    assert rows[0]["profile"] == profile


def test_freehit_refinement_values_keeper_fallback_and_restores_the_same_account(market, rules, state):
    ids = sorted(market)
    points = np.zeros((40, len(ids)), dtype=np.int16)
    minutes = np.zeros_like(points)
    points[:, :15] = 5
    minutes[:, :15] = 90
    points[:, 1] = 0
    minutes[:, 1] = 0
    points[:20, 0] = 8
    points[20:, 0] = 0
    minutes[20:, 0] = 0
    points[:, 15] = 2
    minutes[:, 15] = 90
    means = dict(zip(ids, points.mean(axis=0)))
    original = best_lineup(list(range(1, 16)), means, market, rules)
    chosen, evidence = refine_freehit_lineup(state, original, means, market, rules, points, minutes, ids, 3)
    before = lineup_worlds(original, points, minutes, ids, market, rules, "freehit")
    after = lineup_worlds(chosen, points, minutes, ids, market, rules, "freehit")
    assert after.mean() >= before.mean() + 0.5
    assert evidence["after"] >= evidence["before"]
    owned = {lot.player_id for lot in state.lots}
    playing = set(chosen.xi + chosen.bench)
    changed, _ = transition(
        state,
        Action(sell=sorted(owned - playing), buy=sorted(playing - owned), lineup=chosen, chip="freehit"),
        market,
        rules,
    )
    held, _ = transition(state, Action(lineup=original, chip="freehit"), market, rules)
    assert changed.model_dump() == held.model_dump()
