import httpx
import numpy as np
import pytest

from fpl_intelligence.account_sync import free_transfers_after, reconcile_public, revealed_pick_events
from fpl_intelligence.discipline import BanDecision, ban_service, infer_caution_bans
from fpl_intelligence.domain import Action, Lot
from fpl_intelligence.objectives import cvar, select_policy
from fpl_intelligence.odds import devig, goal_constraints, score_markets
from fpl_intelligence.rules import quote, transition
from fpl_intelligence.score_process import log_scores
from fpl_intelligence.storage import Store
from fpl_intelligence.transport import PublicTransport, retry_seconds


def account_store(tmp_path, market, rules, state, monkeypatch, freehit=False):
    store = Store(tmp_path)
    monkeypatch.setattr("fpl_intelligence.account_sync.now", lambda: "2026-11-04T12:00:00+00:00")
    store.snapshot("bootstrap", "https://example.test/", {})
    store.put(
        "market",
        {
            "season": "2026/27",
            "players": [p.model_dump() for p in market.values()],
            "teams": [],
            "fixtures": [],
            "events": [{"id": g, "deadline_time": f"2026-10-{g + 1:02d}T11:00:00Z"} for g in range(1, 10)]
            + [{"id": 10, "deadline_time": "2026-11-06T18:30:00Z"}],
            "source_hash": store.latest("bootstrap")[1]["hash"],
        },
    )
    original = store.put("state", state.model_copy(update={"entry_id": 123, "event": 9}).model_dump())
    history = {
        "current": [
            {"event": g, "event_transfers": 1, "bank": 30 if g == 8 else 40, "total_points": 400 + g}
            for g in range(1, 10)
        ],
        "chips": [{"event": 9, "name": "freehit"}] if freehit else [],
    }
    store.snapshot("entry_history_123", "https://example.test/", history)
    store.snapshot("picks_123_8", "https://example.test/", {"picks": [{"element": p} for p in range(1, 16)]})
    store.snapshot(
        "picks_123_9", "https://example.test/", {"picks": [{"element": p} for p in [*range(2, 16), 16]]}
    )
    store.snapshot(
        "transfers_123",
        "https://example.test/",
        [{"event": 9, "element_in": 16, "element_out": 1, "element_in_cost": 51, "element_out_cost": 49}],
    )
    monkeypatch.setattr("fpl_intelligence.rules.Rules.from_bootstrap", lambda *_: rules)
    return store, original


def test_refresh_reconciles_revealed_transfer_exact_cost_and_is_idempotent(
    tmp_path, market, rules, state, monkeypatch
):
    store, old = account_store(tmp_path, market, rules, state, monkeypatch)
    result = reconcile_public(store, old)
    saved = store.get("state")
    assert result["account_changed"] and saved["event"] == 10
    assert 16 in [r["player_id"] for r in saved["lots"]] and 1 not in [r["player_id"] for r in saved["lots"]]
    assert saved["bank"] == 40 and saved["free_transfers"] == 1 and saved["hit_paid"] == 0
    assert next(r for r in saved["lots"] if r["player_id"] == 16)["purchase_price"] == 51
    assert reconcile_public(store)["account_changed"] is False
    assert store.get("state", old)["event"] == 9  # immutable original survives


def test_public_freehit_restores_underlying_squad_and_bank(tmp_path, market, rules, state, monkeypatch):
    store, _ = account_store(tmp_path, market, rules, state, monkeypatch, True)
    reconcile_public(store)
    saved = store.get("state")
    assert set(r["player_id"] for r in saved["lots"]) == set(range(1, 16))
    assert saved["bank"] == 30 and saved["free_transfers"] == 1


def test_refresh_preserves_newer_confirmed_hidden_transfers(tmp_path, market, rules, state, monkeypatch):
    store, _ = account_store(tmp_path, market, rules, state, monkeypatch)
    saved = state.model_copy(update={"entry_id": 123, "event": 10, "known_at": "2026-11-04T11:00:00+00:00"})
    identity = store.put("state", saved.model_dump())
    assert reconcile_public(store)["account_changed"] is False
    assert store.get("state")["id"] == identity


def test_ft_replay_preserves_saved_transfers_on_both_reset_chips(rules):
    history = {
        "current": [{"event": g, "event_transfers": 10 if g in (4, 5) else 0} for g in range(1, 7)],
        "chips": [{"event": 4, "name": "wildcard"}, {"event": 5, "name": "freehit"}],
    }
    assert free_transfers_after(history, 6, rules) == 4


def test_late_joiner_only_requests_entered_deadlines_and_gets_one_ft(rules):
    history = {"current": [{"event": 9, "event_transfers": 0}], "chips": []}
    assert revealed_pick_events(history, 9) == [9]
    assert free_transfers_after(history, 9, rules) == 1
    history["current"].append({"event": 10, "event_transfers": 1})
    assert revealed_pick_events(history, 9) == [9]  # later captures cannot supply future picks
    assert revealed_pick_events(history, 10) == [9, 10]
    with pytest.raises(ValueError, match="Missing revealed"):
        revealed_pick_events(history, 11)


def test_mixed_order_bundle_pairs_roles_and_charges_only_new_hit(market, rules, state):
    pending = state.model_copy(update={"free_transfers": 1, "transfers_made": 2, "hit_paid": 4})
    _, row = transition(pending, Action(sell=[3, 1], buy=[16, 18]), market, rules)
    assert [
        (market[a["player_id"]].position, market[b["player_id"]].position)
        for a, b in zip(row["sell"], row["buy"])
    ] == [(1, 1), (2, 2)]
    assert row["hit"] == 8 and row["event_total_hit"] == 12
    with pytest.raises(ValueError, match="confirmation"):
        quote(Lot(player_id=1, sell_price=50, quote_market_price=50, quote_confirmed=False), 50)


def test_ban_service_cups_postponement_and_appeal():
    decision = BanDecision(
        player_code=1,
        offence="serious_foul_play",
        imposed_at="2026-10-01T12:00:00+00:00",
        known_at="2026-10-01T18:00:00+00:00",
        source_url="https://www.thefa.com/decision",
    )
    fixtures = [
        {
            "id": i,
            "competition": comp,
            "kickoff_time": f"2026-10-{i + 2:02d}T12:00:00+00:00",
            "postponed": i == 2,
        }
        for i, comp in [(1, "LEAGUE_CUP"), (2, "PL"), (3, "PL"), (4, "FA_CUP"), (5, "PL")]
    ]
    assert ban_service(decision, fixtures, "2026-10-02T00:00:00+00:00")["fixture_ids"] == [1, 3, 4]
    assert (
        ban_service(
            decision.model_copy(update={"appeal": "rescinded"}), fixtures, "2026-10-02T00:00:00+00:00"
        )["fixture_ids"]
        == []
    )
    assert not ban_service(decision, fixtures, "2026-10-01T13:00:00+00:00")["active"]


def test_caution_threshold_uses_club_matches_not_gameweek_number():
    fixtures = [
        {
            "id": g,
            "competition": "PL",
            "home": "A",
            "away": "B",
            "kickoff_time": f"2026-10-{g:02d}T12:00:00+00:00",
        }
        for g in range(1, 23)
    ]
    rows = [
        {"team": "A", "fixture": g, "kickoff_time": fixtures[g - 1]["kickoff_time"], "yellow_cards": 1}
        for g in (1, 3, 6, 10, 19)
    ]
    bans, count = infer_caution_bans(rows, fixtures)
    assert count == 5 and bans[0]["club_match"] == 19 and bans[0]["fixture_ids"] == [20]
    rows[-1].update(fixture=20, kickoff_time=fixtures[19]["kickoff_time"])
    assert infer_caution_bans(rows, fixtures)[0] == []


def test_markets_are_exhaustive_devig_and_intensities_fit():
    p = devig([2, 3.5, 4])
    assert np.isclose(p.sum(), 1) and np.all(p > 0)
    expected, _ = score_markets(1.7, 1.1)
    fitted = goal_constraints((1 / expected).tolist())
    assert [fitted["home_goals"], fitted["away_goals"]] == pytest.approx([1.7, 1.1], abs=0.002)
    with pytest.raises(ValueError):
        devig([1, 2])


def test_dc_four_cell_correction_preserves_joint_probability():
    h, a = np.meshgrid(np.arange(30), np.arange(30), indexing="ij")
    actual = np.stack([h, a], axis=-1)
    probabilities = np.exp(log_scores(actual, np.array([1.6, 1.1]), "dixon_coles", -0.1))
    assert probabilities.sum() == pytest.approx(1, abs=1e-10)
    assert np.exp(log_scores(np.array([0, 0]), [1.6, 1.1], "gamma_poisson", 10)) > 0


def test_objectives_honor_ev_sacrifice_and_fractional_cvar():
    rows = [
        {"plan_id": "a", "expected_net_points": 60, "robust_joint_win": 0.6},
        {"plan_id": "b", "expected_net_points": 58, "robust_joint_win": 0.8},
    ]
    assert select_policy(rows, "beat_both", 1)["selected_plan_id"] == "a"
    assert select_policy(rows, "beat_both", 2)["selected_plan_id"] == "b"
    assert cvar([0, 10, 20], 0.5) == pytest.approx(10 / 3)


def test_public_transport_records_measured_bytes_and_retry_date(tmp_path):
    import asyncio

    store = Store(tmp_path)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": 1}))
        ) as client:
            return await PublicTransport(store).get(client, "https://example.test/feed", "test-feed")

    assert asyncio.run(run()) == {"ok": 1}
    record = store.get("source_request")
    assert record["bytes"] > 0 and record["status"] == 200 and record["latency_seconds"] >= 0
    assert retry_seconds("120", 0) == 120
    assert retry_seconds("Tue, 01 Jan 2019 00:00:00 GMT", 0) == 0


def test_weekly_replay_features_cannot_read_target_outcomes(market):
    from copy import deepcopy

    from fpl_intelligence.replay import deadline_rows

    bootstrap = {
        "teams": [{"id": i, "name": f"Club {i}"} for i in range(1, 11)],
        "elements": [
            {
                "id": p.id,
                "code": p.code,
                "first_name": "Player",
                "second_name": str(p.id),
                "web_name": p.web_name,
                "team": p.team,
                "element_type": p.position,
                "now_cost": p.price,
                "can_select": True,
                "status": "a",
                "chance_of_playing_next_round": None,
            }
            for p in market.values()
        ],
    }
    schedule = [{"id": 9, "event": 9, "team_h": 1, "team_a": 2, "kickoff_time": "2026-10-31T15:00:00Z"}]
    previous = {
        "season": "2026/27",
        "round": 8,
        "fixture": 8,
        "element": 1,
        "code": 1,
        "position": 1,
        "team": "Club 1",
        "opponent_name": "Club 2",
        "was_home": True,
        "team_h_score": 1,
        "team_a_score": 0,
        "minutes": 90,
        "starts": 1,
        "total_points": 6,
        "kickoff_time": "2026-10-25T15:00:00Z",
    }
    target = {
        **previous,
        "round": 9,
        "fixture": 9,
        "kickoff_time": "2026-10-31T15:00:00Z",
        "total_points": 100,
    }
    _, one = deadline_rows([previous, target], bootstrap, schedule, "2026/27", 9, "2026-10-31T11:00:00Z")
    changed = deepcopy(target)
    changed.update(minutes=0, total_points=-30, expected_goals=50, goals_scored=40)
    _, two = deadline_rows([previous, changed], bootstrap, schedule, "2026/27", 9, "2026-10-31T11:00:00Z")
    assert one == two


def test_chunked_worlds_restore_with_hash_checks(tmp_path):
    from fpl_intelligence.worlds import copy_worlds, load_worlds, save_worlds

    one = Store(tmp_path / "one")
    two = Store(tmp_path / "two")
    array = np.zeros((26000, 1000), dtype=np.int16)
    reference = save_worlds(one, "test", {"points_9": array, "minutes_10": array})
    assert reference.endswith(".json")
    frozen = {"scenario_path": reference}
    copy_worlds(one, frozen, two.root)
    with load_worlds(two, frozen) as samples:
        assert samples["points_9"].shape == (26000, 1000)
        assert samples["minutes_10"].sum() == 0
    import json

    manifest = json.loads(one.artifact_path(reference).read_text())
    chunk = two.artifact_path(manifest["arrays"]["points_9"]["path"])
    chunk.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        with load_worlds(two, frozen) as samples:
            samples["points_9"]


def test_long_retry_after_defers_without_hammering_source(tmp_path):
    import asyncio

    store = Store(tmp_path)
    transport = PublicTransport(store)
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(429, headers={"Retry-After": "120"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError, match="deferred"):
                await transport.get(client, "https://example.test/feed", "limited")
            with pytest.raises(ValueError, match="circuit open"):
                await transport.get(client, "https://example.test/feed", "limited")

    asyncio.run(run())
    assert len(calls) == 1 and store.get("source_request")["status"] == 429


def test_shot_refresh_never_erases_last_good_archive(tmp_path, monkeypatch):
    import asyncio

    from fpl_intelligence.shot_data import collect_shots

    store = Store(tmp_path)
    store.put("shot_archive", {"matched": 1, "records": [{"sentinel": "retained"}]})
    monkeypatch.setattr("fpl_intelligence.shot_data.canonical_rows", lambda *_: ([], {}, {}))
    monkeypatch.setattr("fpl_intelligence.shot_data.load_market", lambda *_: ({}, {"season": "2026/27"}))

    async def unavailable(*args, **kwargs):
        raise ValueError("Source returned HTML")

    monkeypatch.setattr(PublicTransport, "get", unavailable)
    result = asyncio.run(collect_shots(store, [2026]))
    assert result["records"] == [{"sentinel": "retained"}] and store.get("shot_archive")["matched"] == 1


def test_captain_and_vice_use_joint_appearance_not_marginal_rank():
    from fpl_intelligence.domain import Lineup
    from fpl_intelligence.rules import optimize_captain

    # B is the second-highest mean, but disappears exactly when captain A is
    # absent. C is the better vice because its points cover A's missing world.
    lineup = Lineup(xi=[1, 2, 3], bench=[], captain=1, vice=2)
    points = np.array([[20, 15, 0], [20, 15, 0], [0, 0, 12]])
    minutes = np.where(points > 0, 90, 0)
    best = optimize_captain(lineup, points, minutes, [1, 2, 3])
    assert (best.captain, best.vice) == (1, 3)


def test_refresh_cycle_replans_from_reconciled_next_deadline(tmp_path, market, rules, state, monkeypatch):
    from fpl_intelligence import ingestion, optimization, simulation
    from fpl_intelligence.storage import now
    from fpl_intelligence.workflow import refresh_cycle

    store, _ = account_store(tmp_path, market, rules, state, monkeypatch)
    store.put("ingestion", {"captured_at": now(), "histories_requested": True})
    model_id = store.put(
        "decision_model", {"promoted": True, "predictions": [{"season": "2026/27", "event": 9}]}
    )
    monkeypatch.setattr("fpl_intelligence.role_minutes.ensure", lambda saved, model: model)
    calls = []

    async def ingest(self, *, histories=True, entry=None):
        calls.append((histories, entry))
        return {}

    def forecast(saved, start, end, worlds):
        assert start == 10
        identity = saved.put(
            "forecast",
            {
                "start_event": start,
                "end_event": end,
                "market_id": saved.get("market")["id"],
                "decision_model_id": model_id,
            },
        )
        return saved.get("forecast", identity)

    def optimize(saved, state_id, forecast_id, end, seconds):
        current = saved.get("state", state_id)
        assert current["event"] == 10 and current["bank"] == 40
        assert {r["player_id"] for r in current["lots"]} == {*range(2, 16), 16}
        identity = saved.put(
            "plan", {"state_id": state_id, "forecast_id": forecast_id, "market_id": saved.get("market")["id"]}
        )
        return saved.get("plan", identity)

    monkeypatch.setattr(ingestion.Official, "ingest", ingest)
    monkeypatch.setattr(simulation, "forecast", forecast)
    monkeypatch.setattr(optimization, "optimize", optimize)
    monkeypatch.setattr(
        "fpl_intelligence.objectives.choose_recommendation", lambda s, plans, f: (plans[0], {})
    )
    result = refresh_cycle(store)
    assert calls == [(False, 123)]
    assert result["account_changed"] and result["status"] == "Refreshed and recomputed"
    assert store.get("plan", store.get("selected_plan")["plan_id"])["state_id"] == store.get("state")["id"]


def test_plan_endpoint_rejects_retired_model_even_when_state_and_market_match(tmp_path):
    from fastapi.testclient import TestClient

    from fpl_intelligence.api import create_app

    app = create_app(tmp_path)
    store = app.state.store
    state_id = store.put("state", {"event": 10})
    market_id = store.put("market", {"events": [{"id": 10, "deadline_time": "2099-01-01T00:00:00Z"}]})
    model_id = store.put("decision_model", {"marker": "validated original"})
    forecast_id = store.put("forecast", {"decision_model_id": model_id})
    plan_id = store.put(
        "plan",
        {
            "state_id": state_id,
            "market_id": market_id,
            "forecast_id": forecast_id,
            "start_event": 10,
            "execution_ready": True,
        },
    )
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get(f"/api/plans/{plan_id}").json()["stale"] is False
        store.put("decision_model", {"marker": "new validated fit"})
        plan = client.get(f"/api/plans/{plan_id}").json()
        assert plan["stale"] and plan["execution_ready"] is False


def test_frozen_enrichment_parameters_survive_new_temporal_experiment(tmp_path):
    from fpl_intelligence.enrichment import EnrichmentIndex

    store = Store(tmp_path)
    frozen = store.put("temporal_model", {"adopted": {"decay": 0.94, "finishing_strength": None}})
    store.put("temporal_model", {"adopted": {"decay": 1.0, "finishing_strength": 30}})
    old_features = EnrichmentIndex(store, temporal_model_id=frozen)
    assert old_features.params == {"decay": 0.94, "finishing_strength": None}
    assert old_features.lineage["temporal_model"] == frozen
    assert EnrichmentIndex(store).params == {"decay": 1.0, "finishing_strength": 30}


def test_duplicate_refresh_reuses_pending_job_without_spending_again(tmp_path, monkeypatch):
    from threading import Event

    from fastapi.testclient import TestClient

    from fpl_intelligence.api import create_app

    entered, finish = Event(), Event()
    calls = []

    def work(store):
        calls.append(True)
        entered.set()
        assert finish.wait(10)
        return {"status": "completed once"}

    monkeypatch.setattr("fpl_intelligence.workflow.refresh_cycle", work)
    app = create_app(tmp_path)
    store = app.state.store
    store.put("state", {"event": 10})
    store.put("market", {"events": [{"id": 10, "deadline_time": "2099-01-01T00:00:00Z"}]})
    store.put("decision_model", {"marker": "validated fit"})
    try:
        with TestClient(app, base_url="http://localhost") as client:
            first = client.post("/api/refresh").json()
            assert entered.wait(5)
            second = client.post("/api/refresh").json()
            assert second["job_id"] == first["job_id"] and second["reused"]
            assert calls == [True]
    finally:
        finish.set()


def test_structural_worlds_conserve_goals_channels_and_on_pitch_scoring(market, rules):
    from fpl_intelligence.features import profile
    from fpl_intelligence.simulation import fixture_worlds

    players = {p: market[p].model_copy(update={"team": 1 if p <= 15 else 2}) for p in range(1, 31)}
    profiles = {p: profile(v.position, []) for p, v in players.items()}
    profiles[1]["availability_override"] = 0
    rates = {t: {"mean": 1.4, "attack": 1, "defence": 1} for t in (1, 2)}
    ids, points, minutes, starts, components, audit = fixture_worlds(
        np.random.default_rng(12),
        {"id": 1, "team_h": 1, "team_a": 2},
        players,
        profiles,
        rates,
        rules,
        128,
        0,
        {t: np.zeros((128, 15)) for t in (1, 2)},
    )
    assert audit["scored_goals"] + audit["own_goals"] == audit["team_goals"]
    assert sum(audit["goal_channels"].values()) == audit["team_goals"]
    assert all(q["short_lineup_worlds"] == 0 for q in audit["lineups"])
    assert np.all(points[minutes == 0] == 0)
    assert np.all(minutes[:, ids.index(1)] == 0)
    assert np.array_equal(points, sum(components.values()))
    assert np.all((minutes >= 0) & (minutes <= 90))


def test_public_transport_does_not_decode_compressed_article_twice(tmp_path):
    import asyncio
    import gzip

    text = "<html><p>Confirmed football report.</p></html>"

    async def handler(request):
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-type": "text/html"},
            content=gzip.compress(text.encode()),
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await PublicTransport(Store(tmp_path)).get(
                client,
                "https://example.test/article",
                "article",
                json_body=False,
                return_response=True,
            )
            assert response.text == text
            assert "content-encoding" not in response.headers

    asyncio.run(run())


def test_random_squad_sampler_handles_a_legal_low_cost_market(market, rules):
    from fpl_intelligence.policy_learning import random_state
    from fpl_intelligence.rules import validate_squad

    players = {p: v.model_copy(update={"price": 40}) for p, v in market.items()}
    state = random_state(players, {p: 1 for p in players}, rules, 10, "2026/27", np.random.default_rng(83))
    validate_squad(state, players, rules)
    assert state.bank == 400 and len(state.lots) == 15


@pytest.mark.parametrize("chip", [None, "bboost", "3xc"])
def test_batched_world_scoring_matches_independent_rule_replay(market, rules, chip):
    from fpl_intelligence.rules import best_lineup, score_gameweek
    from fpl_intelligence.simulation import lineup_worlds

    ids = list(range(1, 16))
    rng = np.random.default_rng(83)
    minutes = rng.choice([0, 25, 90], size=(48, 15))
    points = np.where(minutes > 0, rng.integers(-3, 19, size=(48, 15)), 0)
    lineup = best_lineup(ids, {p: float(points[:, p - 1].mean()) for p in ids}, market, rules)
    expected = [
        score_gameweek(dict(zip(ids, pp)), dict(zip(ids, mm)), lineup, market, rules, chip)
        for pp, mm in zip(points, minutes)
    ]
    assert np.array_equal(lineup_worlds(lineup, points, minutes, ids, market, rules, chip), expected)


def test_aggregated_bench_search_matches_exhaustive_rule_expectation(market, rules):
    from itertools import permutations

    from fpl_intelligence.rules import best_lineup, optimize_bench, scoring_ids

    ids = list(range(1, 16))
    rng = np.random.default_rng(841)
    minutes = rng.choice([0, 90], size=(48, 15))
    points = np.where(minutes > 0, rng.integers(-3, 19, size=(48, 15)), 0)
    lineup = best_lineup(ids, {p: float(points[:, p - 1].mean()) for p in ids}, market, rules)

    def ordinary(candidate):
        return sum(
            sum(pp[p - 1] for p in scoring_ids(candidate, dict(zip(ids, mm > 0)), market, rules))
            for pp, mm in zip(points, minutes)
        )

    maximum = max(
        ordinary(lineup.model_copy(update={"bench": [lineup.bench[0], *order]}))
        for order in permutations(lineup.bench[1:])
    )
    chosen = optimize_bench(lineup, points, minutes, ids, market, rules)
    assert ordinary(chosen) == maximum
