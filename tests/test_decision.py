from pathlib import Path

import httpx
import numpy as np
import pytest

from fpl_intelligence.decision import NAMES, build_dataset, learned_forecast
from fpl_intelligence.storage import Store


def test_deadline_features_carry_stable_identity_without_doubleweek_leakage(tmp_path, monkeypatch):
    import fpl_intelligence.decision as module

    rows = []
    fixtures = {}
    for season, event, fixture, player, points, price in [
        ("2024/25", 38, 1, 1, 5, 45),
        ("2025/26", 1, 2, 101, 9, 99),
        ("2025/26", 1, 3, 101, 3, 98),
        ("2025/26", 2, 4, 101, 2, 97),
    ]:
        rows.append(
            {
                "season": season,
                "round": event,
                "fixture": fixture,
                "element": player,
                "code": 777,
                "position": 2,
                "team": "Alpha",
                "opponent_name": "Beta",
                "minutes": 90,
                "starts": 1,
                "start_observed": True,
                "total_points": points,
                "value": price,
                "was_home": True,
                "kickoff_time": None,
                "expected_goals": 0.1,
                "expected_assists": 0.1,
            }
        )
        fixtures[season, fixture] = {
            "home": "Alpha",
            "away": "Beta",
            "home_goals": 1,
            "away_goals": 0,
            "event": event,
        }
    monkeypatch.setattr(module, "canonical_rows", lambda store: (rows, fixtures, {}))
    _, before = build_dataset(Store(tmp_path))
    first = before.filter(
        (__import__("polars").col("season") == "2025/26") & (__import__("polars").col("event") == 1)
    )
    assert first["price_lag"].to_list() == [45, 45]
    assert first["history_missing"].to_list() == [0, 0]
    for row in rows:
        if row["season"] == "2025/26" and row["round"] == 1:
            row["total_points"] = 100
            row["expected_goals"] = 10
            row["minutes"] = 0
    _, after = build_dataset(Store(tmp_path))
    second = after.filter(
        (__import__("polars").col("season") == "2025/26") & (__import__("polars").col("event") == 1)
    )
    np.testing.assert_array_equal(first.select(NAMES).to_numpy(), second.select(NAMES).to_numpy())
    assert first["target"].to_list() != second["target"].to_list()


def test_optional_provider_failure_is_isolated_and_secret_free(tmp_path, monkeypatch):
    from fpl_intelligence.providers import configure

    monkeypatch.setenv("GEMINI_API_KEY", "private-example-gemini")
    monkeypatch.setenv("GROQ_API_KEY", "private-example-groq")
    monkeypatch.setattr("fpl_intelligence.providers.load_environment", lambda: None)

    def handler(request):
        if request.url.host == "api.groq.com":
            return httpx.Response(401, json={"error": "unauthorized"})
        return httpx.Response(200, json={"models": [{"name": "models/gemini-3.5-flash-lite"}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = configure(Store(tmp_path), True, client)
    assert {r["provider"]: r["status"] for r in result["providers"]} == {
        "gemini": "authenticated",
        "groq": "unavailable",
    }
    assert result["numerical_planner_available"]
    assert "private-example" not in str(result)
    assert Store(tmp_path).get("provider_selection")["provider"] == "gemini"


def test_uncalibrated_legacy_forecast_requires_explicit_research_mode(tmp_path):
    from fpl_intelligence.simulation import forecast

    with pytest.raises(ValueError, match="research-only"):
        forecast(Store(tmp_path))


def test_live_means_components_and_optimizer_worlds_are_same_distribution(
    tmp_path, market, rules, monkeypatch
):
    import fpl_intelligence.decision as module

    store = Store(tmp_path)
    curves = {str(k): {"x": [0, 1], "y": [0, 1]} for k in range(4)} | {"start": {"x": [0, 1], "y": [0, 1]}}
    spec = {
        "promoted": True,
        "directory": "test",
        "model_hashes": {},
        "curves": curves,
        "dependence": {"defence": 0.2, "attack": 0.1},
        "residuals": {f"{p}:{k}": [0, 0] for p in range(1, 5) for k in range(1, 4)},
        "minute_quantiles": {"0": [0, 0], "1": [24, 24], "2": [74, 74], "3": [90, 90]},
        "feature_id": "features",
        "seasons": ["2024/25", "2025/26"],
    }
    store.put("decision_model", spec)
    frozen = {
        "id": "market",
        "season": "2026/27",
        "teams": [],
        "events": [{"id": 9, "is_next": True}],
        "fixtures": [],
        "source_hash": "bootstrap",
        "fixture_hash": "fixtures",
    }
    monkeypatch.setattr(module, "load_market", lambda s: (market, frozen))
    monkeypatch.setattr(store, "latest", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr("fpl_intelligence.rules.Rules.from_bootstrap", lambda *a: rules)

    class Minutes:
        def predict(self, x, **kwargs):
            return np.tile([0, 0, 0, 1.0], (len(x), 1))

    class Starter:
        def predict(self, x, **kwargs):
            return np.ones(len(x))

    class Points:
        def predict(self, x, **kwargs):
            return np.where(x[:, -1] == 1, 1, np.where(x[:, 0] == 4, 7, 3))

    monkeypatch.setattr(
        module,
        "load_bundle",
        lambda *a: {"minutes": Minutes(), "starter": Starter(), "points": Points(), "curves": curves},
    )
    rows = [
        {
            "player_id": p,
            "fixture_id": 1,
            "event": 9,
            "team": player.team,
            "opponent": 99,
            "x": [player.position] + [0] * (len(NAMES) - 1),
            "profile": {"history_count": 20, "action_coverage": 1},
            "availability": 0 if p == 1 else 1,
            "minutes_cap": 45 if p == 2 else 90,
            "dc_probability": [0, 0, 0, 0],
        }
        for p, player in market.items()
    ]
    monkeypatch.setattr(module, "future_inputs", lambda *a: (rows, []))
    f = learned_forecast(store, start_event=9, end_event=9, worlds=500)
    with np.load(store.artifact_path(f["scenario_path"]), allow_pickle=False) as worlds:
        points = worlds["points_9"]
        minutes = worlds["minutes_9"]
        for i, row in enumerate(f["players"]):
            assert row["mean"] == points[:, i].mean()
            assert sum(row["components"].values()) == pytest.approx(row["mean"], abs=1e-4)
            assert np.all(points[minutes[:, i] == 0, i] == 0)
        assert np.all(minutes[:, 1] <= 45)
        assert np.all(points[:, 0] == 0)
        forwards = [i for i, p in enumerate(f["player_ids"]) if market[p].position == 4]
        assert np.all(points[:, forwards] == 7)
    assert Path(f["scenario_path"]).is_file()


def test_greedy_transfer_repairs_or_preserves_grandfathered_club_limit(state, market, rules):
    from fpl_intelligence.optimization import greedy_bundle
    from fpl_intelligence.rules import transition

    # Four owned players acquired an extra common club after real club moves.
    for p in (1, 3, 8, 13):
        market[p] = market[p].model_copy(update={"team": 99})
    state = state.model_copy(update={"grandfathered_clubs": [99]})
    means = {p: 3.0 for p in market}
    means[31] = 100
    action = greedy_bundle(state, market, rules, means)
    from fpl_intelligence.rules import best_lineup

    owned = {lot.player_id for lot in state.lots} - set(action.sell) | set(action.buy)
    action.lineup = best_lineup(sorted(owned), means, market, rules)
    following, _ = transition(state, action, market, rules)
    assert not action.buy or sum(market[lot.player_id].team == 99 for lot in following.lots) <= 3


def test_update_keeps_existing_account_and_credentials(tmp_path):
    import importlib.util
    import json
    import sqlite3

    spec = importlib.util.spec_from_file_location(
        "fpl_update", Path(__file__).resolve().parents[1] / "scripts/install_update.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install = module.install

    source = tmp_path / "new"
    target = tmp_path / "existing"
    source.mkdir()
    target.mkdir()
    (target / "pyproject.toml").write_text('name = "fpl-intelligence"')
    (target / ".env").write_text("EXISTING_PRIVATE_VALUE=keep")
    (target / "data").mkdir()
    with sqlite3.connect(target / "data/state.sqlite") as db:
        db.execute("CREATE TABLE account(bank INTEGER)")
        db.execute("INSERT INTO account VALUES(3)")
    (source / "app-files.json").write_text(json.dumps(["app.py"]))
    (source / "app.py").write_text("new implementation")
    (target / "app.py").write_text("old implementation")
    result = install(source, target)
    assert (target / "app.py").read_text() == "new implementation"
    assert (target / ".env").read_text() == "EXISTING_PRIVATE_VALUE=keep"
    with sqlite3.connect(target / "data/state.sqlite") as db:
        assert db.execute("SELECT bank FROM account").fetchone()[0] == 3
    assert (Path(result["backup"]) / "app/app.py").read_text() == "old implementation"
    (source / "app-files.json").write_text(json.dumps([".env"]))
    with pytest.raises(ValueError, match="private or unsafe"):
        install(source, target)


def test_no_immediate_hit_budget_is_enforced_by_solver_and_ledger(
    tmp_path, market, rules, state, monkeypatch
):
    import fpl_intelligence.optimization as optimizer

    state = state.model_copy(update={"free_transfers": 1})
    store = Store(tmp_path)
    state_id = store.put("state", state.model_dump())
    frozen = {
        "id": "market",
        "season": state.season,
        "source_hash": "bootstrap",
        "teams": [{"id": team} for team in sorted({p.team for p in market.values()})],
    }
    monkeypatch.setattr(optimizer, "load_market", lambda *a: (market, frozen))
    monkeypatch.setattr("fpl_intelligence.simulation.load_market", lambda *a: (market, frozen))
    monkeypatch.setattr("fpl_intelligence.rules.Rules.from_bootstrap", lambda *a: rules)
    monkeypatch.setattr(store, "latest", lambda *a, **kw: ({}, {}))
    ids = sorted(market)
    means = {p: 1.0 if p <= 15 else 20.0 for p in ids}
    path = tmp_path / "scenarios" / "test.npz"
    path.parent.mkdir()
    np.savez_compressed(
        path,
        points_9=np.tile([means[p] for p in ids], (100, 1)).astype(np.int16),
        minutes_9=np.full((100, len(ids)), 90, dtype=np.int16),
    )
    decision_id = store.put("decision_model", {"promoted": True})
    forecast_id = store.put(
        "forecast",
        {
            "market_id": "market",
            "rules_hash": rules.hash,
            "start_event": 9,
            "end_event": 9,
            "worlds": 100,
            "seed": 11,
            "scenario_path": str(path),
            "player_ids": ids,
            "players": [{"event": 9, "player_id": p, "mean": means[p]} for p in ids],
            "input_hashes": [],
            "model_version": "learned-points-v1",
            "decision_model_id": decision_id,
        },
    )
    result = optimizer.optimize(
        store, state_id, forecast_id, 9, 3, chips={}, automatic_chips=False, first_week_hit_budget=0
    )
    assert result["ledger"][0]["hit"] == 0
    assert len(result["ledger"][0]["buy"]) == 1
    assert result["validation"]["legal"] and result["validation"]["affordable"]

    store.put("decision_model", {"promoted": True, "retrained": True})
    with pytest.raises(ValueError, match="current model"):
        optimizer.optimize(store, state_id, forecast_id, 9, 3, chips={}, automatic_chips=False)
