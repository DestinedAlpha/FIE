import numpy as np
import pytest

from fpl_intelligence.manager_behavior import behavior_rows
from fpl_intelligence.policy_learning import random_state
from fpl_intelligence.rules import validate_squad
from fpl_intelligence.storage import Store


def test_relocated_nested_model_artifacts_reject_parent_traversal(tmp_path):
    store = Store(tmp_path)
    assert (
        store.artifact_path("/old/computer/data/models/fusion-123/heldout.parquet")
        == tmp_path / "models/fusion-123/heldout.parquet"
    )
    with pytest.raises(ValueError):
        store.artifact_path("data/models/../../.env")


def test_manager_training_features_ignore_future_transfers_and_initial_ft():
    history = {
        "current": [
            {
                "event": g,
                "event_transfers": 0,
                "event_transfers_cost": 0,
                "bank": 3,
                "value": 1000,
                "points": 60,
            }
            for g in range(1, 5)
        ],
        "chips": [],
    }
    manager = {
        "entry": 1,
        "history": history,
        "transfers": [
            {"event": 2, "element_in": 2, "element_out": 1, "element_in_cost": 60, "element_out_cost": 55}
        ],
        "picks": {str(g): {"picks": [{"element": 2, "is_captain": True}]} for g in range(1, 5)},
    }
    packets = {
        (g, 2): {"status": "i", "chance_of_playing_next_round": 0, "now_cost": 60, "team": 1}
        for g in range(1, 6)
    }
    before = behavior_rows({"members": [manager]}, packets)
    assert before[0]["event"] == 2 and before[0]["x"][1] == 1
    assert before[1]["x"][1] == 2
    assert before[1]["x"][10:12] == [1, 1]  # Explicit zero chance is observed injury, not missing.
    manager["transfers"].append(
        {"event": 4, "element_in": 3, "element_out": 2, "element_in_cost": 150, "element_out_cost": 60}
    )
    after = behavior_rows({"members": [manager]}, packets)
    assert before[1]["x"] == after[1]["x"]  # Target GW3 cannot read a GW4 transfer.


def test_random_counterfactual_squads_are_legal_and_reproducible(market, rules):
    players = {p: player.model_copy(update={"price": 65}) for p, player in market.items()}
    means = {p: 2 for p in players}
    one = random_state(players, means, rules, 9, "2026/27", np.random.default_rng(83))
    two = random_state(players, means, rules, 9, "2026/27", np.random.default_rng(83))
    validate_squad(one, players, rules)
    assert one.bank == 25 and 1 <= one.free_transfers <= rules.ft_cap
    assert [lot.player_id for lot in one.lots] == [lot.player_id for lot in two.lots]


def test_advanced_stats_use_shared_predeadline_history_for_a_double(tmp_path, monkeypatch):
    import fpl_intelligence.horizon as module

    raw = []
    metadata = []
    for event, fixture, influence in [(1, 1, 10), (2, 2, 20), (2, 3, 30), (3, 4, 40)]:
        raw.append(
            {
                "season": "2025/26",
                "round": event,
                "fixture": fixture,
                "element": 1,
                "code": 100,
                "minutes": 90,
                "influence": influence,
            }
        )
        metadata.append({"season": "2025/26", "event": event, "player_id": 1, "fixture_id": fixture})
    monkeypatch.setattr(module, "canonical_rows", lambda store: (raw, {}, {}))
    before = module.advanced_history(Store(tmp_path), metadata)
    raw[1]["influence"] = 2000
    raw[2]["influence"] = 3000
    after = module.advanced_history(Store(tmp_path), metadata)
    np.testing.assert_array_equal(before[1:3], after[1:3])
    assert not np.array_equal(before[3], after[3])


def test_hit_comparison_reuses_stronger_known_feasible_no_hit_policy(tmp_path, monkeypatch):
    import fpl_intelligence.chips as module

    store = Store(tmp_path)
    forecast = store.put("forecast", {})

    def save(mean, hit, forecast_id=forecast):
        body = {
            "state_id": "same-state",
            "forecast_id": forecast_id,
            "end_event": 19,
            "ledger": [{"event": 9, "hit": hit, "chip": None}, {"event": 19, "hit": 0, "chip": None}],
            "simulation": {"mean": mean},
            "solver": {"gap": 0.01},
            "validation": {"legal": True},
        }
        identity = store.put("plan", body)
        return store.get("plan", identity)

    base = save(110, 8)
    weak = save(95, 0)
    strong = save(109.8, 0)
    save(999, 0, "stale-forecast")
    monkeypatch.setattr(module, "optimize", lambda *args, **kwargs: weak)
    monkeypatch.setattr(
        module, "score_plan", lambda store, plan, forecast: np.full(100, plan["simulation"]["mean"])
    )
    result = module.compare_immediate_hit(store, base["id"])
    assert result["alternative_plan_id"] == strong["id"]
    assert result["net_mean_advantage"] == pytest.approx(0.2)
    assert result["feasible_counterfactuals_compared"] == 2


def test_imported_older_object_does_not_replace_newer_local_result(tmp_path):
    store = Store(tmp_path)
    fresh = store.put("fusion_model", {"version": "newer-local"})
    old = store.put("fusion_model", {"version": "older-package"})
    with store.db() as db:
        db.execute(
            "UPDATE objects SET created_at=? WHERE kind=? AND id=?",
            ("2020-01-01T00:00:00+00:00", "fusion_model", old),
        )
    assert store.get("fusion_model")["id"] == fresh
    assert store.get("fusion_model", old)["version"] == "older-package"


def test_manager_free_hit_features_restore_underlying_squad_and_bank():
    history = {
        "current": [
            {
                "event": g,
                "event_transfers": 0,
                "event_transfers_cost": 0,
                "bank": g,
                "value": 1000,
                "points": 50,
            }
            for g in range(1, 5)
        ],
        "chips": [{"name": "freehit", "event": 4}],
    }
    history["current"][-1]["bank"] = 90
    manager = {
        "entry": 1,
        "history": history,
        "transfers": [],
        "picks": {
            str(g): {"picks": [{"element": 2 if g < 4 else 3, "is_captain": True}]} for g in range(1, 5)
        },
    }
    packets = {
        (5, 2): {"status": "i", "chance_of_playing_next_round": 0, "now_cost": 60, "team": 1},
        (5, 3): {"status": "a", "chance_of_playing_next_round": 100, "now_cost": 150, "team": 2},
    }
    future = behavior_rows({"members": [manager]}, packets)[-1]
    assert future["event"] == 5 and future["x"][2] == 0.3
    assert future["x"][10:12] == [1, 1]


def test_rival_free_hit_does_not_treat_temporary_picks_as_permanent(tmp_path, market, rules, monkeypatch):
    import fpl_intelligence.rivals as module
    from fpl_intelligence.storage import now

    store = Store(tmp_path)
    history = {
        "current": [{"event": g, "event_transfers": 0} for g in range(1, 9)],
        "chips": [{"name": "freehit", "event": 8}],
    }
    underlying = {"picks": [{"element": p} for p in range(1, 16)], "entry_history": {"bank": 30}}
    temporary = {"picks": [{"element": p} for p in range(16, 31)], "entry_history": {"bank": 99}}

    def latest(source):
        if source == "entry_history_9":
            return history, {}
        if source == "picks_9_7":
            return underlying, {}
        if source == "picks_9_8":
            return temporary, {}
        return {}, {}

    monkeypatch.setattr(store, "latest", latest)
    monkeypatch.setattr(module, "load_market", lambda *args: (market, {}))
    monkeypatch.setattr("fpl_intelligence.rules.Rules.from_bootstrap", lambda *args: rules)
    monkeypatch.setattr(module, "score_plan", lambda *args: np.zeros(100))
    forecast = {
        "season": "2026/27",
        "start_event": 9,
        "players": [{"event": 9, "player_id": p, "mean": 1} for p in market],
    }
    plan = module.rival_policy(
        store,
        {"entry": 9, "team": "Example", "name": "Example", "points": 0},
        {"released_event": 8, "captured_at": now()},
        forecast,
        9,
        "hold",
    )
    assert plan["ledger"][0]["bank_before"] == 30
    assert set(plan["ledger"][0]["permanent_squad"]) == set(range(1, 16))
