"""Independent financial, temporal and structural boundaries of the full strategy stack."""

from itertools import product

import numpy as np
import pytest

from fpl_intelligence.decision_runtime import commit_before_deadline, continuation_terms
from fpl_intelligence.decision_training import ACTION_FEATURES, partition
from fpl_intelligence.domain import Action
from fpl_intelligence.market_model import market_features
from fpl_intelligence.rules import best_lineup, transition
from fpl_intelligence.storage import Store
from fpl_intelligence.strategy import allowed_chips, solve_action


def test_full_market_one_transfer_matches_exhaustive_oracle(market, rules, state):
    state = state.model_copy(update={"free_transfers": 1})
    means = {p: float((p * 17) % 19) / 2 for p in market}
    choices = []
    owned = {x.player_id for x in state.lots}
    for old, new in product(owned, market.keys() - owned):
        if market[old].position != market[new].position:
            continue
        squad = sorted(owned - {old} | {new})
        action = Action(sell=[old], buy=[new], lineup=best_lineup(squad, means, market, rules))
        try:
            _, row = transition(state, action, market, rules)
        except ValueError:
            continue
        choices.append(sum(means[p] for p in action.lineup.xi) + means[action.lineup.captain] - row["hit"])
    hold = best_lineup(sorted(owned), means, market, rules)
    choices.append(sum(means[p] for p in hold.xi) + means[hold.captain])
    result = solve_action(state, market, rules, means, max_transfers=1, seconds=3, relative_gap=0)
    assert result.utility == pytest.approx(max(choices))
    assert result.solver["gap"] == pytest.approx(0)


def test_early_transfers_and_late_recourse_charge_only_incremental_hits(market, rules, state):
    state = state.model_copy(update={"free_transfers": 1})
    means = {p: 1.0 for p in market}
    first = Action(
        sell=[1, 3],
        buy=[16, 18],
        lineup=best_lineup(
            [p.player_id for p in state.lots if p.player_id not in (1, 3)] + [16, 18], means, market, rules
        ),
    )
    committed = commit_before_deadline(state, first, market, rules)
    assert committed.event == state.event and committed.hit_paid == 4 and committed.transfers_made == 2
    following = Action(
        sell=[4],
        buy=[19],
        lineup=best_lineup(
            [p.player_id for p in committed.lots if p.player_id != 4] + [19], means, market, rules
        ),
    )
    _, row = transition(committed, following, market, rules)
    assert row["hit"] == 4 and row["event_total_hit"] == 8
    direct = Action(sell=[1, 3, 4], buy=[16, 18, 19], lineup=following.lineup)
    _, one = transition(state, direct, market, rules)
    assert row["bank_after"] == one["bank_after"] and row["permanent_squad"] == one["permanent_squad"]


def test_freehit_recourse_restores_original_account_after_early_transfers(market, rules, state):
    means = {p: float(p % 11) for p in market}
    first = Action(
        sell=[1, 3],
        buy=[16, 18],
        lineup=best_lineup(
            [p.player_id for p in state.lots if p.player_id not in (1, 3)] + [16, 18], means, market, rules
        ),
    )
    committed = commit_before_deadline(state, first, market, rules)
    d = solve_action(committed, market, rules, means, chip="freehit", seconds=3)
    assert d.state.bank == state.bank
    assert {x.player_id for x in d.state.lots} == {x.player_id for x in state.lots}
    assert d.ledger["event_total_hit"] == 0


def test_planning_states_do_not_replace_actual_account(tmp_path):
    store = Store(tmp_path)
    actual = store.put("state", {"bank": 3})
    branch = store.put("planning_state", {"bank": 99})
    assert store.get("state")["id"] == actual
    assert store.get("state", branch)["bank"] == 99
    assert store.list("state", 10) == [{"id": actual, "created_at": store.list("state", 1)[0]["created_at"]}]


def test_purged_policy_split_never_crosses_a_target_window():
    rows = [{"group": f"2025/26-GW{g:02d}", "label_end": f"2025/26-GW{g + 3:02d}"} for g in range(2, 36)]
    train, cal, test = partition(rows)
    assert max(r["label_end"] for r in train) < min(r["group"] for r in cal)
    assert max(r["label_end"] for r in cal) < min(r["group"] for r in test)
    assert not {"actual", "return", "oracle", "minutes", "outcome"} & set(ACTION_FEATURES)


def test_market_predictors_preserve_zero_chance_and_ignore_future_outcomes():
    p = {"now_cost": 60, "element_type": 2, "chance_of_playing_next_round": 0, "selected_by_percent": "12.5"}
    x = market_features(p, 9)
    p.update(total_points=900, goals_scored=40, actual_next_price=200)
    assert market_features(p, 9) == x and x[6:8] == [0, 1]


def test_real_lineups_keep_eleven_players_and_at_most_five_substitutes(market):
    from fpl_intelligence.features import profile
    from fpl_intelligence.simulation import sample_lineup

    members = [market[p].model_copy(update={"team": 1}) for p in range(1, 26)]
    profiles = {p.id: profile(p.position, [{"minutes": 75, "starts": 1}] * 8) for p in members}
    starts, ends, audit = sample_lineup(
        np.random.default_rng(193), members, profiles, 256, 0, np.zeros((256, len(members)))
    )
    assert np.all(((starts == 0) & (ends > 0)).sum(1) == 11)
    assert np.all((starts > 0).sum(1) <= 5)
    for minute in (1, 30, 60, 89):
        assert np.all(((starts < minute) & (ends >= minute)).sum(1) == 11)
    assert audit["short_lineup_worlds"] == 0


def test_learned_terminal_rewards_bank_ft_and_preserved_chips(tmp_path, market, rules, state):
    store = Store(tmp_path)
    store.put("state_value_model", {"promoted": True, "coefficients": [0.2, 1, 2, 3, 4, 5, 6]})
    means = {(g, p): 1.0 for g in range(9, 39) for p in market}
    terms = continuation_terms(store, 12, means, market, rules, state)
    assert terms["model_id"] and terms["bank"] > 0 and terms["ft"] > 0
    assert terms["chip_use"][9, "freehit"] < 0
    assert terms["players"][1] == pytest.approx(0.6)
    # A chip expiring before the continuing window has no opportunity cost.
    assert not continuation_terms(store, 19, means, market, rules, state)["chip_use"]


def test_an_activated_chip_is_preserved_and_cleared_after_deadline(market, rules, state):
    active = state.model_copy(update={"active_chip": "wildcard", "free_transfers": 1})
    means = {p: float(p % 9) for p in market}
    d = solve_action(active, market, rules, means, seconds=3)
    assert allowed_chips(active, rules) == ["wildcard"]
    assert d.ledger["chip"] == "wildcard" and d.ledger["event_total_hit"] == 0
    assert d.state.active_chip is None and d.state.used_chips[-1].chip == "wildcard"
    with pytest.raises(ValueError, match="activated chip"):
        transition(active, Action(chip="3xc"), market, rules)


def test_quarantined_evidence_never_reappears_in_context_branches(tmp_path):
    from datetime import datetime, timedelta, timezone

    from fpl_intelligence.context import evidence_branches

    s = Store(tmp_path)
    at = datetime.now(timezone.utc)
    claim = {
        "player_id": 1,
        "known_at": at.isoformat(),
        "expires_at": (at + timedelta(days=1)).isoformat(),
        "claim_type": "unavailable",
        "affected_events": [9],
        "statement": "Unavailable",
        "status": "reported",
        "source_url": "https://www.arsenal.com/news",
        "origin_hash": "one",
    }
    identity = s.put("claim", claim)
    s.put("claim_rejection", {"claim_id": identity, "reason": "Wrong player"})
    assert evidence_branches(s, 9) == []


def test_training_replay_counts_future_bench_boost_and_triple_captain(market, rules, state):
    from fpl_intelligence.decision_training import rescore_records
    from fpl_intelligence.rules import score_gameweek

    points = {p: 2 for p in market}
    minutes = {p: 90 for p in market}
    lineup = best_lineup([lot.player_id for lot in state.lots], points, market, rules)
    rows = [
        {"event": event, "chip": chip, "hit": 0, "lineup": lineup.model_dump()}
        for event, chip in [(2, None), (3, "bboost"), (4, "3xc")]
    ]
    actions = [{"group": "2026/27-GW02", "ledger": rows}]
    states = [{"target": -999}]
    keys = [("2026/27", event) for event in (2, 3, 4)]
    rescore_records(
        actions,
        states,
        dict.fromkeys(keys, market),
        dict.fromkeys(keys, points),
        dict.fromkeys(keys, minutes),
        {"2026/27": rules},
    )
    expected = [score_gameweek(points, minutes, lineup, market, rules, row["chip"]) for row in rows]
    assert actions[0]["return"] == sum(expected)
    assert states[0]["target"] == expected[1] + expected[2]
    assert expected == [24, 32, 26]


def test_equivalent_freehit_actions_reuse_a_validated_continuation(
    tmp_path, monkeypatch, market, rules, state
):
    import fpl_intelligence.ingestion as ingestion
    import fpl_intelligence.optimization as optimization
    import fpl_intelligence.rules as rule_module
    import fpl_intelligence.simulation as simulation
    from fpl_intelligence.decision_runtime import conditional_plan
    from fpl_intelligence.rules import Rules

    store = Store(tmp_path)
    sid = store.put("state", state.model_dump())
    means = {p: 2.0 for p in market}
    first = Action(chip="freehit", lineup=best_lineup(list(range(1, 16)), means, market, rules))
    following, first_row = transition(state, first, market, rules)
    later = Action(lineup=best_lineup(list(range(1, 16)), means, market, rules))
    final, later_row = transition(following, later, market, rules)
    store.put("plan", {"state_id": sid, "market_id": "market", "ledger": [first_row, later_row]})
    path = tmp_path / "scenarios" / "worlds.npz"
    path.parent.mkdir()
    np.savez(path, points_9=np.ones((4, 45)), minutes_9=np.ones((4, 45)) * 90)
    fid = store.put(
        "forecast",
        {
            "scenario_path": str(path),
            "player_ids": list(market),
            "players": [{"event": 9, "player_id": p, "mean": 2} for p in market],
            "input_hashes": [],
        },
    )
    monkeypatch.setattr(
        ingestion, "load_market", lambda *args: (market, {"id": "market", "source_hash": "hash"})
    )
    monkeypatch.setattr(store, "latest", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(Rules, "from_bootstrap", classmethod(lambda cls, bootstrap: rules))
    monkeypatch.setattr(rule_module, "optimize_bench", lambda lineup, *args: lineup)
    monkeypatch.setattr(simulation, "score_plan", lambda *args: np.zeros(4))

    def check_seed(saved_store, identity, forecast_id, end, *args, **kwargs):
        seed = saved_store.get("warm_start")
        assert seed["state_id"] == identity and seed["ledger"] == [later_row]
        return {"id": "tail", "ledger": [later_row], "solver": {"gap": 0}, "final_state": final.model_dump()}

    monkeypatch.setattr(optimization, "optimize", check_seed)
    alternative = first.model_copy(deep=True)
    alternative.lineup.bench[1:] = reversed(alternative.lineup.bench[1:])
    plan = conditional_plan(store, sid, fid, 10, alternative)
    assert len(plan["ledger"]) == 2
    assert store.get("state")["id"] == sid


@pytest.mark.parametrize("new_finalized", [False, True])
def test_refresh_budget_skips_all_player_histories_until_labels_are_due(tmp_path, monkeypatch, new_finalized):
    from fpl_intelligence import decision, ingestion, optimization, prospective, simulation
    from fpl_intelligence.storage import now
    from fpl_intelligence.workflow import refresh_cycle

    store = Store(tmp_path)
    store.put(
        "market",
        {
            "events": [
                {"id": 8, "data_checked": True, "is_current": True},
                {"id": 9, "data_checked": False, "is_next": True},
            ]
        },
    )
    original = store.put("state", {"event": 9, "bank": 3})
    store.put("ingestion", {"captured_at": now(), "histories_requested": True})
    calls, fits = [], []

    async def ingest(self, *, histories=True):
        calls.append(histories)
        if new_finalized:
            self.store.put(
                "market",
                {
                    "events": [
                        {"id": 9, "data_checked": True, "is_current": True},
                        {"id": 10, "is_next": True},
                    ]
                },
            )
        return {}

    def forecast(saved_store, *args):
        fid = saved_store.put("forecast", {"start_event": 9})
        return {"id": fid}

    monkeypatch.setattr(ingestion.Official, "ingest", ingest)
    monkeypatch.setattr(decision, "train_decisions", lambda *args: fits.append(True))
    monkeypatch.setattr(prospective, "score_frozen_decisions", lambda *args: {})
    monkeypatch.setattr(simulation, "forecast", forecast)
    monkeypatch.setattr(optimization, "optimize", lambda *args: {"id": "test-plan"})
    result = refresh_cycle(store)
    assert calls == ([False, True] if new_finalized else [False])
    assert len(fits) == int(new_finalized)
    assert store.get("state")["id"] == original
    assert result["account_changed"] is False
