from collections import defaultdict

import pytest

from fpl_intelligence.domain import Action
from fpl_intelligence.rules import best_lineup, score_gameweek
from fpl_intelligence.shadow_accounts import advance_shadow_accounts, freeze_shadow_accounts
from fpl_intelligence.storage import Store


@pytest.fixture
def shadow_case(tmp_path, market, rules, state, monkeypatch):
    store = Store(tmp_path / "data")
    source = store.snapshot("bootstrap", "fixture://bootstrap", {})
    events = [
        {"id": event, "deadline_time": f"2026-10-{event + 20}T11:00:00Z", "is_next": event == 9}
        for event in (9, 10, 11)
    ]
    market_id = store.put(
        "market",
        {
            "season": state.season,
            "players": [p.model_dump() for p in market.values()],
            "events": events,
            "teams": [],
            "fixtures": [],
            "source_hash": source["hash"],
        },
    )
    model_id = store.put("decision_model", {"fixture": True})
    forecast_id = store.put(
        "forecast",
        {
            "market_id": market_id,
            "decision_model_id": model_id,
            "start_event": 9,
            "players": [
                {"player_id": p, "event": e, "mean": float(p), "fixture_count": 1}
                for p in market
                for e in (9, 10)
            ],
        },
    )
    store.put("state", state.model_dump())
    store.put("selected_plan", {"plan_id": "real-plan"})
    store.put("policy_selection", {"state_id": "real-state"})
    policies = ("primary_multiweek", "ownership_no_hits", "official_expected_no_hits")
    store.put(
        "shadow_registry",
        {
            "season": state.season,
            "registry_key": "test-accounts",
            "seed_event": 9,
            "accounts": {name: state.model_dump() for name in policies},
            "seed_scope": "Equal hypothetical accounts",
            "missed_capture_policy": "Standing hold",
        },
    )
    monkeypatch.setattr("fpl_intelligence.rules.Rules.from_bootstrap", lambda *_: rules)
    monkeypatch.setattr("fpl_intelligence.shadow_accounts.now", lambda: "2026-10-28T11:00:00+00:00")
    points, minutes = defaultdict(dict), defaultdict(dict)
    monkeypatch.setattr("fpl_intelligence.shadow_accounts.actual_events", lambda *_: (points, minutes))

    def propose(saved, states, forecast, players, constraints, seconds):
        actions = {}
        for name, account in states.items():
            ids = [lot.player_id for lot in account.lots]
            if name == "primary_multiweek":
                ids = [16 if p == 1 else p for p in ids]
                action = Action(sell=[1], buy=[16], chip="freehit")
            else:
                action = Action()
            action.lineup = best_lineup(ids, {p: float(p) for p in players}, players, constraints)
            actions[name] = action
        return actions

    monkeypatch.setattr("fpl_intelligence.shadow_accounts.propose_accounts", propose)
    return store, points, minutes, forecast_id


def finalize(store, points, minutes, through, value=1):
    body = store.get("market")
    for e in body["events"]:
        e["data_checked"] = e["id"] <= through
        e["is_next"] = e["id"] == through + 1
    store.put("market", {k: v for k, v in body.items() if k != "id"})
    for event in range(9, through + 1):
        points[event] = {p: float(value) for p in range(1, 46)}
        minutes[event] = {p: 90 for p in range(1, 46)}


def test_frozen_accounts_score_freehit_restore_and_remain_idempotent(shadow_case, market, rules, state):
    store, points, minutes, _ = shadow_case
    kinds = ("state", "selected_plan", "forecast", "decision_model", "policy_selection")
    before = {kind: store.get(kind)["id"] for kind in kinds}
    cohort = freeze_shadow_accounts(store, seconds=1)
    assert freeze_shadow_accounts(store, seconds=1)["id"] == cohort["id"]
    assert len(store.list("shadow_cohort")) == 1 and cohort["real_account_unchanged"]
    finalize(store, points, minutes, 9)
    points[9][16] = 9
    report = advance_shadow_accounts(store)
    for name, row in cohort["actions"].items():
        action = Action.model_validate(row["action"])
        expected = score_gameweek(points[9], minutes[9], action.lineup, market, rules, action.chip)
        assert report["total_net_points"][name] == expected
    saved = store.get("shadow_registry")["accounts"]
    primary = saved["primary_multiweek"]
    assert primary["bank"] == state.bank and primary["free_transfers"] == 2
    assert {lot["player_id"] for lot in primary["lots"]} == set(range(1, 16))
    assert saved["ownership_no_hits"]["free_transfers"] == 3
    assert advance_shadow_accounts(store)["advanced_events"] == []
    assert {kind: store.get(kind)["id"] for kind in kinds} == before


def test_missed_deadline_executes_precommitted_hold_and_revises_points(shadow_case):
    store, points, minutes, _ = shadow_case
    frozen = freeze_shadow_accounts(store, seconds=1)
    finalize(store, points, minutes, 10)
    report = advance_shadow_accounts(store)
    assert report["advanced_events"] == [9, 10]
    assert report["finalized_deadlines"] == 2
    held = next(
        store.get("shadow_cohort", r["id"])
        for r in store.list("shadow_cohort")
        if store.get("shadow_cohort", r["id"])["event"] == 10
    )
    assert held["standing_source_id"] == frozen["id"]
    assert held["frozen_at"] == frozen["frozen_at"]
    for name, row in held["actions"].items():
        assert not row["action"]["buy"] and row["action"]["chip"] is None
        assert row["action"]["lineup"] == frozen["actions"][name]["standing_lineup"]
    points[9] = {p: v + 1 for p, v in points[9].items()}
    corrected = advance_shadow_accounts(store)
    assert corrected["advanced_events"] == []
    assert all(
        corrected["total_net_points"][p] == total + 12 for p, total in report["total_net_points"].items()
    )
    assert advance_shadow_accounts(store)["total_net_points"] == corrected["total_net_points"]


def test_capture_rejects_deadline_race_without_saving_actions(shadow_case, monkeypatch):
    store, _, _, _ = shadow_case
    from fpl_intelligence import shadow_accounts

    original = shadow_accounts.propose_accounts

    def delayed(*args):
        choices = original(*args)
        monkeypatch.setattr(shadow_accounts, "now", lambda: "2026-10-29T11:00:00+00:00")
        return choices

    monkeypatch.setattr(shadow_accounts, "propose_accounts", delayed)
    with pytest.raises(ValueError, match="deadline passed"):
        freeze_shadow_accounts(store, seconds=1)
    assert not store.list("shadow_cohort")


def test_finalized_week_without_match_history_is_not_counted_as_zero(shadow_case):
    store, points, minutes, _ = shadow_case
    freeze_shadow_accounts(store, seconds=1)
    finalize(store, points, minutes, 9)
    points.clear()
    with pytest.raises(ValueError, match="history is missing"):
        advance_shadow_accounts(store)
    assert store.get("shadow_registry")["accounts"]["primary_multiweek"]["event"] == 9


def test_ongoing_week_advances_finances_without_inventing_final_points(shadow_case):
    store, points, minutes, _ = shadow_case
    freeze_shadow_accounts(store, seconds=1)
    body = store.get("market")
    for event in body["events"]:
        event["is_next"] = event["id"] == 10
    store.put("market", {k: v for k, v in body.items() if k != "id"})
    pending = advance_shadow_accounts(store)
    assert pending["advanced_events"] == [9] and pending["finalized_deadlines"] == 0
    assert pending["pending_deadlines"] == [9]
    assert all(v == 0 for v in pending["total_net_points"].values())
    assert all(a["event"] == 10 for a in store.get("shadow_registry")["accounts"].values())
    finalize(store, points, minutes, 9)
    scored = advance_shadow_accounts(store)
    assert scored["advanced_events"] == [] and scored["finalized_deadlines"] == 1
    assert all(v == 12 for v in scored["total_net_points"].values())


def test_hypothetical_selection_receipts_do_not_hide_real_recommendations(tmp_path):
    from fpl_intelligence.shadow_accounts import ShadowStore

    store = Store(tmp_path)
    real = store.put("policy_selection", {"state_id": "real"})
    ShadowStore(tmp_path).put("policy_selection", {"state_id": "hypothetical"})
    assert store.get("policy_selection")["id"] == real
    assert store.get("shadow_policy_selection")["state_id"] == "hypothetical"
