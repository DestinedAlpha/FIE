"""Immutable pre-deadline shadow decisions and observed policy evaluation."""

from __future__ import annotations

from datetime import datetime, timezone

from .domain import Lineup
from .features import histories
from .ingestion import load_market
from .rules import Rules, score_gameweek
from .storage import Store, now


def freeze(store: Store, plan_id: str) -> dict:
    plan = store.get("plan", plan_id)
    state = store.get("state", plan["state_id"])
    current = store.get("state")
    forecast = store.get("forecast", plan["forecast_id"])
    _, market = load_market(store)
    event = plan["ledger"][0]["event"]
    deadline = next(e["deadline_time"] for e in market["events"] if e["id"] == event)
    if datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
        raise ValueError("Cannot freeze a decision after its deadline")
    if plan["state_id"] != current["id"] or plan["market_id"] != market["id"]:
        raise ValueError("Cannot freeze a stale account or market plan")
    if forecast["id"] != store.get("forecast")["id"]:
        raise ValueError("Cannot freeze a superseded forecast")
    if (
        forecast.get("model_version") != "learned-points-v1"
        or forecast.get("decision_model_id") != store.get("decision_model")["id"]
    ):
        raise ValueError("Cannot freeze a decision from a retired forecast model")
    result = {
        "frozen_at": now(),
        "event": event,
        "season": state["season"],
        "deadline": deadline,
        "plan_id": plan_id,
        "state_id": plan["state_id"],
        "forecast_id": plan["forecast_id"],
        "action": plan["ledger"][0],
        "starting_points": state["points"],
        "track": "prospective_shadow_decision",
        "submitted_to_account": False,
        "rules_hash": plan["rules_hash"],
    }
    result["id"] = store.put("shadow_decision", result)
    return result


def score_frozen_decisions(store: Store) -> dict:
    players, market = load_market(store)
    history = histories(store, players)
    finalized = {e["id"] for e in market["events"] if e.get("data_checked")}
    selected = {}
    for row in store.list("shadow_decision", 1000):
        decision = store.get("shadow_decision", row["id"])
        key = (decision["season"], decision["event"])
        if key not in selected and decision["season"] == market["season"]:
            selected[key] = decision
    rows = []
    for (_, event), decision in selected.items():
        if event not in finalized:
            continue
        plan = store.get("plan", decision["plan_id"])
        _, frozen_market = load_market(store, plan["market_id"])
        rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=frozen_market["source_hash"])[0])
        actual = {p: sum(r["total_points"] for r in history[p] if r["round"] == event) for p in players}
        minutes = {p: sum(r["minutes"] for r in history[p] if r["round"] == event) for p in players}
        action = decision["action"]
        points = (
            score_gameweek(
                actual, minutes, Lineup.model_validate(action["lineup"]), players, rules, action["chip"]
            )
            - action["hit"]
        )
        rows.append(
            {
                "decision_id": decision["id"],
                "event": event,
                "realized_net_points": points,
                "hit": action["hit"],
                "frozen_at": decision["frozen_at"],
                "deadline": decision["deadline"],
            }
        )
    report = {
        "at": now(),
        "frozen_decisions": len(selected),
        "finalized_decisions": len(rows),
        "outcomes": rows,
        "total_realized_net_points": sum(r["realized_net_points"] for r in rows),
        "track": "prospective_shadow_decision",
    }
    report["id"] = store.put("prospective_policy", report)
    return report
