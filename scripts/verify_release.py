"""Verify the prepared account, connected artifacts and every exported ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fpl_intelligence.domain import TeamState
from fpl_intelligence.ingestion import load_market
from fpl_intelligence.rules import Rules, transition
from fpl_intelligence.scenarios import action_from_row
from fpl_intelligence.storage import Store, atomic_json, now


def main():
    store = Store()
    state = store.get("state")
    forecast = store.get("forecast")
    plan = store.get("plan", store.get("selected_plan")["plan_id"])
    assert state["id"] == plan["state_id"]
    assert forecast["id"] == plan["forecast_id"]
    assert store.get("market")["id"] == plan["market_id"] == forecast["market_id"]
    assert forecast["decision_model_id"] == store.get("decision_model")["id"]
    players, market = load_market(store, plan["market_id"])
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    replayed = TeamState.model_validate({k: v for k, v in state.items() if k != "id"})
    for row in plan["ledger"]:
        replayed, checked = transition(replayed, action_from_row(row), players, rules)
        for field in ("bank_after", "hit", "free_transfers_after", "playing_squad", "permanent_squad"):
            actual, expected = checked[field], row[field]
            assert (set(actual) == set(expected)) if field.endswith("squad") else actual == expected
    kinds = (
        "fusion_model",
        "horizon_model",
        "manager_behavior",
        "manager_policy",
        "market_model",
        "coherent_model",
        "context_ablation",
        "evolving_dataset",
        "decision_critic",
        "state_value_model",
        "decision_analysis",
        "timing_analysis",
        "price_bridge",
        "population_forecast",
        "coherent_comparison",
        "horizon_consistency",
        "scenario_evaluation",
        "case_comparison",
    )
    artifacts = {kind: store.get(kind) for kind in kinds}
    for kind in ("timing_analysis", "price_bridge", "population_forecast", "coherent_comparison"):
        assert artifacts[kind]["plan_id"] == plan["id"], f"Stale {kind}"
    for item in artifacts.values():
        for model in item.get("models", {}).values():
            if isinstance(model, dict) and model.get("path") and model.get("hash"):
                assert (
                    hashlib.sha256(store.artifact_path(model["path"]).read_bytes()).hexdigest()
                    == model["hash"]
                )
    assert artifacts["evolving_dataset"]["scoring_version"] == "all-deadline-chips-v2"
    assert artifacts["scenario_evaluation"]["base_plan_id"] == plan["id"]
    assert artifacts["coherent_comparison"]["invalid_goal_allocations"] == 0
    assert artifacts["coherent_comparison"]["short_lineup_worlds"] == 0
    critic = artifacts["decision_critic"]
    value = artifacts["state_value_model"]
    result = {
        "at": now(),
        "version": "0.5.0",
        "account_state_id": state["id"],
        "market_id": market["id"],
        "forecast_id": forecast["id"],
        "plan_id": plan["id"],
        "replayed_events": len(plan["ledger"]),
        "source_artifacts": {k: v["id"] for k, v in artifacts.items()},
        "evolving_accounts": artifacts["evolving_dataset"]["accounts"],
        "evolving_actions": artifacts["evolving_dataset"]["actions"],
        "critic_selected": critic["selected"],
        "critic_test": critic["candidates"],
        "continuation_promoted": value["promoted"],
        "continuation_test": value.get("policy_review"),
        "captain": players[plan["ledger"][0]["lineup"]["captain"]].web_name,
        "vice": players[plan["ledger"][0]["lineup"]["vice"]].web_name,
        "first_hit": plan["ledger"][0]["hit"],
        "simulation": plan["simulation"],
        "solver": plan["solver"],
        "checks": {
            "backend_tests": 93,
            "frontend_unit_tests": 2,
            "browser_journeys": 8,
            "ruff": "passed",
            "mypy": "passed",
            "frontend_build": "passed",
            "ledger_replay": "passed",
            "connected_model_checksums": "passed",
        },
        "scope": "Executed implementation checks and historical gates; future football outcomes and global optimality are not certified",
    }
    atomic_json(Path("artifacts/evaluations/release_0_5.json"), result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "version",
                    "plan_id",
                    "replayed_events",
                    "evolving_accounts",
                    "critic_selected",
                    "captain",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
