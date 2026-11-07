"""Verify the prepared account, connected artifacts and every exported ledger."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
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
        "transfer_policy_model",
        "personalized_policy_analysis",
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
        "availability_model",
        "attack_model",
        "score_process_model",
        "temporal_model",
        "discipline_report",
        "odds_refresh",
        "odds_model",
        "evidence_impact",
        "policy_selection",
        "weekly_replay",
        "weekly_baseline_comparison",
        "consensus_model",
        "consensus_comparison",
        "consensus_policy_validation",
        "pipeline_replay",
        "shadow_cohort",
        "shadow_evaluation",
        "chip_counterfactual",
    )
    artifacts = {kind: store.get(kind) for kind in kinds}
    chip_report = artifacts["chip_counterfactual"]
    assert chip_report["base_plan_id"] == plan["id"]
    assert chip_report["forecast_id"] == forecast["id"]
    actual_dates = {(r["chip"], r["event"]) for r in chip_report["calendar"]}
    expected_dates = set()
    account = TeamState.model_validate({k: v for k, v in state.items() if k != "id"})
    for event in range(account.event, plan["end_event"] + 1):
        for chip in ("wildcard", "freehit", "bboost", "3xc"):
            try:
                lo, hi = rules.chip_window(chip, event)
            except ValueError:
                continue
            if any(c.chip == chip and lo <= c.event <= hi for c in account.used_chips):
                continue
            if account.active_chip == chip and lo <= account.event <= hi:
                continue
            if event == account.event:
                if account.active_chip not in (None, chip):
                    continue
                if chip == "freehit" and (
                    any(c.chip == chip and c.event == event - 1 for c in account.used_chips)
                    or (
                        account.transfers_made
                        and (account.deadline_lots is None or account.deadline_bank is None)
                    )
                ):
                    continue
            expected_dates.add((chip, event))
    assert actual_dates == expected_dates
    assert len(actual_dates) == chip_report["dates_evaluated"] == len(chip_report["calendar"])
    for date in chip_report["calendar"]:
        counterfactual = store.get("plan", date["plan_id"])
        assert all(counterfactual[k] == plan[k] for k in ("state_id", "forecast_id", "end_event"))
        assert any(
            r["chip"] == date["chip"] and r["event"] == date["event"] for r in counterfactual["ledger"]
        )
        replayed_date = account.model_copy(deep=True)
        for row in counterfactual["ledger"]:
            replayed_date, checked = transition(replayed_date, action_from_row(row), players, rules)
            assert checked["bank_after"] == row["bank_after"] and checked["hit"] == row["hit"]
            assert checked["free_transfers_after"] == row["free_transfers_after"]
            assert set(checked["playing_squad"]) == set(row["playing_squad"])
    for comparison in chip_report["comparisons"]:
        counterfactual = store.get("plan", comparison["without_plan_id"])
        lo, hi = comparison["window"]
        assert not any(
            r["chip"] == comparison["chip"] and lo <= r["event"] <= hi for r in counterfactual["ledger"]
        )
    from fpl_intelligence.decision import load_bundle

    load_bundle(store, store.get("decision_model"))
    role_id = store.get("decision_model").get("role_minutes_model_id")
    if role_id:
        import importlib.util

        import lightgbm as lgb

        role = store.get("role_minutes_model", role_id)
        validation = store.get("role_minutes_validation", role["validation_id"])
        assert forecast["role_minutes_model_id"] == role_id and role["promoted"] and validation["passed"]
        path = store.artifact_path(role["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == role["hash"]
        assert lgb.Booster(model_file=str(path)).num_feature() == role["width"] == 57
        if role["version"] == "role-minutes-v3":
            assert role.get("context_archive_id")
            audit = store.get("role_contract_audit")
            assert audit["passed"] and audit["role_minutes_model_id"] == role_id
            assert audit["maximum_core_delta"] == audit["maximum_role_source_delta"] == 0
        predictions = store.artifact_path(validation["prediction_path"])
        assert hashlib.sha256(predictions.read_bytes()).hexdigest() == validation["prediction_hash"]
        assert importlib.util.find_spec("torch") is None
        assert not list((store.root / "models").rglob("*.pt"))
    fusion = artifacts["fusion_model"]
    directory = store.root / "models" / fusion["directory"]
    for name, expected in fusion["tree_spec"]["hashes"].items():
        assert hashlib.sha256((directory / (name + ".txt")).read_bytes()).hexdigest() == expected
    horizon = artifacts["horizon_model"]
    assert artifacts["transfer_policy_model"]["horizon_model_id"] == horizon["id"]
    personalized = artifacts["personalized_policy_analysis"]
    assert personalized["policy_model_id"] == artifacts["transfer_policy_model"]["id"]
    assert personalized["horizon_model_id"] == horizon["id"]
    assert personalized["state_id"] == state["id"] and personalized["market_id"] == market["id"]
    path = store.root / "models" / horizon["directory"] / "value.txt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == horizon["model_hash"]
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
    replay = artifacts["weekly_replay"]
    assert [w["event"] for w in replay["weeks"]] == list(range(1, 10))
    assert replay["account_state_unchanged"]
    for week in replay["weeks"]:
        group = f"{week['season']}-GW{week['event']:02d}"
        assert week["training"]["fit_max_group"] < week["training"]["calibration_max_group"] < group
        frozen = store.get("weekly_replay_decision", week["decision_id"])
        assert "actual_points" not in frozen and "hindsight_points" not in frozen
        source = store.get("replay_inputs", week["input_id"])
        from fpl_intelligence.availability import moment

        proof = source["fixture_proof"]
        assert moment(proof.get("git_committed_at", proof.get("source_captured_at"))) < moment(
            source["deadline"]
        )
        for model in week["training"]["models"].values():
            assert (
                hashlib.sha256(store.artifact_path(model["path"]).read_bytes()).hexdigest() == model["hash"]
            )
    from fpl_intelligence.consensus import load_verified_models

    consensus = artifacts["consensus_model"]
    load_verified_models(store, consensus["tree_spec"])
    load_verified_models(store, consensus["reference_spec"])
    assert not consensus["promoted"]
    assert artifacts["consensus_comparison"]["consensus_model_id"] == consensus["id"]
    assert artifacts["consensus_comparison"]["state_id"] == state["id"]
    assert artifacts["consensus_policy_validation"]["consensus_model_id"] == consensus["id"]
    from fpl_intelligence.pipeline_replay import verify_pipeline_replay

    pipeline = artifacts["pipeline_replay"]
    assert pipeline["production_budget"] and pipeline["live_pointers_unchanged"]
    pipeline_proof = verify_pipeline_replay(store, pipeline)
    assert pipeline_proof["sealed_deadlines"] == 28 and pipeline_proof["validated_actions"] == 84
    assert artifacts["shadow_cohort"]["real_account_unchanged"]
    from fpl_intelligence.availability import moment

    assert moment(artifacts["shadow_cohort"]["frozen_at"]) < moment(artifacts["shadow_cohort"]["deadline"])
    future = {}
    if store.list("future_policy_replay", 1):
        from fpl_intelligence.future_policy_verification import verify_future_policy

        accepted = store.get("future_policy_replay")
        future["earlier_policy"] = verify_future_policy(store, accepted)
        future["earlier_points"] = accepted["candidate_points"]
        future["earlier_reference"] = accepted["baseline_points"]
        future["policy_gate_passed"] = accepted["policy_gate_passed"]
        if store.list("future_policy_diagnostic", 1):
            diagnostic = store.get("future_policy_diagnostic")
            future["current_policy"] = verify_future_policy(store, diagnostic)
            future["current_points"] = diagnostic["candidate_points"]
            future["current_weekly_points"] = [
                {"gw": r["event"], "points": r["net_points"]} for r in diagnostic["weeks"]
            ]
    identity = store.get("decision_model").get("future_horizon_model_id")
    role_weekly = None
    if store.list("role_weekly_replay", 1):
        role_weekly = store.get("role_weekly_replay")
        for week in role_weekly["weeks"]:
            group = f"{week['season']}-GW{week['event']:02d}"
            frozen = store.get("weekly_replay_decision", week["decision_id"])
            assert "actual_points" not in frozen
            assert week["training"]["role_minutes"]["calibration_max_group"] < group
            for item in week["training"]["models"].values():
                assert (
                    hashlib.sha256(store.artifact_path(item["path"]).read_bytes()).hexdigest() == item["hash"]
                )
        assert role_weekly["total_points"] == sum(w["actual_points"] for w in role_weekly["weeks"])
    if identity:
        from fpl_intelligence.future_expert import load_expert

        expert = store.get("future_horizon_model", identity)
        distribution = store.get("future_distribution_validation", expert["distribution_validation_id"])
        assert distribution["passed"]
        assert expert["policy_gate_passed"] and expert["forecast_gate_passed"]
        assert forecast["future_horizon_model_id"] == identity
        assert expert["acceptance_id"] == store.get("future_policy_replay")["id"]
        load_expert(store, expert)
        assert (
            hashlib.sha256(store.artifact_path(expert["data_path"]).read_bytes()).hexdigest()
            == expert["data_hash"]
        )
        future["active_expert_id"] = identity
        future["distribution_validation_id"] = distribution["id"]
    checks = json.loads(Path("artifacts/runtime/checks/checks.json").read_text())
    assert checks["passed"] and checks["browser_failed"] == 0
    critic = artifacts["decision_critic"]
    value = artifacts["state_value_model"]
    result = {
        "at": now(),
        "version": version("fpl-intelligence"),
        "serving_policy": {
            "default_solver_time_limit": None,
            "default_live_mip_gap": 0,
            "historical_scorecards_recomputed": False,
            "retained_forecast_cutoff": forecast.get("cutoff"),
            "scope": "Engineering verification with retained v0.13 forecast/plan evidence; does not certify the unlimited-search policy on historical football outcomes",
        },
        "account_state_id": state["id"],
        "market_id": market["id"],
        "forecast_id": forecast["id"],
        "plan_id": plan["id"],
        "replayed_events": len(plan["ledger"]),
        "chip_dates_verified": len(actual_dates),
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
        "role_minutes": {
            "model_id": role_id,
            "validation_id": role["validation_id"],
            "width": role["width"],
            "completed_labels_through": role["completed_labels_through"],
            "weights_and_prediction_hashes": "verified",
            "local_tensor_runtime": "removed",
        }
        if role_id
        else None,
        "resource_selection": artifacts["policy_selection"].get("selection"),
        "current_chip_evidence": artifacts["policy_selection"].get("immediate_chip_evidence"),
        "weekly_scores": [
            {"gw": w["event"], "actual_points": w["actual_points"], "hindsight_points": w["hindsight_points"]}
            for w in replay["weeks"]
        ],
        "role_weekly_scores": [
            {"gw": w["event"], "actual_points": w["actual_points"]} for w in role_weekly["weeks"]
        ]
        if role_weekly
        else None,
        "role_weekly_total": role_weekly["total_points"] if role_weekly else None,
        "consensus_weekly_totals": consensus["current_totals"],
        "consensus_weekly_acceptance": consensus["acceptance"],
        "consensus_continuous_totals": artifacts["consensus_policy_validation"]["current_totals"],
        "consensus_continuous_acceptance": artifacts["consensus_policy_validation"]["acceptance"],
        "consensus_promoted": False,
        "integrated_policy_summaries": pipeline["summaries"],
        "integrated_policy_verification": pipeline_proof,
        "future_fixture_validation": future,
        "prospective_account_cohort_id": artifacts["shadow_cohort"]["id"],
        "prospective_outcomes": {
            k: artifacts["shadow_evaluation"][k] for k in ("finalized_deadlines", "pending_deadlines")
        },
        "checks": {
            **{k: v for k, v in checks.items() if k != "results"},
            "ledger_replay": "passed",
            "connected_model_checksums": "passed",
        },
        "scope": "Executed implementation checks and historical gates; future football outcomes and global optimality are not certified",
    }
    release = "_".join(result["version"].split(".")[:2])
    atomic_json(Path(f"artifacts/evaluations/release_{release}.json"), result)
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
