"""Ordered full-system training and receding-horizon refresh."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from .account_sync import state_lock
from .storage import Store, now


def full_training(store: Store, samples=24, epochs=20, refresh=False):
    from .coherent import compare_football_worlds, train_components, validate_bonus_model
    from .context_archive import import_context
    from .context_validation import evaluate_context
    from .decision import train_decisions
    from .decision_runtime import analyze_decisions, refine_frontier, timing_analysis, validate_horizons
    from .decision_training import evolve, train_decision_models
    from .fusion import compare_live, review_fusion, train_fusion
    from .horizon import train_horizon
    from .manager_behavior import collect_top_managers, train_behavior
    from .manager_policy import train_manager_policy
    from .market_model import collect_market_windows, train_market
    from .population import collect_population, evaluate_population
    from .prospective import freeze, score_frozen_decisions
    from .reports import export
    from .simulation import forecast

    stages = []

    def record(name, result):
        stages.append({"stage": name, "artifact_id": result["id"], "completed_at": now()})
        print("Completed workflow stage", name, flush=True)

    if refresh:
        from .ingestion import Official
        from .intelligence import refresh_sources

        asyncio.run(Official(store).ingest())
        asyncio.run(refresh_sources(store))
    record("context", asyncio.run(import_context(store)))
    record("football_models", football_training(store, refresh))
    record("reference", train_decisions(store, False))
    record("source_fusion", train_fusion(store, epochs))
    record("fusion_review", review_fusion(store))
    compare_live(store)
    from .consensus import compare_live_consensus, fit_consensus, prepare_inputs, validate_continuous_policy

    asyncio.run(prepare_inputs(store))
    record("published_signal_fusion", fit_consensus(store))
    record("published_signal_transfer_validation", validate_continuous_policy(store))
    record("published_signal_current_comparison", compare_live_consensus(store))
    record("horizon", train_horizon(store))
    if refresh or not store.list("top_manager_history", 1):
        record("top_1000", asyncio.run(collect_top_managers(store)))
    record("manager_counts", train_behavior(store))
    record("manager_choices", train_manager_policy(store))
    record("price_windows", asyncio.run(collect_market_windows(store)))
    record("market", train_market(store))
    record("components", train_components(store))
    record("joint_bonus_validation", validate_bonus_model(store))
    record("context_ablation", evaluate_context(store))
    record("evolving_episodes", evolve(store, samples, max_transfers=15))
    record("decision_learning", train_decision_models(store))
    state = store.get("state")
    current_forecast = forecast(store, state["event"], 38, 20000)
    record("forecast", current_forecast)
    from .optimization import optimize

    end = 19 if state["event"] <= 19 else 38

    def select(plan_id):
        with state_lock:
            if (
                store.get("state")["id"] != state["id"]
                or store.get("forecast")["id"] != current_forecast["id"]
            ):
                raise ValueError(
                    "Account or forecast changed during training; completed plans retained as analyses"
                )
            store.put("selected_plan", {"plan_id": plan_id, "at": now()})

    baseline = optimize(store, state["id"], current_forecast["id"], end, 60, use_decision_critic=False)
    select(baseline["id"])
    if state["event"] == 9:
        from .rivals import case_comparison

        cases = case_comparison(store, state["id"], store.get("forecast")["id"], 30)
        record("original_case_families", cases)
        select(cases["selected_plan_id"])
    record("decisions", analyze_decisions(store))
    record("frontier", refine_frontier(store))
    if end - state["event"] > 3:
        record("horizon_consistency", validate_horizons(store))
    from .objectives import choose_recommendation

    selected, selection = choose_recommendation(
        store, [store.get("plan", store.get("selected_plan")["plan_id"])], current_forecast
    )
    select(selected["id"])
    record("final_selection", selection)
    from .context import analyze_evidence

    record("evidence_impact", analyze_evidence(store, selected["id"]))
    record("timing", timing_analysis(store))
    record("football_world_comparison", compare_football_worlds(store))
    record("population_sample", asyncio.run(collect_population(store)))
    record("population", evaluate_population(store))
    record("prospective", freeze(store, store.get("selected_plan")["plan_id"]))
    score_frozen_decisions(store)
    plan = store.get("plan", store.get("selected_plan")["plan_id"])
    from .chips import compare_chips, compare_immediate_hit

    record("chip_counterfactual", compare_chips(store, plan["id"]))
    record("hit_counterfactual", compare_immediate_hit(store, plan["id"]))
    if end > state["event"]:
        from .scenarios import PlayerChange, ScenarioSpec, evaluate_scenarios

        reveal = state["event"] + 1
        record(
            "contingency_recourse",
            evaluate_scenarios(
                store,
                plan["id"],
                [
                    ScenarioSpec(
                        name="Hypothetical captain absence after next observation",
                        reveal_event=reveal,
                        players=[
                            PlayerChange(
                                player_id=plan["ledger"][0]["lineup"]["captain"],
                                from_event=reveal,
                                through_event=reveal,
                                availability=0,
                            )
                        ],
                    )
                ],
            ),
        )
    export(store, plan, Path("artifacts/runtime/report"))
    from .replay import run_replay

    record("weekly_replay", run_replay(store))
    from .pipeline_jobs import run_pipeline_job
    from .shadow_accounts import advance_shadow_accounts, freeze_shadow_accounts

    record("integrated_pipeline_replay", run_pipeline_job(store))
    try:
        advance_shadow_accounts(store)
        shadow = freeze_shadow_accounts(store)
        if shadow.get("id"):
            record("prospective_continuous_accounts", shadow)
    except ValueError as error:
        stages.append(
            {
                "stage": "prospective_continuous_accounts",
                "status": "Capture unavailable",
                "reason": str(error),
            }
        )
    result = {
        "at": now(),
        "stages": stages,
        "selected_plan_id": plan["id"],
        "account_changed": False,
        "scope": "Implementation workflow completed; prospective football outcomes are evaluated only after finalization",
    }
    result["id"] = store.put("training_workflow", result)
    return result


def refresh_cycle(store: Store):
    from .decision import train_decisions
    from .ingestion import Official
    from .optimization import optimize
    from .simulation import forecast

    previous = store.get("market")
    original_state = store.get("state")
    checked = {e["id"] for e in previous["events"] if e.get("data_checked")}
    asyncio.run(Official(store).ingest(entry=original_state.get("entry_id"), histories=False))
    market = store.get("market")
    new_finalized = checked != {e["id"] for e in market["events"] if e.get("data_checked")}
    full_refreshes = [store.get("ingestion", r["id"]) for r in store.list("ingestion", 100)]
    last_full = next((r for r in full_refreshes if r.get("histories_requested")), None)
    current: dict = next((e for e in market["events"] if e.get("is_current")), {})
    daily_due = bool(current.get("data_checked")) and (
        last_full is None
        or (datetime.now(timezone.utc) - datetime.fromisoformat(last_full["captured_at"])).total_seconds()
        >= 86400
    )
    changed_labels = False
    if new_finalized or daily_due:

        def history_hashes():
            with store.db() as db:
                return {
                    r["source"]: r["hash"]
                    for r in db.execute(
                        "SELECT source,hash FROM snapshots WHERE id IN "
                        "(SELECT max(id) FROM snapshots WHERE source LIKE 'history_%' GROUP BY source)"
                    )
                }

        before_histories = history_hashes()
        asyncio.run(Official(store).ingest(histories=True))
        market = store.get("market")
        changed_labels = before_histories != history_hashes()
    active = store.get("decision_model") if store.list("decision_model", 1) else {}
    represented = max(
        (r["event"] for r in active.get("predictions", []) if r["season"] == market.get("season")), default=0
    )
    finalized = max((e["id"] for e in market["events"] if e.get("data_checked")), default=0)
    needs_refit = bool(active) and represented < finalized
    if new_finalized or changed_labels or needs_refit:
        from .prospective import score_frozen_decisions

        score_frozen_decisions(store)
        from .shadow_accounts import advance_shadow_accounts

        try:
            advance_shadow_accounts(store)
        except ValueError as error:
            store.put(
                "shadow_capture_status",
                {"at": now(), "status": "Scoring requires review", "reason": str(error)},
            )
        if active.get("fusion_model_id"):
            from .fusion import refit_active_fusion

            refit_active_fusion(store)
        else:
            train_decisions(store, False)
    from .account_sync import next_event, reconcile_public

    synchronization = reconcile_public(store, original_state["id"])
    state = store.get("state")
    event = next_event(market)
    if event is None:
        return {"at": now(), "status": "Season finished", "account_sync": synchronization}
    if state["event"] != event:
        return {
            "at": now(),
            "status": "Account reconciliation required",
            "market_id": market["id"],
            "account_sync": synchronization,
            "account_changed": synchronization.get("account_changed", False),
        }
    result = forecast(store, event, 38, 4000)
    consensus_status = None
    if store.list("consensus_model", 1):
        from .consensus import compare_live_consensus

        try:
            comparison = compare_live_consensus(store)
            consensus_status = {"status": "Updated separate weekly comparison", "id": comparison["id"]}
        except ValueError as error:
            consensus_status = {
                "status": "Comparison unavailable; primary policy retained",
                "reason": str(error),
            }
    try:
        plan = optimize(store, state["id"], result["id"], 19 if event <= 19 else 38, 60)
        from .objectives import choose_recommendation

        plan, _ = choose_recommendation(store, [plan], result)
        with state_lock:
            if store.get("state")["id"] == state["id"] and store.get("forecast")["id"] == result["id"]:
                store.put("selected_plan", {"plan_id": plan["id"], "at": now()})
    except ValueError as error:
        from .optimization import hold_plan

        fallback = hold_plan(store, state["id"], result["id"], 19 if event <= 19 else 38)
        with state_lock:
            if store.get("state")["id"] == state["id"] and store.get("forecast")["id"] == result["id"]:
                store.put("selected_plan", {"plan_id": fallback["id"], "at": now()})
        return {
            "at": now(),
            "status": "Updated forecast; planning requires review",
            "error": str(error),
            "plan_id": fallback["id"],
            "forecast_id": result["id"],
            "fallback": "Validated hold policy retained; inspect the reported planning error before acting",
            "account_sync": synchronization,
            "account_changed": synchronization.get("account_changed", False),
            "consensus_comparison": consensus_status,
        }
    shadow_status = None
    from .shadow_accounts import freeze_shadow_accounts

    try:
        shadow = freeze_shadow_accounts(store)
        shadow_status = {"status": "Synchronized prospective accounts frozen", "cohort_id": shadow.get("id")}
    except ValueError as error:
        shadow_status = {
            "status": "Shadow capture requires review; real recommendation retained",
            "reason": str(error),
        }
    store.put("shadow_capture_status", {"at": now(), **shadow_status})
    return {
        "at": now(),
        "status": "Refreshed and recomputed",
        "prospective_accounts": shadow_status,
        "consensus_comparison": consensus_status,
        "forecast_id": result["id"],
        "plan_id": plan["id"],
        "account_sync": synchronization,
        "account_changed": synchronization.get("account_changed", False),
    }


def football_training(store, refresh=False):
    """Connect collected event facts to the models that consume them."""
    import httpx

    from .attack import train_attack
    from .availability import train_availability
    from .discipline import build_discipline, collect_discipline_news
    from .odds import calibrate_odds, collect_odds
    from .score_process import train_score_process
    from .shot_data import collect_shots
    from .temporal_validation import luck_report, validate_temporal

    stages = []

    def record(name, result):
        stages.append({"stage": name, "artifact_id": result["id"]})

    if refresh or not store.list("shot_archive", 1):
        record("shots", asyncio.run(collect_shots(store)))
    if refresh:
        for name, source in [("discipline_source", collect_discipline_news), ("odds_source", collect_odds)]:
            try:
                record(name, asyncio.run(source(store)))
            except (ValueError, httpx.HTTPError) as error:
                stages.append(
                    {
                        "stage": name,
                        "status": "Optional source unavailable; retained statistical fallback",
                        "error": str(error)[:180],
                    }
                )
    record("discipline", build_discipline(store))
    record("availability", train_availability(store))
    record("temporal", validate_temporal(store))
    record("attack", train_attack(store))
    record("score_process", train_score_process(store))
    record("market_constraints", calibrate_odds(store))
    record("luck_diagnostic", luck_report(store))
    report = {
        "at": now(),
        "stages": stages,
        "scope": "Complete model fits and later-deadline gates; unavailable optional data never becomes a fabricated signal",
    }
    report["id"] = store.put("football_training", report)
    return report
