"""Ordered full-system training and receding-horizon refresh."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from .account_sync import locked_state
from .storage import Store, now


def full_training(store: Store, samples=24, refresh=False):
    from .coherent import compare_football_worlds, train_components, validate_bonus_model
    from .context_archive import import_context
    from .context_validation import evaluate_context
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
    from .role_minutes import ensure
    from .training import train_current

    record("reference", train_current(store))
    record("source_fusion", train_fusion(store))
    record("fusion_review", review_fusion(store))
    compare_live(store)
    from .consensus import compare_live_consensus, fit_consensus, prepare_inputs, validate_continuous_policy

    asyncio.run(prepare_inputs(store))
    record("published_signal_fusion", fit_consensus(store))
    record("published_signal_transfer_validation", validate_continuous_policy(store))
    record("published_signal_current_comparison", compare_live_consensus(store))
    from .future_training import train_future_component

    record("future_fixture_learning", train_future_component(store))
    record("role_minutes", ensure(store, store.get("decision_model")))
    record("horizon", train_horizon(store))
    from .policy_learning import train_policy

    record("transfer_value_learning", train_policy(store))
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
        from .readiness import selection_is_current
        from .search_control import check_cancelled

        with locked_state(store):
            check_cancelled()
            if not selection_is_current(store, store.get("plan", plan_id)):
                raise ValueError(
                    "Inputs or deadline changed during training; completed plans retained as analyses"
                )
            store.put("selected_plan", {"plan_id": plan_id, "at": now()})

    from .search_control import live_search

    with live_search():
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
        from .chips import compare_chips, compare_immediate_hit, rebase_chip_comparison

        chip_report = compare_chips(store, store.get("selected_plan")["plan_id"])
        from .objectives import choose_recommendation

        selected, selection = choose_recommendation(
            store, [store.get("plan", store.get("selected_plan")["plan_id"])], current_forecast
        )
        select(selected["id"])
        record("chip_counterfactual", rebase_chip_comparison(store, chip_report, selected["id"]))
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


def refresh_data(store: Store, *, expected_state_id=None):
    """User-facing data/team synchronization never trains or launches a solver."""
    from .account_sync import reconcile_public
    from .ingestion import Official
    from .search_control import report_phase

    original = store.get("state", expected_state_id) if store.list("state", 1) else {}
    report_phase("Fetching official data and team")
    report = asyncio.run(
        Official(store).ingest(entry=original.get("entry_id"), histories=False, auxiliary=False)
    )
    report_phase("Updating team and checking forecast freshness")
    sync = (
        reconcile_public(store, original["id"])
        if original
        else {"status": "Load an account first", "account_changed": False}
    )
    return {
        "status": "Data and team refreshed",
        "account_sync": sync,
        "account_changed": sync.get("account_changed", False),
        "market_id": store.get("market")["id"],
        "ingestion_id": report["id"],
        "forecast_updated": False,
        "plan_optimized": False,
    }


def refresh_cycle(store: Store, *, plan=True, expected_state_id=None):
    from .ingestion import Official
    from .optimization import optimize
    from .search_control import report_phase
    from .simulation import forecast

    report_phase("Fetching official data")
    previous = store.get("market")
    original_state = store.get("state", expected_state_id)
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
    active = store.get("decision_model") if store.list("decision_model", 1) else {}
    from .model_refresh import represented_group, training_needed

    represented = represented_group(active)
    finalized = max((e["id"] for e in market["events"] if e.get("data_checked")), default=0)
    needs_history = represented < f"{market['season']}-GW{finalized:02d}" and (
        last_full is None or last_full.get("finalized_through", 0) < finalized
    )
    if (
        new_finalized
        or daily_due
        or needs_history
        or (last_full and last_full.get("histories_complete") is False)
    ):
        refreshed = asyncio.run(Official(store).ingest(histories=True, auxiliary=False))
        market = store.get("market")
        if refreshed.get("histories_complete") is False:
            return {
                "at": now(),
                "status": "Incomplete finalized player-history refresh; accepted models and forecasts retained",
                "history_successes": refreshed.get("history_successes"),
                "history_requests": refreshed.get("history_requests"),
                "errors": refreshed.get("errors", []),
                "training_published": False,
            }
    from .workload import refresh_workload_due

    refresh_workload_due(store)
    if not active:
        return {
            "at": now(),
            "status": "Training required; no accepted numerical model is installed",
            "training_published": False,
            "market_id": store.get("market")["id"],
        }
    needs_refit = bool(active) and training_needed(store, active)
    training_warning = None
    if needs_refit:
        report_phase("Learning newly finalized results")
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
        from .training import train_current

        try:
            train_current(store)
        except (ValueError, RuntimeError) as error:
            from .settings import redact

            training_warning = redact(error)[:500]
            store.put(
                "model_update_attempt",
                {
                    "at": now(),
                    "status": "failed",
                    "decision_model_id": active["id"],
                    "reason": training_warning,
                },
            )
    elif store.list("decision_model", 1):
        from .training import train_current

        train_current(store, refit=False)
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
    report_phase("Generating updated forecasts")
    result = forecast(store, event, 38, 4000)
    if training_warning:
        store.put(
            "model_update_warning",
            {
                "at": now(),
                "message": training_warning,
                "forecast_id": result["id"],
                "decision_model_id": store.get("decision_model")["id"],
            },
        )
    if not plan:
        return {
            "at": now(),
            "status": "Data, models and forecasts updated; optimize the plan separately",
            "training_warning": training_warning,
            "forecast_id": result["id"],
            "plan_optimized": False,
            "account_sync": synchronization,
            "account_changed": synchronization.get("account_changed", False),
        }
    report_phase("Optimizing transfer and chip strategy")
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
        with locked_state(store):
            from .readiness import selection_is_current

            published = selection_is_current(store, plan)
            if published:
                store.put("selected_plan", {"plan_id": plan["id"], "at": now()})
    except ValueError as error:
        from .optimization import hold_plan

        fallback = hold_plan(store, state["id"], result["id"], 19 if event <= 19 else 38)
        with locked_state(store):
            from .readiness import selection_is_current

            fallback_current = selection_is_current(store, fallback)
            if fallback_current:
                store.put("selected_plan", {"plan_id": fallback["id"], "at": now()})
        return {
            "at": now(),
            "status": "Updated forecast; planning requires review",
            "error": str(error),
            "plan_id": fallback["id"],
            "forecast_id": result["id"],
            "stale": not fallback_current,
            "fallback": "Validated hold policy retained; inspect the reported planning error before acting",
            "account_sync": synchronization,
            "account_changed": synchronization.get("account_changed", False),
            "consensus_comparison": consensus_status,
        }
    if not published:
        return {
            "at": now(),
            "status": "Completed for superseded inputs; refresh again",
            "stale": True,
            "plan_id": plan["id"],
            "forecast_id": result["id"],
            "account_sync": synchronization,
            "account_changed": synchronization.get("account_changed", False),
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
        "training_warning": training_warning,
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
        try:
            record("shots", asyncio.run(collect_shots(store)))
        except (ValueError, httpx.HTTPError) as error:
            stages.append(
                {
                    "stage": "shots",
                    "status": "Source unavailable; retained cached evidence or missing-input fallback",
                    "error": str(error)[:180],
                }
            )
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
    shots = store.get("shot_archive")["records"] if store.list("shot_archive", 1) else []
    groups = {f"{r['season']}-GW{r['event']:02d}" for r in shots}
    shot_splits = (
        any(g < "2024/25-GW25" for g in groups),
        any("2024/25-GW25" <= g < "2025/26-GW01" for g in groups),
        any(g >= "2025/26-GW01" for g in groups),
    )
    if all(shot_splits):
        record("temporal", validate_temporal(store))
        record("attack", train_attack(store))
        record("luck_diagnostic", luck_report(store))
    else:
        stages.append(
            {
                "stage": "shot_dependent_fits",
                "status": "Unavailable: verified shot history does not cover all chronological splits; retained accepted models and official-data forecasts",
                "records": len(shots),
                "chronological_splits_present": shot_splits,
            }
        )
    record("score_process", train_score_process(store))
    record("market_constraints", calibrate_odds(store))
    report = {
        "at": now(),
        "stages": stages,
        "scope": "Complete model fits and later-deadline gates; unavailable optional data never becomes a fabricated signal",
    }
    report["id"] = store.put("football_training", report)
    return report
