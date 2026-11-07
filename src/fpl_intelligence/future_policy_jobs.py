"""Validate the connected expert without relabelling old reference results."""

from __future__ import annotations

import polars as pl

from .fixture_horizon import build_data
from .future_policy_replay import run_future_policy
from .future_policy_verification import verify_future_policy
from .horizon_trial import fitted
from .storage import now


def run_future_suite(store, *, worlds=4000, seconds=60, periods=None):
    active = store.get("decision_model")
    expert = store.get("future_horizon_model", active["future_horizon_model_id"])
    accepted = store.get("future_policy_replay", expert["acceptance_id"])
    if not accepted["policy_gate_passed"]:
        raise ValueError("The connected future expert has no accepted policy replay")
    verify_future_policy(store, accepted)
    market = store.get("market")
    last = max((e["id"] for e in market["events"] if e.get("data_checked")), default=0)
    periods = periods or [(market["season"], list(range(1, last + 1)))]
    reports = []
    baseline = store.get("pipeline_replay")
    same_budget = baseline["protocol"]["worlds"] == worlds and baseline["protocol"]["seconds"] == seconds
    available = {(r["season"], r["event"]) for r in baseline["weeks"]}
    if not same_budget or any((s, e) not in available for s, events in periods for e in events):
        import asyncio

        from .consensus import prepare_inputs
        from .pipeline_replay import run_pipeline_replay

        asyncio.run(prepare_inputs(store))
        baseline = run_pipeline_replay(
            store,
            worlds=worlds,
            seconds=seconds,
            reference_family_only=True,
            resume_protocol_id=baseline["protocol_id"] if same_budget else None,
        )
    for season, events in periods:
        if season == accepted["season"] and events == [r["event"] for r in accepted["weeks"]]:
            reports.append(accepted)
            continue
        if season != market["season"] or not events:
            raise ValueError(
                "The accepted future-policy suite supports its prior window and completed current deadlines"
            )
        prior = store.get("future_policy_diagnostic") if store.list("future_policy_diagnostic", 1) else None
        if (
            prior
            and prior["season"] == season
            and [r["event"] for r in prior["weeks"]] == events
            and prior["protocol"]["refinement_id"] == expert["refinement_id"]
            and prior["protocol"]["worlds"] == worlds
            and prior["protocol"]["seconds"] == seconds
        ):
            verify_future_policy(store, prior)
            reports.append(prior)
            continue
        refinement = store.get("horizon_refinement", expert["refinement_id"])
        family = refinement["selected"].rsplit("_", 1)[0]
        directory = store.artifact_path(refinement["directory"] + "/protocol.json").parent
        fitted(store, pl.read_parquet(build_data(store)), season + "-GW01", family, directory)
        reports.append(run_future_policy(store, season=season, events=events, worlds=worlds, seconds=seconds))
    body = {
        "at": now(),
        "decision_model_id": active["id"],
        "future_horizon_model_id": expert["id"],
        "acceptance_id": accepted["id"],
        "matches_current_model_composition": not bool(active.get("role_minutes_model_id")),
        "evaluated_role_minutes_model_id": None,
        "matches_current_search_policy": False,
        "current_policy_acceptance": False,
        "active_role_minutes_model_id": active.get("role_minutes_model_id"),
        "scope": "Historical future-expert composition: immediate reference refitted at each deadline, future expert frozen before the evaluation season. Excludes the later role-minutes head and unlimited-search policy; do not treat this receipt as acceptance of the complete current policy.",
        "summaries": {
            r["season"]: {
                "candidate_points": r["candidate_points"],
                "baseline_points": r["baseline_points"],
                "report_id": r["id"],
            }
            for r in [accepted, *reports]
        },
    }
    return body | {"id": store.put("future_policy_suite", body)}
