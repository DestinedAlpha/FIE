"""Whole-policy chip counterfactuals, rescored in identical match worlds."""

from __future__ import annotations

import numpy as np

from .optimization import optimize
from .simulation import score_plan
from .storage import Store, now


def compare_chips(store: Store, plan_id: str, seconds: float = 30) -> dict:
    base = store.get("plan", plan_id)
    forecast = store.get("forecast", base["forecast_id"])
    original = score_plan(store, base, forecast)
    rows = []
    for chip in ("freehit", "bboost"):
        if not any(r["chip"] == chip for r in base["ledger"]):
            continue
        without = optimize(
            store, base["state_id"], base["forecast_id"], base["end_event"], seconds, disabled_chips=(chip,)
        )
        alternative = score_plan(store, without, forecast)
        delta = original - alternative
        standard = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        rows.append(
            {
                "chip": chip,
                "chosen_event": next(r["event"] for r in base["ledger"] if r["chip"] == chip),
                "without_plan_id": without["id"],
                "baseline_mean": float(original.mean()),
                "without_mean": float(alternative.mean()),
                "whole_policy_mean_delta": float(delta.mean()),
                "paired_mc_interval_95": [
                    float(delta.mean() - 1.96 * standard),
                    float(delta.mean() + 1.96 * standard),
                ],
                "baseline_solver_gap": base["solver"]["gap"],
                "counterfactual_solver_gap": without["solver"]["gap"],
                "counterfactual_validation": without["validation"],
            }
        )
    result = {
        "at": now(),
        "base_plan_id": plan_id,
        "comparisons": rows,
        "scope": "Independent whole-policy re-optimization; includes transfer preparation, cleanup, bank and ordinary autosubs",
        "uncertainty": "Monte Carlo interval excludes forecast and solver-incumbent uncertainty; compare reported gaps",
    }
    result["id"] = store.put("chip_counterfactual", result)
    return result


def compare_immediate_hit(store: Store, plan_id: str, seconds: float = 90):
    """Show the full-horizon tradeoff against a policy taking no new hit now."""
    base = store.get("plan", plan_id)
    forecast = store.get("forecast", base["forecast_id"])
    alternative = optimize(
        store, base["state_id"], base["forecast_id"], base["end_event"], seconds, first_week_hit_budget=0
    )
    first = alternative["ledger"][0]
    if first["hit"] > 0:
        raise ValueError("The no-hit counterfactual violates its immediate cost constraint")
    # A timeout incumbent is not a counterfactual optimum. Reuse stronger
    # already-validated feasible policies on exactly these frozen inputs.
    candidates = [alternative]
    for item in store.list("plan", 1000):
        known = store.get("plan", item["id"])
        ledger = known.get("ledger", [])
        if (
            known.get("state_id") == base["state_id"]
            and known.get("forecast_id") == base["forecast_id"]
            and ledger
            and ledger[-1]["event"] == base["end_event"]
            and ledger[0]["hit"] <= 0
            and known.get("validation", {}).get("legal")
        ):
            candidates.append(known)
    candidates = list({plan["id"]: plan for plan in candidates}.values())
    alternative = max(candidates, key=lambda plan: score_plan(store, plan, forecast).mean())
    first = alternative["ledger"][0]
    delta = score_plan(store, base, forecast) - score_plan(store, alternative, forecast)
    error = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    result = {
        "at": now(),
        "base_plan_id": plan_id,
        "alternative_plan_id": alternative["id"],
        "immediate_hit": base["ledger"][0]["hit"],
        "alternative_immediate_hit": first["hit"],
        "alternative_immediate_chip": first["chip"],
        "base_mean": base["simulation"]["mean"],
        "alternative_mean": alternative["simulation"]["mean"],
        "net_mean_advantage": float(delta.mean()),
        "paired_mc_interval_95": [float(delta.mean() - 1.96 * error), float(delta.mean() + 1.96 * error)],
        "base_gap": base["solver"]["gap"],
        "alternative_gap": alternative.get("solver", {}).get("gap"),
        "feasible_counterfactuals_compared": len(candidates),
        "uncertainty": "Net of hits, same horizon and match worlds; model uncertainty and solver gaps are not inside this Monte Carlo interval",
    }
    result["id"] = store.put("hit_comparison", result)
    return result
