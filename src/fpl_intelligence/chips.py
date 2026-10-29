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
