"""Whole-policy chip counterfactuals, rescored in identical match worlds."""

from __future__ import annotations

import numpy as np

from .optimization import optimize
from .simulation import score_plan
from .storage import Store, now


def compare_chips(store: Store, plan_id: str, seconds: float = 30) -> dict:
    """Test every available chip entitlement/date against a complete no-chip policy.

    Other chips remain jointly optimized. A second-half entitlement is independent
    of the first; disabling one period must not erase the other. Each solve has a
    stated time budget, and its feasible incumbent is never called an optimum.
    """
    from .domain import TeamState
    from .ingestion import load_market
    from .rules import Rules

    if seconds <= 0:
        raise ValueError("Chip comparison requires a positive per-solve budget")
    from .search_control import effective_time_limit

    base = store.get("plan", plan_id)
    forecast = store.get("forecast", base["forecast_id"])
    state = TeamState.model_validate(
        {k: v for k, v in store.get("state", base["state_id"]).items() if k != "id"}
    )
    _, market = load_market(store, base["market_id"])
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    original = score_plan(store, base, forecast)
    known = [base]
    for item in store.list("plan", 1000):
        plan = store.get("plan", item["id"])
        if (
            plan.get("state_id") == base["state_id"]
            and plan.get("forecast_id") == forecast["id"]
            and plan.get("end_event") == base["end_event"]
            and plan.get("validation", {}).get("legal")
        ):
            known.append(plan)

    def strongest(solved, permitted):
        known.append(solved)
        return max((p for p in known if permitted(p)), key=lambda p: p["simulation"]["mean"])

    def difference(plan):
        values = score_plan(store, plan, forecast)
        delta = values - original
        error = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        return {
            "mean": float(values.mean()),
            "delta_vs_selected": float(delta.mean()),
            "paired_mc_interval_95": [float(delta.mean() - 1.96 * error), float(delta.mean() + 1.96 * error)],
        }

    rows, calendar = [], []
    for chip in ("wildcard", "freehit", "bboost", "3xc"):
        windows = sorted(
            {
                rules.chip_window(chip, t)
                for t in range(state.event, base["end_event"] + 1)
                if any(c["name"] == chip and c["start_event"] <= t <= c["stop_event"] for c in rules.chips)
            }
        )
        for lo, hi in windows:
            if any(c.chip == chip and lo <= c.event <= hi for c in state.used_chips):
                continue
            dates = [
                t
                for t in range(max(lo, state.event), min(hi, base["end_event"]) + 1)
                if not (
                    chip == "freehit"
                    and t == state.event
                    and any(c.chip == chip and c.event == t - 1 for c in state.used_chips)
                )
                and not (t == state.event and state.active_chip not in (None, chip))
                and not (
                    chip == "freehit"
                    and t == state.event
                    and state.transfers_made
                    and (state.deadline_lots is None or state.deadline_bank is None)
                )
            ]
            if not dates or (state.active_chip == chip and lo <= state.event <= hi):
                continue
            blocked = {t: (chip,) for t in range(max(lo, state.event), min(hi, base["end_event"]) + 1)}
            without = strongest(
                optimize(
                    store,
                    base["state_id"],
                    forecast["id"],
                    base["end_event"],
                    seconds,
                    blocked_chips=blocked,
                    use_decision_critic=False,
                ),
                lambda p, c=chip, a=lo, b=hi: (
                    not any(r["chip"] == c and a <= r["event"] <= b for r in p["ledger"])
                ),
            )
            delta = original - score_plan(store, without, forecast)
            standard = float(delta.std(ddof=1) / np.sqrt(len(delta)))
            rows.append(
                {
                    "chip": chip,
                    "window": [lo, hi],
                    "chosen_event": next(
                        (r["event"] for r in base["ledger"] if r["chip"] == chip and lo <= r["event"] <= hi),
                        None,
                    ),
                    "without_plan_id": without["id"],
                    "baseline_mean": float(original.mean()),
                    "without_mean": without["simulation"]["mean"],
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
            for event in dates:
                forced = strongest(
                    optimize(
                        store,
                        base["state_id"],
                        forecast["id"],
                        base["end_event"],
                        seconds,
                        pin_chips={event: chip},
                        use_decision_critic=False,
                    ),
                    lambda p, c=chip, t=event: any(r["chip"] == c and r["event"] == t for r in p["ledger"]),
                )
                calendar.append(
                    {
                        "chip": chip,
                        "event": event,
                        "window": [lo, hi],
                        "plan_id": forced["id"],
                        **difference(forced),
                        "solver": forced["solver"],
                        "validation": forced["validation"],
                        "all_chip_dates": [
                            {"event": r["event"], "chip": r["chip"]} for r in forced["ledger"] if r["chip"]
                        ],
                    }
                )
    best = max(known, key=lambda p: p["simulation"]["mean"])
    result = {
        "at": now(),
        "base_plan_id": plan_id,
        "forecast_id": forecast["id"],
        "comparisons": rows,
        "calendar": calendar,
        "dates_evaluated": len(calendar),
        "best_candidate_plan_id": best["id"],
        "best_candidate_mean": best["simulation"]["mean"],
        "per_solve_seconds": effective_time_limit(seconds),
        "scope": "Every feasible date of each unused entitlement inside the displayed horizon; each forced-date policy jointly optimizes the other chips, preparation and cleanup. Joint MIP covers combinations without enumerating their cross-product.",
        "uncertainty": "Monte Carlo intervals exclude forecast and solver-incumbent uncertainty; every date retains its actual solver diagnostics. Remaining season dates require a season-end planning horizon.",
    }
    result["id"] = store.put("chip_counterfactual", result)
    return result


def rebase_chip_comparison(store: Store, comparison: dict, plan_id: str) -> dict:
    """Keep completed searches when final policy selection finds a better route."""
    import copy

    result = copy.deepcopy(comparison)
    old = store.get("plan", comparison["base_plan_id"])
    base = store.get("plan", plan_id)
    if any(old[key] != base[key] for key in ("state_id", "forecast_id", "end_event")):
        raise ValueError("Chip comparisons cannot be rebased onto different frozen inputs")
    forecast = store.get("forecast", base["forecast_id"])
    original = score_plan(store, base, forecast)
    for row in result["comparisons"]:
        alternative = store.get("plan", row["without_plan_id"])
        delta = original - score_plan(store, alternative, forecast)
        error = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        lo, hi = row["window"]
        row.update(
            chosen_event=next(
                (r["event"] for r in base["ledger"] if r["chip"] == row["chip"] and lo <= r["event"] <= hi),
                None,
            ),
            baseline_mean=float(original.mean()),
            whole_policy_mean_delta=float(delta.mean()),
            baseline_solver_gap=base["solver"]["gap"],
            paired_mc_interval_95=[float(delta.mean() - 1.96 * error), float(delta.mean() + 1.96 * error)],
        )
    for row in result["calendar"]:
        delta = score_plan(store, store.get("plan", row["plan_id"]), forecast) - original
        error = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        row.update(
            delta_vs_selected=float(delta.mean()),
            paired_mc_interval_95=[float(delta.mean() - 1.96 * error), float(delta.mean() + 1.96 * error)],
        )
    result.update(at=now(), base_plan_id=plan_id, search_report_id=comparison["id"])
    result.pop("id", None)
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
