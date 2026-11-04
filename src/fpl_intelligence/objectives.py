"""Explicit policy objectives on shared outcome worlds and visible EV constraints."""

from __future__ import annotations

import numpy as np


def select_policy(rows, mode="expected_points", sacrifice_budget=0, tail_fraction=0.1):
    if not rows or sacrifice_budget < 0 or not 0 < tail_fraction <= 1:
        raise ValueError("Objective requires candidates and lawful risk/EV settings")
    if mode not in ("expected_points", "beat_both", "top_one_percent", "cvar"):
        raise ValueError("Unknown policy objective")
    best = max(float(r["expected_net_points"]) for r in rows)
    eligible = [
        r
        for r in rows
        if r["expected_net_points"] >= best - (0 if mode == "expected_points" else sacrifice_budget) - 1e-9
    ]
    key = {
        "expected_points": "expected_net_points",
        "beat_both": "robust_joint_win",
        "top_one_percent": "top_one_percent_probability",
        "cvar": "cvar_points",
    }[mode]
    if any(r.get(key) is None for r in eligible):
        raise ValueError(f"The {mode} objective lacks its required outcome analysis")
    chosen = max(eligible, key=lambda r: (r[key], r["expected_net_points"]))
    return {
        "selected_plan_id": chosen["plan_id"],
        "objective": mode,
        "objective_value": float(chosen[key]),
        "sacrifice_budget": sacrifice_budget,
        "actual_ev_sacrifice": best - float(chosen["expected_net_points"]),
        "eligible_policies": len(eligible),
        "tail_fraction": tail_fraction if mode == "cvar" else None,
    }


def cvar(points, tail_fraction=0.1):
    values = np.sort(np.asarray(points, dtype=float))
    if not len(values) or not 0 < tail_fraction <= 1:
        raise ValueError("Invalid outcome-tail fraction")
    # Fractional boundary mass avoids silently changing the requested quantile.
    mass = len(values) * tail_fraction
    n = int(np.floor(mass))
    remainder = mass - n
    return float((values[:n].sum() + (values[n] * remainder if remainder and n < len(values) else 0)) / mass)


def choose_recommendation(
    store,
    plans,
    forecast,
    mode="expected_points",
    sacrifice_budget=0,
    tail_fraction=0.1,
    fixed_chips=None,
    max_transfers=15,
    first_week_hit_budget=None,
):
    """Always compare hold and an independently solved no-new-hit alternative."""
    from .optimization import hold_plan, optimize
    from .simulation import score_plans
    from .storage import now

    reference = plans[0]
    comparison_seconds = min(
        300.0, max(60.0, max(p.get("solver", {}).get("runtime_seconds", 0) for p in plans))
    )
    plans = [*plans, hold_plan(store, reference["state_id"], forecast["id"], reference["end_event"])]
    if any(p["ledger"][0]["hit"] > 0 for p in plans):
        plans.append(
            optimize(
                store,
                reference["state_id"],
                forecast["id"],
                reference["end_event"],
                comparison_seconds,
                first_week_hit_budget=0,
                chips=fixed_chips,
                max_transfers=max_transfers,
            )
        )
    for item in store.list("plan", 500):
        p = store.get("plan", item["id"])
        if (
            p.get("state_id") == reference["state_id"]
            and p.get("forecast_id") == forecast["id"]
            and p.get("end_event") == reference["end_event"]
            and p.get("validation", {}).get("legal")
        ):
            plans.append(p)
    plans = list({p["id"]: p for p in plans}.values())
    plans = [
        p
        for p in plans
        if all(
            (fixed_chips is None or row["chip"] == fixed_chips.get(row["event"]))
            and (row["chip"] in ("wildcard", "freehit") or len(row["buy"]) <= max_transfers)
            for row in p["ledger"]
        )
        and (first_week_hit_budget is None or p["ledger"][0]["hit"] <= first_week_hit_budget)
    ]
    if not plans:
        raise ValueError("No candidate satisfies the requested transfer/chip constraints")
    outcomes = score_plans(store, plans, forecast)
    rows = [
        {"plan_id": p["id"], "expected_net_points": float(v.mean()), "cvar_points": cvar(v, tail_fraction)}
        for p, v in zip(plans, outcomes)
    ]
    if mode == "beat_both":
        from .rivals import compare

        state = store.get("state", reference["state_id"])
        result = compare(
            store,
            plans,
            forecast["id"],
            store.get("league")["id"],
            state["points"],
            state["entry_id"],
            sacrifice_budget,
        )
        return store.get("plan", result["selected_plan_id"]), result
    if mode == "top_one_percent":
        from .population import population_frontier

        result = population_frontier(store, plans, forecast)
        by_id = {r["plan_id"]: r for r in result["candidates"]}
        rows = [{**r, **by_id[r["plan_id"]]} for r in rows]
    selection = select_policy(rows, mode, sacrifice_budget, tail_fraction)
    index = next(i for i, p in enumerate(plans) if p["id"] == selection["selected_plan_id"])
    hit_evidence = None
    if mode == "expected_points" and plans[index]["ledger"][0]["hit"] > 0:
        nonhit = [i for i, p in enumerate(plans) if p["ledger"][0]["hit"] <= 0]
        if nonhit:
            alternative = max(nonhit, key=lambda i: float(outcomes[i].mean()))
            difference = outcomes[index] - outcomes[alternative]
            error = float(difference.std(ddof=1) / np.sqrt(len(difference)))
            interval = [float(difference.mean() - 1.96 * error), float(difference.mean() + 1.96 * error)]
            hit_evidence = {
                "candidate_plan_id": plans[index]["id"],
                "no_hit_plan_id": plans[alternative]["id"],
                "net_mean_gain": float(difference.mean()),
                "paired_mc_interval_95": interval,
                "hit_supported_beyond_mc_error": interval[0] > 0,
                "scope": "Paired simulation error only; forecast and solver uncertainty remain additional",
            }
            if interval[0] <= 0:
                index = alternative
                selection.update(
                    selected_plan_id=plans[index]["id"],
                    objective_value=float(outcomes[index].mean()),
                    actual_ev_sacrifice=max(r["expected_net_points"] for r in rows)
                    - float(outcomes[index].mean()),
                    reason="No-new-hit policy retained: extra hit benefit is unresolved within Monte Carlo error",
                )
    receipt = {
        "at": now(),
        "forecast_id": forecast["id"],
        "state_id": reference["state_id"],
        "selection": selection,
        "candidates": rows,
        "immediate_hit_evidence": hit_evidence,
        "no_hit_search_seconds": comparison_seconds,
    }
    receipt["id"] = store.put("policy_selection", receipt)
    return plans[index], receipt
