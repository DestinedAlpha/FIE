"""Explicit policy objectives on shared outcome worlds and visible EV constraints."""

from __future__ import annotations

import numpy as np

SELECTION_VERSION = "paid-transfer-evidence-v2"


def stale_paid_evidence(plan, receipt):
    """Invalidate a persisted expected-points paid recommendation after upgrading."""
    selection = receipt.get("selection", {})
    return bool(
        plan["ledger"][0]["hit"] > 0
        and selection.get("objective") == "expected_points"
        and selection.get("selected_plan_id") == plan["id"]
        and (
            receipt.get("selection_version") != SELECTION_VERSION
            or not (receipt.get("immediate_hit_evidence") or {}).get("hit_supported_beyond_search_error")
        )
    )


def source_forecast_review(store, plan, forecast):
    """Expose disagreement without treating published EP or fitness as start certainty."""
    from .ingestion import load_market

    if not store.list("market", 1):
        return []
    players, market = load_market(store, plan.get("market_id"))
    first = plan["ledger"][0]
    ids = {r["player_id"] for kind in ("sell", "buy") for r in first.get(kind, [])}
    projected = {r["player_id"]: r for r in forecast.get("players", []) if r["event"] == first["event"]}
    teams = {t["id"]: t["short_name"] for t in market["teams"]}
    result = []
    for identity in sorted(ids):
        player, row = players[identity], projected.get(identity)
        if row is None:
            continue
        published = player.raw.get("ep_next")
        completed = sum(
            bool(f.get("finished")) and player.team in (f["team_h"], f["team_a"]) for f in market["fixtures"]
        )
        fixture = [
            teams[f["team_a"] if f["team_h"] == player.team else f["team_h"]]
            + (" (H)" if f["team_h"] == player.team else " (A)")
            for f in market["fixtures"]
            if f.get("event") == first["event"] and player.team in (f["team_h"], f["team_a"])
        ]
        result.append(
            {
                "player_id": identity,
                "name": player.web_name,
                "fixtures": fixture,
                "model_points": row["mean"],
                "model_minutes": row["minutes_mean"],
                "model_start_probability": row["p_start"],
                "official_next_points": float(published)
                if published not in (None, "") and first["event"] == forecast["start_event"]
                else None,
                "official_status": player.status,
                "official_availability": player.chance,
                "season_starts": player.raw.get("starts"),
                "completed_club_fixtures": completed,
            }
        )
    return result


def paid_transfer_evidence(candidate, alternative, difference, no_hit_search):
    """Separate numerical sampling error from unfinished counterfactual search.

    The solver allowance is a conservative sensitivity in the additive surrogate,
    not a confidence interval for football or a bound on nonlinear autosubs.
    Unknown/conditional search bounds cannot justify an irreversible points cost.
    """
    for other in (alternative, no_hit_search):
        if other and any(other.get(k) != candidate.get(k) for k in ("state_id", "forecast_id", "end_event")):
            raise ValueError("Paid transfer evidence requires identical frozen inputs")
    if alternative["ledger"][0]["hit"] > 0 or (no_hit_search and no_hit_search["ledger"][0]["hit"] > 0):
        raise ValueError("The no-hit counterfactual charges a new hit")
    delta = np.asarray(difference, dtype=float)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("Paid transfer evidence needs paired finite outcome worlds")
    error = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    interval = [float(delta.mean() - 1.96 * error), float(delta.mean() + 1.96 * error)]
    solver = no_hit_search.get("solver", {}) if no_hit_search else {}
    bound, objective = solver.get("best_bound"), solver.get("objective")
    complete_search = bool(
        no_hit_search
        and no_hit_search.get("first_week_hit_budget") == 0
        and "continuation_plan_id" not in solver
        and bound is not None
        and objective is not None
        and np.isfinite(bound)
        and np.isfinite(objective)
        and bound >= objective - 1e-6
    )
    allowance = None
    if complete_search:
        assert bound is not None and objective is not None
        allowance = max(0.0, float(bound - objective))
    supported = allowance is not None and interval[0] > allowance
    a = {r["event"]: r for r in alternative["ledger"]}
    trajectory, cumulative = [], 0.0
    for row in candidate["ledger"]:
        other = a[row["event"]]
        left, right = row.get("simulated_net_mean"), other.get("simulated_net_mean")
        if left is not None and right is not None:
            cumulative += left - right
            trajectory.append(
                {
                    "event": row["event"],
                    "candidate_net": left,
                    "alternative_net": right,
                    "cumulative_net_advantage": cumulative,
                }
            )
    return {
        "candidate_plan_id": candidate["id"],
        "no_hit_plan_id": alternative["id"],
        "net_mean_gain": float(delta.mean()),
        "paired_mc_interval_95": interval,
        "hit_supported_beyond_mc_error": interval[0] > 0,
        "no_hit_search_plan_id": no_hit_search.get("id") if no_hit_search else None,
        "no_hit_search_allowance": allowance,
        "hit_supported_beyond_search_error": supported,
        "weekly_tradeoff": trajectory,
        "scope": "Paired sampling error and unfinished no-hit additive-search allowance are separate. Forecast error remains additional; this is not a statistical confidence interval for the true transfer value.",
        "reason": "Paid transfer has numerical support beyond the unfinished no-hit search"
        if supported
        else "No-new-hit policy retained: paid-transfer advantage is unresolved after the no-hit search allowance"
        if allowance is not None
        else "No-new-hit policy retained: a full no-hit search bound is unavailable",
    }


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
    no_hit_search = None
    comparison_seconds = min(
        300.0, max(60.0, max(p.get("solver", {}).get("runtime_seconds", 0) for p in plans))
    )
    plans = [*plans, hold_plan(store, reference["state_id"], forecast["id"], reference["end_event"])]
    for item in store.list_matching(
        "plan", {"state_id": reference["state_id"], "forecast_id": forecast["id"]}, 500
    ):
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
    if any(p["ledger"][0]["hit"] > 0 for p in plans):
        no_hit_search = optimize(
            store,
            reference["state_id"],
            forecast["id"],
            reference["end_event"],
            comparison_seconds,
            first_week_hit_budget=0,
            chips=fixed_chips,
            max_transfers=max_transfers,
        )
        plans.append(no_hit_search)
        plans = list({p["id"]: p for p in plans}.values())
    outcomes = score_plans(store, plans, forecast)
    rows = [
        {
            "plan_id": p["id"],
            "expected_net_points": float(v.mean()),
            "cvar_points": cvar(v, tail_fraction),
            "immediate_net_points": p["ledger"][0].get("simulated_net_mean"),
            "immediate_hit": p["ledger"][0]["hit"],
        }
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
            hit_evidence = paid_transfer_evidence(plans[index], plans[alternative], difference, no_hit_search)
            hit_evidence["source_forecast_review"] = source_forecast_review(store, plans[index], forecast)
            if not hit_evidence["hit_supported_beyond_search_error"]:
                index = alternative
                selection.update(
                    selected_plan_id=plans[index]["id"],
                    objective_value=float(outcomes[index].mean()),
                    actual_ev_sacrifice=max(r["expected_net_points"] for r in rows)
                    - float(outcomes[index].mean()),
                    reason=hit_evidence["reason"],
                )
    receipt = {
        "at": now(),
        "selection_version": SELECTION_VERSION,
        "forecast_id": forecast["id"],
        "state_id": reference["state_id"],
        "selection": selection,
        "candidates": rows,
        "immediate_hit_evidence": hit_evidence,
        "no_hit_search_seconds": comparison_seconds,
    }
    receipt["id"] = store.put("policy_selection", receipt)
    return plans[index], receipt
