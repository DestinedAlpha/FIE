"""Revealed league membership, effective exposure and shared-world policy tests."""

from __future__ import annotations

import asyncio
from collections import Counter
from itertools import product
from typing import Any

import httpx
import numpy as np

from .domain import Action, ChipUse, Lot, TeamState
from .ingestion import Official, load_market
from .optimization import greedy_bundle, hold_plan, optimize
from .rules import Rules, best_lineup, transition
from .simulation import score_plan
from .storage import Store, now


async def discover(store: Store, league_id: int, own_entry: int) -> dict:
    _, market = load_market(store)
    current = next(e["id"] for e in market["events"] if e.get("is_current"))
    adapter = Official(store)
    members = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        for page in range(1, 101):
            league = await adapter.fetch(
                client,
                f"leagues-classic/{league_id}/standings/?page_standings={page}",
                f"league_{league_id}_{page}",
            )
            if not league:
                break
            members.extend(league["standings"]["results"])
            if not league["standings"]["has_next"]:
                break
        for member in members:
            identity = member["entry"]
            history, _ = await asyncio.gather(
                adapter.fetch(client, f"entry/{identity}/history/", f"entry_history_{identity}"),
                adapter.fetch(client, f"entry/{identity}/transfers/", f"transfers_{identity}"),
            )
            from .account_sync import revealed_pick_events

            try:
                events = revealed_pick_events(history or {}, current)
            except ValueError as error:
                adapter.errors.append({"entry": identity, "error": str(error)})
                continue
            await asyncio.gather(
                *(
                    adapter.fetch(client, f"entry/{identity}/event/{g}/picks/", f"picks_{identity}_{g}")
                    for g in events
                )
            )
    result = {
        "league_id": league_id,
        "own_entry": own_entry,
        "captured_at": now(),
        "released_event": current,
        "members": [
            {"entry": r["entry"], "name": r["player_name"], "team": r["entry_name"], "points": r["total"]}
            for r in members
        ],
        "retrieved": len(members),
        "errors": adapter.errors,
        "population": "entire supplied mini-league, not an elite-manager cohort",
    }
    result["id"] = store.put("league", result)
    return result


def effective_ownership(store: Store, league: dict) -> dict:
    exposure: dict[int, float] = {}
    retrieved = 0
    for member in league["members"]:
        try:
            picks, _ = store.latest(f"picks_{member['entry']}_{league['released_event']}")
        except ValueError:
            continue
        retrieved += 1
        for p in picks["picks"]:
            # Released multiplier already includes autosubs and captain fallback
            # when finalized; before finalization it is provisional exposure.
            exposure[p["element"]] = exposure.get(p["element"], 0) + p["multiplier"]
    return {
        "cohort": league["population"],
        "retrieved": retrieved,
        "at": now(),
        "event": league["released_event"],
        "eo_percent": {str(p): 100 * v / max(1, retrieved) for p, v in exposure.items()},
    }


def rival_policy(
    store: Store, member: dict, league: dict, forecast: dict, end_event: int, style: str, score=True
) -> dict:
    players, frozen_market = load_market(store, forecast["market_id"])
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=frozen_market["source_hash"])[0])
    released = league["released_event"]
    picks, _ = store.latest(f"picks_{member['entry']}_{released}")
    history, _ = store.latest(f"entry_history_{member['entry']}")
    history = {
        **history,
        "current": [r for r in history["current"] if r["event"] <= released],
        "chips": [r for r in history["chips"] if r["event"] <= released],
    }
    if any(c["name"] == "freehit" and c["event"] == released for c in history["chips"]):
        picks, _ = store.latest(f"picks_{member['entry']}_{released - 1}")
    from .account_sync import free_transfers_after

    ft = free_transfers_after(history, released, rules)
    state = TeamState(
        season=forecast["season"],
        event=forecast["start_event"],
        name=member["team"],
        entry_id=member["entry"],
        bank=picks["entry_history"]["bank"],
        free_transfers=ft,
        grandfathered_clubs=[
            club
            for club, count in Counter(players[p["element"]].team for p in picks["picks"]).items()
            if count > rules.club_limit
        ],
        lots=[
            Lot(
                player_id=p["element"],
                sell_price=players[p["element"]].price,
                quote_market_price=players[p["element"]].price,
                provenance="illustrative_current_market_sell_quote",
            )
            for p in picks["picks"]
        ],
        used_chips=[
            ChipUse(chip=c["name"], event=c["event"])
            for c in history["chips"]
            if c["name"] in ("wildcard", "freehit", "bboost", "3xc")
        ],
        points=member["points"],
        known_at=league["captured_at"],
        status="public_deadline",
    )
    means = {(r["event"], r["player_id"]): r["mean"] for r in forecast["players"]}
    ledger = []
    learned = None
    if style.startswith("learned"):
        from .manager_policy import policy_models

        learned = policy_models(store, forecast)
        if forecast.get("manager_behavior_id"):
            learned[1]["__behavior"] = store.get("manager_behavior", forecast["manager_behavior_id"])
    for event in range(forecast["start_event"], end_event + 1):
        owned = [lot.player_id for lot in state.lots]
        chosen = Action()
        if learned is not None:
            from .manager_policy import rollout_action

            decision = rollout_action(
                store,
                member,
                state,
                players,
                rules,
                {p: means.get((event, p), 0) for p in players},
                history,
                int(style.removeprefix("learned")),
                event - forecast["start_event"],
                learned,
            )
            state = decision.state
            ledger.append(decision.ledger)
            continue
        if style.startswith("behavior"):
            maximum = int(style[-1]) if event == forecast["start_event"] else 1
            chosen = greedy_bundle(
                state, players, rules, {p: means.get((event, p), 0) for p in players}, maximum
            )
            owned = [p for p in owned if p not in chosen.sell] + chosen.buy
        elif style != "hold" and state.free_transfers:
            base = best_lineup(owned, {p: means.get((event, p), 0) for p in players}, players, rules)
            base_value = sum(means[event, p] for p in base.xi) + means[event, base.captain]
            candidates = sorted(
                (p for p in players if p not in owned and players[p].selectable),
                key=lambda p: means.get((event, p), 0),
                reverse=True,
            )[:80]
            best_gain = 0.5
            for old in owned:
                for new in candidates:
                    if players[old].position != players[new].position:
                        continue
                    if players[new].price > state.bank + players[old].price:
                        continue
                    trial = [p for p in owned if p != old] + [new]
                    if sum(players[p].team == players[new].team for p in trial) > 3:
                        continue
                    lineup = best_lineup(
                        trial, {p: means.get((event, p), 0) for p in players}, players, rules
                    )
                    value = sum(means[event, p] for p in lineup.xi) + means[event, lineup.captain]
                    if value - base_value > best_gain:
                        best_gain = value - base_value
                        chosen = Action(sell=[old], buy=[new])
            owned = [p for p in owned if p not in chosen.sell] + chosen.buy
        chosen.lineup = best_lineup(owned, {p: means.get((event, p), 0) for p in players}, players, rules)
        state, row = transition(state, chosen, players, rules)
        ledger.append(row)
    plan = {
        "ledger": ledger,
        "entry": member["entry"],
        "name": member["name"],
        "starting_points": member["points"],
        "style": style,
        "assumptions": [
            "Revealed prior-deadline squad; hidden transfers unknown",
            "Current market sale quotes and published bank are sensitivity assumptions",
            "Learned branches carry purchases, sales, chip use, bank and FT through exact transitions"
            if learned
            else "No future rival chip use in these hold/greedy scenarios",
        ],
    }
    if score:
        score_plan(store, plan, forecast)
    return plan


def compare(
    store: Store,
    plans: list[dict],
    forecast_id: str,
    league_id: str,
    own_points: int,
    own_entry: int,
    sacrifice_budget: float = 5,
) -> dict:
    forecast = store.get("forecast", forecast_id)
    league = store.get("league", league_id)
    rivals = [
        m
        for m in league["members"]
        if m["entry"] != own_entry and any(name in m["name"].lower() for name in ("ashwin", "raza"))
    ]
    if len(rivals) != 2:
        raise ValueError("Exactly two target rivals must be identified from actual league membership")
    end_event = max(r["event"] for r in plans[0]["ledger"])
    behavior = (
        store.get("manager_behavior", forecast.get("manager_behavior_id"))
        if store.list("manager_behavior", 1)
        else None
    )
    if behavior and behavior["event"] != forecast["start_event"]:
        behavior = None
    learned_policy = (
        store.get("manager_policy", forecast.get("manager_policy_id"))
        if store.list("manager_policy", 1)
        else None
    )
    styles = ["hold", "greedy"]
    weights = {}
    if behavior:
        from .manager_behavior import rival_count_probabilities

        weights = {
            member["entry"]: rival_count_probabilities(store, member["entry"], behavior) for member in rivals
        }
        styles += ["behavior" + str(i) for i in range(4)]
    if learned_policy:
        styles += ["learned" + str(i) for i in range(4)]
    from .rival_cache import rival_outcomes
    from .simulation import score_plans

    rival_worlds, rival_worlds_id = rival_outcomes(store, rivals, league, forecast, end_event, styles)
    own_scores = score_plans(store, plans, forecast)
    learned_in_objective = bool(
        learned_policy
        and behavior
        and behavior.get("promoted_over_frequency")
        and learned_policy["metrics"]["buy"]["promoted"]
        and learned_policy["metrics"]["sell"]["promoted"]
    )
    rows = []
    for plan, own_score in zip(plans, own_scores):
        outcomes = own_points + own_score
        probabilities: list[dict[str, Any]] = []
        ties = []
        for combination in product(("hold", "greedy"), repeat=2):
            a = rival_worlds[rivals[0]["entry"], combination[0]]
            b = rival_worlds[rivals[1]["entry"], combination[1]]
            win = (outcomes > a) & (outcomes > b)
            tie = ((outcomes == a) & (outcomes >= b)) | ((outcomes == b) & (outcomes >= a))
            probabilities.append(
                {
                    "rival_policies": list(combination),
                    "joint_strict_win": float(win.mean()),
                    "mc_standard_error": float(np.sqrt(win.mean() * (1 - win.mean()) / len(win))),
                    "joint_tie_mass": float(tie.mean()),
                }
            )
            ties.append(float(tie.mean()))
        behavior_probability = None
        if behavior:
            behavior_probability = 0.0
            for ca, cb in product(range(4), repeat=2):
                a = rival_worlds[rivals[0]["entry"], "behavior" + str(ca)]
                b = rival_worlds[rivals[1]["entry"], "behavior" + str(cb)]
                weight = weights[rivals[0]["entry"]][ca] * weights[rivals[1]["entry"]][cb]
                behavior_probability += weight * float(((outcomes > a) & (outcomes > b)).mean())
        learned_probabilities = (
            [
                float(
                    (
                        (outcomes > rival_worlds[rivals[0]["entry"], "learned" + str(a)])
                        & (outcomes > rival_worlds[rivals[1]["entry"], "learned" + str(b)])
                    ).mean()
                )
                for a, b in product(range(4), repeat=2)
            ]
            if learned_policy
            else []
        )
        legacy_robust = min(p["joint_strict_win"] for p in probabilities)
        robust = min([legacy_robust] + learned_probabilities) if learned_in_objective else legacy_robust
        rows.append(
            {
                "plan_id": plan["id"],
                "label": plan.get("label", "optimized"),
                "expected_net_points": plan["simulation"]["mean"],
                "policies": probabilities,
                "legacy_robust_joint_win": legacy_robust,
                "robust_joint_win": robust,
                "learned_policy_range": [min(learned_probabilities), max(learned_probabilities)]
                if learned_probabilities
                else None,
                "behavior_conditioned_joint_win": behavior_probability,
                "learned_policy_joint_win": float(np.mean(learned_probabilities))
                if learned_probabilities
                else None,
            }
        )
    from .objectives import select_policy

    selection = select_policy(rows, "beat_both", sacrifice_budget)
    result = {
        "created_at": now(),
        "forecast_id": forecast_id,
        "rivals": rivals,
        "starting_points": own_points,
        "candidate_frontier": rows,
        "sacrifice_budget": sacrifice_budget,
        "selected_plan_id": selection["selected_plan_id"],
        "selection": selection,
        "rival_worlds_id": rival_worlds_id,
        "learned_policy_in_selection": learned_in_objective,
        "robust_policy_set": "Hold/greedy plus all sixteen validated learned-branch combinations"
        if learned_in_objective
        else "Hold/greedy; unaccepted behavioral challengers excluded",
        "optimization_scope": "Robust joint probability among generated feasible policies, not a global probability optimum",
        "ties": "Reported separately; strict-win probability excludes ties. Exact final rank needs final transfer tie-breaks.",
        "shared_worlds": True,
        "manager_behavior_id": behavior["id"] if behavior else None,
        "manager_policy_id": learned_policy["id"] if learned_policy else None,
        "learned_policy_scope": "Four reproducible learned purchase/sale/count/chip trajectories per rival, sharing the football worlds; finite policy samples and conditional independence remain assumptions"
        if learned_policy
        else None,
        "behavior_assumptions": [
            "A top-1,000 conditional behavioral prior weights the first decision; calibration on these mini-league rivals is unproven and later actions follow an explicit greedy scenario",
            "Count 3+ is represented by at most three greedy transfers; future rival chips are not forecast",
            "Conditional independence of rival action types is an approximation; hold/greedy sensitivities remain visible",
        ]
        if behavior
        else [],
        "worlds": forecast["worlds"],
    }
    result["id"] = store.put("rival_comparison", result)
    return result


def current_comparison(store: Store):
    """Compare the current recommendation and hold using the same forecast worlds."""
    from .readiness import selection_is_current

    state = store.get("state")
    forecast = store.get("forecast")
    selected = store.get("plan", store.get("selected_plan")["plan_id"])
    if not selection_is_current(store, selected):
        raise ValueError("Refresh the current recommendation before comparing rivals")
    selected = {**selected, "label": "Current recommendation"}
    hold = hold_plan(store, state["id"], forecast["id"], selected["end_event"])
    hold["label"] = "Hold current squad"
    return compare(
        store,
        [selected] if selected["id"] == hold["id"] else [selected, hold],
        forecast["id"],
        store.get("league")["id"],
        state["points"],
        state["entry_id"],
    )


def case_comparison(store: Store, state_id: str, forecast_id: str, time_limit=15) -> dict:
    from .ingestion import resolve_name

    players, _ = load_market(store)
    state = store.get("state", state_id)
    if state["event"] != 9:
        raise ValueError("Original GW9–19 case requires a GW9 state")
    pedro = resolve_name("João Pedro", 4, players)
    plans = []
    baseline = optimize(store, state_id, forecast_id, 19, time_limit)
    baseline["label"] = "E · independent joint chip/transfer plan"
    plans.append(baseline)
    hold = hold_plan(store, state_id, forecast_id, 19)
    plans.append(hold)
    for label, fh, sale in [("A", 9, 10), ("B", 12, 9), ("C", 15, 9), ("D", 19, 9)]:
        plan = optimize(
            store,
            state_id,
            forecast_id,
            19,
            time_limit,
            pin_chips={fh: "freehit"},
            forced_sell={sale: [pedro]},
        )
        plan["label"] = f"{label} · FH GW{fh}, João Pedro sold GW{sale}"
        plans.append(plan)
    for plan in plans:
        plan["id"] = store.put("plan", {k: v for k, v in plan.items() if k != "id"})
    league = store.get("league")
    result = compare(store, plans, forecast_id, league["id"], state["points"], state["entry_id"])
    result["plans"] = [p["id"] for p in plans]
    result["id"] = store.put("case_comparison", {k: v for k, v in result.items() if k != "id"})
    return result
