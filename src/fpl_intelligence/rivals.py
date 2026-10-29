"""Revealed league membership, effective exposure and shared-world policy tests."""

from __future__ import annotations

import asyncio
from itertools import product
from typing import Any

import httpx
import numpy as np

from .domain import Action, ChipUse, Lot, TeamState
from .ingestion import Official, load_market
from .optimization import hold_plan, optimize
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
            await asyncio.gather(
                adapter.fetch(
                    client, f"entry/{identity}/event/{current}/picks/", f"picks_{identity}_{current}"
                ),
                adapter.fetch(client, f"entry/{identity}/history/", f"entry_history_{identity}"),
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
    store: Store, member: dict, league: dict, forecast: dict, end_event: int, style: str
) -> dict:
    players, _ = load_market(store)
    rules = Rules.from_bootstrap(store.latest("bootstrap")[0])
    released = league["released_event"]
    picks, _ = store.latest(f"picks_{member['entry']}_{released}")
    history, _ = store.latest(f"entry_history_{member['entry']}")
    ft = 1
    for row in history["current"]:
        gw = row["event"]
        chips = [c for c in history["chips"] if c["event"] == gw]
        if not any(c["name"] in ("wildcard", "freehit") for c in chips):
            ft = min(rules.ft_cap, max(0, ft - row["event_transfers"]) + 1)
    state = TeamState(
        season=forecast["season"],
        event=forecast["start_event"],
        name=member["team"],
        entry_id=member["entry"],
        bank=picks["entry_history"]["bank"],
        free_transfers=ft,
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
    for event in range(forecast["start_event"], end_event + 1):
        owned = [lot.player_id for lot in state.lots]
        chosen = Action()
        if style != "hold" and state.free_transfers:
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
            "No future rival chip use in these hold/greedy scenarios",
        ],
    }
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
    rival_worlds = {}
    for member in rivals:
        for style in ("hold", "greedy"):
            policy = rival_policy(store, member, league, forecast, end_event, style)
            rival_worlds[member["entry"], style] = member["points"] + score_plan(store, policy, forecast)
    rows = []
    for plan in plans:
        outcomes = own_points + score_plan(store, plan, forecast)
        probabilities: list[dict[str, Any]] = []
        ties = []
        for styles in product(("hold", "greedy"), repeat=2):
            a = rival_worlds[rivals[0]["entry"], styles[0]]
            b = rival_worlds[rivals[1]["entry"], styles[1]]
            win = (outcomes > a) & (outcomes > b)
            tie = ((outcomes == a) & (outcomes >= b)) | ((outcomes == b) & (outcomes >= a))
            probabilities.append(
                {
                    "rival_policies": list(styles),
                    "joint_strict_win": float(win.mean()),
                    "mc_standard_error": float(np.sqrt(win.mean() * (1 - win.mean()) / len(win))),
                    "joint_tie_mass": float(tie.mean()),
                }
            )
            ties.append(float(tie.mean()))
        rows.append(
            {
                "plan_id": plan["id"],
                "label": plan.get("label", "optimized"),
                "expected_net_points": plan["simulation"]["mean"],
                "policies": probabilities,
                "robust_joint_win": min(p["joint_strict_win"] for p in probabilities),
            }
        )
    best_ev = max(row["expected_net_points"] for row in rows)
    eligible = [r for r in rows if r["expected_net_points"] >= best_ev - sacrifice_budget]
    selected = max(eligible, key=lambda r: (r["robust_joint_win"], r["expected_net_points"]))
    result = {
        "created_at": now(),
        "forecast_id": forecast_id,
        "rivals": rivals,
        "starting_points": own_points,
        "candidate_frontier": rows,
        "sacrifice_budget": sacrifice_budget,
        "selected_plan_id": selected["plan_id"],
        "optimization_scope": "Robust joint probability among generated feasible policies, not a global probability optimum",
        "ties": "Reported separately; strict-win probability excludes ties. Exact final rank needs final transfer tie-breaks.",
        "shared_worlds": True,
        "worlds": forecast["worlds"],
    }
    result["id"] = store.put("rival_comparison", result)
    return result


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
