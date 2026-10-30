"""Connected decision inference, transfer timing and value-of-information analysis."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np

from .domain import Action, Lot, TeamState
from .rules import Rules, best_lineup, transition
from .storage import now
from .strategy import Decision, allowed_chips, candidate_actions, solve_action


def continuation_terms(store, end_event, means, players, rules, state):
    """Compile a validated additive continuation model into MIP coefficients."""
    fallback = {
        p: 0.1 * sum(means.get((e, p), 0) for e in range(end_event + 1, min(38, end_event + 4) + 1))
        for p in players
    }
    result: dict[str, Any] = {
        "players": fallback,
        "bank": 0.0,
        "ft": 0.0,
        "chip_use": {},
        "model_id": None,
        "method": "Declared fixed terminal sensitivity",
        "bank_max": 0,
    }
    if not store.list("state_value_model", 1) or end_event >= 38:
        return result
    model = store.get("state_value_model")
    if not model["promoted"]:
        return result
    coef = model["coefficients"]
    result.update(
        players={
            p: coef[0] * sum(means.get((e, p), 0) for e in range(end_event + 1, min(38, end_event + 3) + 1))
            for p in players
        },
        bank=coef[1] / 10,
        bank_max=int(round(model.get("feature_bounds", [[0, 0], [0, 1000]])[1][1] * 10)),
        ft=coef[2],
        model_id=model["id"],
        method="Learned continuation under the audited evolving-account policy",
    )
    for chip, weight in zip(("wildcard", "freehit", "bboost", "3xc"), coef[3:]):
        try:
            a, b = rules.chip_window(chip, end_event + 1)
            if any(u.chip == chip and a <= u.event <= b for u in state.used_chips):
                continue
            value = float(weight) * min(1, (b - end_event) / 4)
            for event in range(state.event, end_event + 1):
                if a <= event <= b:
                    result["chip_use"][event, chip] = -value
        except ValueError:
            continue
    return result


def critic_residual(store, features, kind):
    spec = store.get("decision_critic")
    item = spec["models"][kind]
    path = store.artifact_path(item["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["hash"]:
        raise ValueError("Decision critic checksum mismatch")
    x = np.asarray(features, dtype=np.float32)
    if kind == "tree":
        return lgb.Booster(model_file=str(path)).predict(x, num_threads=1)
    import torch
    from torch import nn

    model = nn.Sequential(
        nn.Linear(len(spec["features"]), 64),
        nn.SiLU(),
        nn.LayerNorm(64),
        nn.Linear(64, 32),
        nn.SiLU(),
        nn.Linear(32, 1),
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        return (
            model(
                torch.tensor(
                    (x - np.array(spec["neural"]["mean"])) / np.array(spec["neural"]["scale"]),
                    dtype=torch.float32,
                )
            )
            .squeeze(1)
            .numpy()
        )


def decision_inputs(store):
    from .fusion import compare_live
    from .horizon import live_values

    values, players, market, _ = live_values(store)
    saved = store.get("state")
    state = TeamState.model_validate({k: v for k, v in saved.items() if k != "id"})
    forecast = store.get("forecast")
    if forecast["market_id"] != market["id"]:
        raise ValueError("Refresh forecasts for current market before strategy analysis")
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    comparison = compare_live(store)
    q: dict[str, dict[int, float]] = {k: {} for k in ("reference", "tree", "network")}
    for r in comparison["players"]:
        for kind, field in (("reference", "production"), ("tree", "tree"), ("network", "network")):
            q[kind][r["player_id"]] = r[field]
    for p in players:
        for kind in q:
            q[kind].setdefault(p, 0.0)
    long = {p: v["fourweek"] for p, v in values.items()}
    return saved, state, players, market, rules, forecast, q, long


def analyze_decisions(store, seconds=0.4):
    from .decision_training import action_features, state_features

    saved, state, players, market, rules, forecast, q, long = decision_inputs(store)
    candidates, errors = candidate_actions(state, players, rules, q, seconds=seconds, alternatives=2)
    baseline = store.get("plan", store.get("selected_plan")["plan_id"])
    if baseline["market_id"] != market["id"] or baseline["state_id"] != saved["id"]:
        raise ValueError("Selected baseline plan is stale")
    from .scenarios import action_from_row

    a = action_from_row(baseline["ledger"][0])
    after, ledger = transition(state, a, players, rules)
    candidates.append(
        Decision(a, after, ledger, baseline["solver"], baseline["ledger"][0]["expected_additive_net"])
    )
    features = [action_features(d, state, players, rules, q, long) for d in candidates]
    critic = store.get("decision_critic") if store.list("decision_critic", 1) else None
    predictions = (
        {kind: critic_residual(store, features, kind) for kind in ("tree", "network")} if critic else {}
    )
    result = []
    value = store.get("state_value_model") if store.list("state_value_model", 1) else None
    for i, (d, x) in enumerate(zip(candidates, features)):
        continuation = (
            float(
                np.dot(state_features(d.state, players, rules, long, q["reference"]), value["coefficients"])
            )
            if value and value["promoted"]
            else 0.1 * x[12]
        )
        critic_config = critic or {}
        item = {
            "action": d.action.model_dump(),
            "ledger": d.ledger,
            "reference": x[0],
            "tree": x[1],
            "network": x[2],
            "continuation": continuation,
            "reference_plus_continuation": x[0] + continuation,
            "critic_scores": {
                k: float(x[0] + critic_config["candidates"][k]["weight"] * p[i])
                for k, p in predictions.items()
            },
            "solver": d.solver,
            "baseline_policy_first_action": i == len(candidates) - 1,
        }
        result.append(item)
    selected = (
        max(range(len(result)), key=lambda i: result[i]["critic_scores"][critic["selected"]])
        if critic and critic["selected"]
        else len(result) - 1
    )
    report = {
        "at": now(),
        "state_id": saved["id"],
        "market_id": market["id"],
        "forecast_id": forecast["id"],
        "decision_critic_id": critic["id"] if critic else None,
        "state_value_model_id": value["id"] if value else None,
        "candidates": result,
        "selected_candidate": selected,
        "selected_architecture": critic["selected"] if critic else None,
        "promotion_gate_passed": bool(critic and critic["selected"]),
        "search_exclusions": errors,
        "automatic_execution": False,
        "status": "Validated critic selects a legal first action"
        if critic and critic["selected"]
        else "Complete challenger comparison; current multiweek policy retained by validation gate",
        "connections": {
            "official_and_history": "Reference forecast and rule transitions",
            "news_managers_workload": "Tree/neural fusion and source-conditioned inputs",
            "neural": "Temporal fusion produces player forecasts; separate listwise critic learns decision regret",
            "optimizer": "All selectable players, transfers, hits, all currently legal chips and exact bank/FT transitions",
            "hindsight": "Outcome labels and diagnostic ceilings only",
            "continuation": "Audited learned state value when its held-out gate passes",
        },
    }
    report["id"] = store.put("decision_analysis", report)
    return report


def commit_before_deadline(state, action, players, rules):
    """Keep a proposed early action within its current deadline for late recourse."""
    _, ledger = transition(state, action, players, rules)
    lots = {lot.player_id: lot for lot in state.lots}
    playing = [
        lots[p]
        if p in lots
        else Lot(
            player_id=p,
            purchase_price=players[p].price,
            sell_price=players[p].price,
            quote_market_price=players[p].price,
            provenance="Conditional early committed purchase",
        )
        for p in ledger["playing_squad"]
    ]
    return state.model_copy(
        update={
            "lots": playing,
            "bank": ledger["bank_after"],
            "active_chip": action.chip or state.active_chip,
            "transfers_made": state.transfers_made + len(action.buy),
            "hit_paid": state.hit_paid + ledger["hit"],
            "deadline_lots": state.deadline_lots or state.lots,
            "deadline_bank": state.deadline_bank if state.deadline_bank is not None else state.bank,
        }
    )


def timing_analysis(store, worlds=24):
    from .market_model import market_predictions, price_bridge
    from .scenarios import action_from_row, sale_bounds

    saved, state, players, market, rules, forecast, q, long = decision_inputs(store)
    plan = store.get("plan", store.get("selected_plan")["plan_id"])
    action = action_from_row(plan["ledger"][0])
    committed = commit_before_deadline(state, action, players, rules)
    pred, model = market_predictions(store, players, state.event)
    rng = np.random.default_rng(173)
    deadline = next(e["deadline_time"] for e in market["events"] if e["id"] == state.event)
    hours = (
        datetime.fromisoformat(deadline.replace("Z", "+00:00")) - datetime.fromisoformat(now())
    ).total_seconds() / 3600
    # Prioritize sources by whether a feasible alternative or the actual account uses them.
    relevant = set(p.player_id for p in state.lots) | set(action.buy)
    relevant.update(sorted(players, key=lambda p: -q["reference"].get(p, 0))[:30])
    results = []
    source_values = []
    for sample in range(worlds):
        changed = dict(players)
        means = dict(q["reference"])
        news = []
        for p in relevant:
            direction = int(rng.choice(3, p=pred[p]["price_probabilities"]))
            pool = model["magnitudes"][str(direction)] or [direction - 1]
            delta = int(rng.choice(pool))
            chance_zero = rng.random() < pred[p]["new_unavailable_probability"]
            if chance_zero:
                means[p] = 0.0
                news.append(p)
            changed[p] = players[p].model_copy(update={"price": max(0, players[p].price + delta)})

        def priced(s):
            def update(lots):
                return [
                    lot.model_copy(
                        update={
                            "sell_price": sale_bounds(
                                lot, players[lot.player_id].price, changed[lot.player_id].price
                            )[0],
                            "quote_market_price": changed[lot.player_id].price,
                            "provenance": lot.provenance + "; conditional conservative price bridge",
                        }
                    )
                    for lot in lots
                ]

            return s.model_copy(
                update={
                    "lots": update(s.lots),
                    "deadline_lots": update(s.deadline_lots) if s.deadline_lots is not None else None,
                }
            )

        early = priced(committed)
        waiting = priced(state)
        # Both sides may make further legal transfers after observing information.
        # An activated chip remains the only chip in the early branch.
        early_modes = [action.chip] if action.chip else allowed_chips(early, rules)

        def choose(s, modes):
            choices = []
            for chip in modes:
                try:
                    choices.append(
                        solve_action(s, changed, rules, means, chip=chip, seconds=0.15, max_transfers=3)
                    )
                except ValueError:
                    continue
            if not choices:
                raise ValueError("No lawful timing recourse under the price branch")
            return max(choices, key=lambda d: d.utility)

        e = choose(early, early_modes)
        w = choose(waiting, allowed_chips(waiting, rules))
        early_value = e.utility - plan["ledger"][0]["hit"]
        wait_value = w.utility
        results.append(
            {
                "sample": sample,
                "early": early_value,
                "wait": wait_value,
                "wait_minus_early": wait_value - early_value,
                "new_unavailable": news,
                "early_hit": e.ledger["event_total_hit"],
                "wait_hit": w.ledger["event_total_hit"],
                "early_solver_gap": e.solver["gap"],
                "wait_solver_gap": w.solver["gap"],
            }
        )
    gains = np.array([r["wait_minus_early"] for r in results])
    stderr = float(gains.std(ddof=1) / np.sqrt(len(gains)))
    # Value of a targeted availability report over legal, pre-generated candidates.
    analysis = (
        store.get("decision_analysis") if store.list("decision_analysis", 1) else analyze_decisions(store)
    )
    for p in sorted(relevant, key=lambda p: -q["reference"].get(p, 0))[:12]:
        probability = pred[p]["new_unavailable_probability"]
        payoffs = []
        for row in analysis["candidates"]:
            a = Action.model_validate(row["action"])
            squad = row["ledger"]["playing_squad"]
            values = []
            for absent in (False, True):
                local = dict(q["reference"])
                if absent:
                    local[p] = 0
                line = best_lineup(squad, local, players, rules)
                value = (
                    sum(local.get(k, 0) for k in line.xi)
                    + local.get(line.captain, 0) * (2 if a.chip == "3xc" else 1)
                    - row["ledger"]["hit"]
                )
                if a.chip == "bboost":
                    value += sum(local.get(k, 0) for k in line.bench)
                values.append(value)
            payoffs.append(values)
        matrix = np.array(payoffs)
        prior = np.array([1 - probability, probability])
        before = float((matrix @ prior).max())
        after = float(matrix.max(0) @ prior)
        source_values.append(
            {
                "player_id": p,
                "expected_value_of_report": max(0, after - before),
                "report_unavailable_probability": probability,
                "scope": "Best legal candidate after versus before a new official unavailability report; numerical probability comes from the separately tested pre-deadline news-transition model",
            }
        )
    bridge = price_bridge(store, plan["id"])
    result = {
        "at": now(),
        "state_id": saved["id"],
        "market_id": market["id"],
        "plan_id": plan["id"],
        "market_model_id": model["id"],
        "worlds": worlds,
        "hours_to_deadline": hours,
        "within_trained_time_support": 40 <= hours <= 72,
        "early_mean": float(np.mean([r["early"] for r in results])),
        "wait_mean": float(np.mean([r["wait"] for r in results])),
        "wait_minus_early": float(gains.mean()),
        "paired_mc_interval_95": [float(gains.mean() - 1.96 * stderr), float(gains.mean() + 1.96 * stderr)],
        "price_bridge_id": bridge["id"],
        "affordability_probability_bounds": bridge["affordable_probability_bounds"],
        "research_priority": sorted(source_values, key=lambda r: -r["expected_value_of_report"]),
        "branches": results,
        "scope": "Conditional current-deadline timing experiment with legal post-news/price recourse on both sides, including extra hits and activated-chip constraints",
        "limits": [
            "Short-horizon timing utility; the chosen full-horizon plan remains independently scored",
            "Finite scenario and solver uncertainty are additional to the paired Monte Carlo interval",
            "Price/news transitions and players are sampled independently; outside 40–72 hours this is a sensitivity analysis",
            "Unknown acquisition lots use conservative bounds, never reconstructed actual purchases",
        ],
    }
    result["id"] = store.put("timing_analysis", result)
    return result


def conditional_plan(store, state_id, forecast_id, end_event, action, seconds=20, origin=None):
    """Fix one legal first action and solve its full remaining horizon independently."""
    from .ingestion import load_market
    from .optimization import optimize
    from .rules import optimize_bench
    from .simulation import score_plan

    players, market = load_market(store)
    saved = store.get("state", state_id)
    state = TeamState.model_validate({k: v for k, v in saved.items() if k != "id"})
    forecast = store.get("forecast", forecast_id)
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    with np.load(store.artifact_path(forecast["scenario_path"])) as worlds:
        action.lineup = optimize_bench(
            action.lineup,
            worlds[f"points_{state.event}"],
            worlds[f"minutes_{state.event}"],
            forecast["player_ids"],
            players,
            rules,
        )
    following, row = transition(state, action, players, rules)
    means = {r["player_id"]: r["mean"] for r in forecast["players"] if r["event"] == state.event}
    line = action.lineup
    row["expected_additive_net"] = (
        sum(means.get(p, 0) for p in line.xi)
        + means.get(line.captain, 0) * (2 if action.chip == "3xc" else 1)
        - row["hit"]
    )
    if action.chip == "bboost":
        row["expected_additive_net"] += sum(means.get(p, 0) for p in line.bench)
    row["assumptions"] = [
        "Exact fixed first action; later decisions are re-optimized on frozen current forecasts"
    ]
    if end_event > state.event:
        identity = store.put("planning_state", following.model_dump())
        # Equivalent first actions (especially restored FH squads) must not look
        # worse merely because a fresh continuation search has a weaker incumbent.
        from .scenarios import action_from_row

        compatible = []
        for item in store.list("plan", 100):
            prior = store.get("plan", item["id"])
            rows = prior.get("ledger", [])
            if prior.get("state_id") != state_id or prior.get("market_id") != market["id"] or not rows:
                continue
            if rows[0]["event"] != state.event or rows[-1]["event"] < end_event:
                continue
            try:
                after, _ = transition(state, action_from_row(rows[0]), players, rules)
                if after.model_dump() != following.model_dump():
                    continue
                suffix = [r for r in rows if following.event <= r["event"] <= end_event]
                trial = following
                for step in suffix:
                    trial, _ = transition(trial, action_from_row(step), players, rules)
                compatible.append(
                    (sum(r.get("expected_additive_net", 0) for r in suffix), suffix, prior["id"])
                )
            except ValueError:
                continue
        if compatible:
            _, suffix, parent = max(compatible, key=lambda entry: entry[0])
            seed = {
                "state_id": identity,
                "forecast_id": forecast_id,
                "market_id": market["id"],
                "start_event": following.event,
                "end_event": end_event,
                "ledger": suffix,
                "validation": {"legal": True, "affordable": True},
                "warm_parent_id": parent,
            }
            store.put("warm_start", seed)
        tail = optimize(store, identity, forecast_id, end_event, seconds, use_decision_critic=False)
        plan = {k: v for k, v in tail.items() if k != "id"}
        plan["ledger"] = [row] + tail["ledger"]
        plan["solver"] = {
            **tail["solver"],
            "scope": "MIP bound/gap applies to the continuation conditional on the independently selected first action",
            "first_action_source": origin,
            "continuation_plan_id": tail["id"],
        }
    else:
        plan = {
            "ledger": [row],
            "solver": {
                "status": "Independently validated fixed action",
                "gap": None,
                "scope": "No claim of a global solve bound",
            },
            "final_state": following.model_dump(),
        }
    plan.update(
        created_at=now(),
        state_id=state_id,
        forecast_id=forecast_id,
        market_id=market["id"],
        start_event=state.event,
        end_event=end_event,
        rules_hash=rules.hash,
        input_hashes=forecast["input_hashes"],
        state_status=state.status,
        execution_ready=state.status == "confirmed",
        objective="decision_focused_first_action_and_multiweek_continuation",
        validation={"legal": True, "affordable": True, "independent_transition": True},
    )
    score_plan(store, plan, forecast)
    plan["id"] = store.put("plan", plan)
    return plan


def refine_frontier(store, seconds=20):
    from .optimization import hold_plan, optimize
    from .rivals import compare

    analysis = analyze_decisions(store)
    state = store.get("state")
    forecast = store.get("forecast")
    old = store.get("plan", store.get("selected_plan")["plan_id"])
    end = old["end_event"]
    pool = analysis["candidates"]
    indices = {analysis["selected_candidate"]}
    for key in ("reference_plus_continuation", "tree", "network"):
        indices.add(max(range(len(pool)), key=lambda i: pool[i][key]))
    if analysis["promotion_gate_passed"]:
        indices.add(
            max(range(len(pool)), key=lambda i: pool[i]["critic_scores"][analysis["selected_architecture"]])
        )
    plans = [
        optimize(store, state["id"], forecast["id"], end, max(60, seconds), use_decision_critic=False),
        hold_plan(store, state["id"], forecast["id"], end),
    ]
    plans[0]["label"] = "Joint full-market search"
    if store.list("case_comparison", 1):
        previous = store.get("case_comparison")
        for plan_id in previous.get("plans", []):
            candidate = store.get("plan", plan_id)
            if (
                candidate["state_id"] == state["id"]
                and candidate["forecast_id"] == forecast["id"]
                and candidate.get("end_event", candidate["ledger"][-1]["event"]) == end
            ):
                plans.append(candidate)
    for i in sorted(indices):
        action = Action.model_validate(pool[i]["action"])
        p = conditional_plan(
            store,
            state["id"],
            forecast["id"],
            end,
            action,
            seconds,
            origin={"decision_analysis": analysis["id"], "candidate": i},
        )
        p["label"] = f"Action family {i + 1}: " + (action.chip or f"{len(action.buy)} transfers")
        plans.append(p)
    for p in plans:
        p["id"] = store.put("plan", {k: v for k, v in p.items() if k != "id"})
    league = store.get("league")
    result = compare(store, plans, forecast["id"], league["id"], state["points"], state["entry_id"])
    result["plans"] = [p["id"] for p in plans]
    result["decision_analysis_id"] = analysis["id"]
    result["id"] = store.put("case_comparison", {k: v for k, v in result.items() if k != "id"})
    if store.get("state")["id"] == state["id"] and store.get("forecast")["id"] == forecast["id"]:
        store.put("selected_plan", {"plan_id": result["selected_plan_id"], "at": now()})
    else:
        result["selection_status"] = (
            "Account or forecast changed during planning; result retained as an analysis only"
        )
    return result


def validate_horizons(store, seconds=30):
    """Audit a tactical first decision against the longer chip-boundary policy."""
    from .optimization import optimize
    from .scenarios import action_from_row
    from .simulation import score_plan

    state = store.get("state")
    forecast = store.get("forecast")
    long_end = min(forecast["end_event"], 19 if state["event"] <= 19 else 38)
    short_end = min(long_end, state["event"] + 3)
    if short_end == long_end:
        raise ValueError("A longer forecast horizon is required for this comparison")
    short = optimize(store, state["id"], forecast["id"], short_end, seconds, use_decision_critic=False)
    long = optimize(store, state["id"], forecast["id"], long_end, seconds, use_decision_critic=False)
    continued = conditional_plan(
        store,
        state["id"],
        forecast["id"],
        long_end,
        action_from_row(short["ledger"][0]),
        seconds,
        origin={"short_horizon_plan_id": short["id"]},
    )
    delta = score_plan(store, continued, forecast) - score_plan(store, long, forecast)
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    report = {
        "at": now(),
        "state_id": state["id"],
        "forecast_id": forecast["id"],
        "short_plan_id": short["id"],
        "long_plan_id": long["id"],
        "continued_plan_id": continued["id"],
        "short_end": short_end,
        "long_end": long_end,
        "short_first_action": action_from_row(short["ledger"][0]).model_dump(),
        "long_first_action": action_from_row(long["ledger"][0]).model_dump(),
        "continued_short_minus_long": mean,
        "paired_mc_interval_95": [mean - 1.96 * se, mean + 1.96 * se],
        "solver": {"short": short["solver"], "long": long["solver"], "continued": continued["solver"]},
        "scope": "Model-based common-world horizon sensitivity, including finite solver search; not a prospective terminal-value acceptance test",
    }
    report["id"] = store.put("horizon_consistency", report)
    return report
