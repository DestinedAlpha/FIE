"""Challenge a single-transfer incumbent with independent, legal first actions.

Published EP proposes a candidate only. Every resulting policy is scored with
the same production forecast, exact account transitions and frozen worlds.
"""

import math

from .domain import Action, TeamState
from .ingestion import load_market
from .rules import Rules, best_lineup, transition


def alternative_actions(store, reference, forecast):
    first = reference["ledger"][0]
    if first.get("chip") or len(first.get("buy", [])) != 1 or len(first.get("sell", [])) != 1:
        return []
    if not forecast.get("players") or not store.list("state", 1):
        return []
    players, market = load_market(store, forecast["market_id"])
    state = TeamState.model_validate(
        {k: v for k, v in store.get("state", reference["state_id"]).items() if k != "id"}
    )
    bought, sold = first["buy"][0]["player_id"], first["sell"][0]["player_id"]
    owned = {lot.player_id for lot in state.lots}
    means = {r["player_id"]: r["mean"] for r in forecast["players"] if r["event"] == state.event}
    horizon: dict[int, float] = {}
    for row in forecast["players"]:
        if state.event <= row["event"] <= reference["end_event"]:
            horizon[row["player_id"]] = horizon.get(row["player_id"], 0.0) + row["mean"]
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    eligible = {}
    for identity, player in players.items():
        if (
            identity in owned
            or identity == bought
            or not player.selectable
            or player.position != players[bought].position
        ):
            continue
        action = Action(
            sell=[sold],
            buy=[identity],
            lineup=best_lineup(sorted(owned - {sold} | {identity}), means, players, rules),
        )
        try:
            _, row = transition(state, action, players, rules)
        except ValueError:
            continue
        if row["hit"] <= first["hit"]:
            eligible[identity] = action
    if not eligible:
        return []
    choices = []
    seen = set()

    def propose(identity, origin):
        if identity not in seen:
            choices.append((eligible[identity], origin))
            seen.add(identity)

    propose(
        max(eligible, key=lambda p: (horizon.get(p, 0), -players[p].price, -p)),
        "Highest modeled horizon points among affordable alternatives",
    )
    propose(
        max(eligible, key=lambda p: (horizon.get(p, 0) / players[p].price, -players[p].price, -p)),
        "Highest modeled points per price among affordable alternatives",
    )
    published = {}
    for identity in eligible:
        value = players[identity].raw.get("ep_next")
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and score > 0:
            published[identity] = score
    if published and state.event == forecast.get("start_event", state.event):
        levels = sorted(set(published.values()), reverse=True)[:2]
        for identity in sorted(published, key=lambda p: (-published[p], players[p].price, p)):
            if published[identity] in levels:
                propose(identity, "Published FPL EP proposes a candidate; production model scores its policy")
            if len(choices) >= 6:
                break

    def observed_rate(identity, names, prior):
        raw = players[identity].raw
        minutes = float(raw.get("minutes") or 0)
        total = sum(float(raw.get(n) or 0) for n in names)
        return (total + 5 * prior) / (minutes / 90 + 5)

    propose(
        max(eligible, key=lambda p: (observed_rate(p, ("total_points",), 4), -players[p].price, -p)),
        "Strong observed season output, exposure-shrunk; not a future outcome label",
    )
    from .features import PRIOR

    prior = PRIOR[players[bought].position]
    propose(
        max(
            eligible,
            key=lambda p: (
                observed_rate(p, ("expected_goals", "expected_assists"), prior["goals"] + prior["assists"]),
                -players[p].price,
                -p,
            ),
        ),
        "Strong observed xG/xA, exposure-shrunk; not a future outcome label",
    )
    return choices[:8]


def compare_cheaper_actions(store, reference, forecast, *, fixed_chips=None, max_transfers=15):
    from .decision_runtime import conditional_plan
    from .search_control import effective_time_limit, planning_budget, report_phase

    remaining = effective_time_limit(None)
    if remaining is not None and remaining < 1:
        return []
    if fixed_chips and reference["ledger"][0]["event"] in fixed_chips:
        return []
    candidates = alternative_actions(store, reference, forecast) if max_transfers else []
    first = reference["ledger"][0]
    if (
        not candidates
        and max_transfers
        and forecast.get("players")
        and (first.get("chip") or len(first.get("buy", [])) != 1)
    ):
        from .optimization import greedy_bundle

        saved = store.get("state", reference["state_id"])
        state = TeamState.model_validate({k: v for k, v in saved.items() if k != "id"})
        if not state.initial and not state.active_chip:
            players, market = load_market(store, forecast["market_id"])
            rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
            means = {r["player_id"]: r["mean"] for r in forecast["players"] if r["event"] == state.event}
            future: dict[int, float] = {}
            for row in forecast["players"]:
                if state.event < row["event"] <= min(reference["end_event"], state.event + 3):
                    future[row["player_id"]] = future.get(row["player_id"], 0.0) + 0.15 * row["mean"]
            # A tiny MIP may only return its unchanged-squad warm start. Use an
            # independently legal improving action to challenge that incumbent.
            action = greedy_bundle(
                state,
                players,
                rules,
                {p: means.get(p, 0) + future.get(p, 0) for p in players},
                min(max_transfers, max(0, state.free_transfers - state.transfers_made)),
            )
            squad = {lot.player_id for lot in state.lots} - set(action.sell) | set(action.buy)
            action.lineup = best_lineup(sorted(squad), means, players, rules)
            _, ledger = transition(state, action, players, rules)
            ordinary = {**reference, "ledger": [ledger]}
            candidates = [(action, "Ordinary no-new-hit alternative to the initial chip or bundle")]
            candidates.extend(alternative_actions(store, ordinary, forecast))
    remaining = effective_time_limit(None)
    if remaining is not None and remaining < 1:
        return []
    seconds = 60.0 if remaining is None else remaining / (len(candidates) + 1)
    plans = []
    for action, origin in candidates:
        report_phase("Comparing cheaper transfer alternatives and later squad options")
        with planning_budget(seconds):
            plan = conditional_plan(
                store,
                reference["state_id"],
                forecast["id"],
                reference["end_event"],
                action,
                seconds,
                origin=origin,
                continuation_options={
                    "chips": None
                    if fixed_chips is None
                    else {e: c for e, c in fixed_chips.items() if e > reference["ledger"][0]["event"]},
                    "max_transfers": max_transfers,
                },
            )
        plans.append(plan)
    return plans
