"""Full-market action search shared by learning, rival policies and live recourse.

Every candidate is solved under the same account rules, then independently
transitioned. Forecasts and hindsight labels enter through separate arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

import numpy as np

from .domain import Action, Chip, TeamState
from .optimization import MIP
from .rules import Rules, best_lineup, quote, transition


@dataclass
class Decision:
    action: Action
    state: TeamState
    ledger: dict
    solver: dict
    utility: float


def allowed_chips(state: TeamState, rules: Rules):
    if state.active_chip is not None:
        return [state.active_chip]
    result: list[str | None] = [None]
    for chip in ("wildcard", "freehit", "bboost", "3xc"):
        try:
            lo, hi = rules.chip_window(chip, state.event)
        except ValueError:
            continue
        if not any(u.chip == chip and lo <= u.event <= hi for u in state.used_chips):
            if chip == "freehit" and any(
                u.chip == chip and u.event == state.event - 1 for u in state.used_chips
            ):
                continue
            if (
                chip == "freehit"
                and (state.transfers_made or state.active_chip == "freehit")
                and state.deadline_lots is None
            ):
                continue
            result.append(chip)
    return result


def solve_action(
    state,
    players,
    rules,
    means: Mapping[int, float],
    *,
    chip=None,
    max_transfers=15,
    min_transfers=0,
    seconds=0.3,
    squad_value=None,
    bank_value=0.0,
    transfer_bias=None,
    forbidden=(),
    relative_gap=0.01,
):
    """Joint squad/XI/captain/chip solve; hits and gain-sharing use exact quotes.

    `squad_value` is an externally validated continuation value. It is never inferred
    from future realized outcomes by this solver. Free Hit has no permanent-squad
    continuation reward. `forbidden` creates genuinely different first actions.
    """
    chip = chip or state.active_chip
    if chip not in allowed_chips(state, rules):
        raise ValueError("Chip is unavailable for this account and deadline")
    lots = {x.player_id: x for x in state.lots}
    owned = set(lots)
    ids = sorted(p for p in players if players[p].selectable or p in owned)
    if not owned <= set(ids):
        raise ValueError("Owned player missing from market")
    financial_lots = (
        {lot.player_id: lot for lot in state.deadline_lots}
        if chip == "freehit"
        and (state.transfers_made or state.active_chip == "freehit")
        and state.deadline_lots
        else lots
    )
    costs = {
        p: quote(financial_lots[p], players[p].price) if p in financial_lots else players[p].price
        for p in ids
    }
    financial_bank = (
        state.deadline_bank
        if chip == "freehit" and (state.transfers_made or state.active_chip == "freehit")
        else state.bank
    )
    assert financial_bank is not None
    budget = financial_bank + sum(costs[p] for p in financial_lots)
    model = MIP()
    unlimited = chip in ("wildcard", "freehit") or state.initial
    squad_value = squad_value or {}
    transfer_bias = transfer_bias or {}
    y, xi, cap = {}, {}, {}
    for p in ids:
        continuation = 0 if chip == "freehit" else squad_value.get(p, 0)
        preference = transfer_bias.get(p, 0)
        y[p] = model.var(("y", p), cost=continuation + preference - bank_value * costs[p])
        xi[p] = model.var(("xi", p), cost=means.get(p, 0))
        cap[p] = model.var(("cap", p), cost=means.get(p, 0) * (2 if chip == "3xc" else 1))
        model.row({xi[p]: 1, y[p]: -1}, upper=0)
        model.row({cap[p]: 1, xi[p]: -1}, upper=0)
        if chip == "bboost":
            model.costs[y[p]] += means.get(p, 0)
            model.costs[xi[p]] -= means.get(p, 0)
    model.equal([(y[p], 1) for p in ids], 15)
    model.equal([(xi[p], 1) for p in ids], 11)
    model.equal([(cap[p], 1) for p in ids], 1)
    model.row([(y[p], costs[p]) for p in ids], upper=budget)
    for pos, count in rules.composition.items():
        group = [p for p in ids if players[p].position == pos]
        model.equal([(y[p], 1) for p in group], count)
        model.row([(xi[p], 1) for p in group], rules.minimum[pos], rules.maximum[pos])
    from collections import Counter

    oldclubs = Counter(players[p].team for p in owned)
    for club in {players[p].team for p in ids}:
        # Grandfathered excess is only preserved when no ordinary transfers occur.
        limit = (
            max(rules.club_limit, oldclubs[club])
            if max_transfers == 0 and not unlimited
            else rules.club_limit
        )
        model.row([(y[p], 1) for p in ids if players[p].team == club], upper=limit)
    purchases = [(y[p], 1) for p in ids if p not in owned]
    if not unlimited:
        model.row(
            purchases,
            lower=min_transfers,
            upper=min(max_transfers, rules.transfer_cap - state.transfers_made),
        )
    hit = model.var(("excess",), cost=0 if unlimited else -rules.hit, upper=15)
    if not unlimited:
        model.row(purchases + [(hit, -1)], upper=state.free_transfers - state.transfers_made)
    for buys in forbidden:
        current = set(buys)
        # Exact exclusion, including the all-zero roll vector.
        terms = [(y[p], -1 if p in current else 1) for p in ids if p not in owned]
        model.row(terms, lower=1 - len(current))
    warm = np.zeros(len(model.costs))
    seed_owned = set(financial_lots) if chip == "freehit" else owned
    lineup = best_lineup(sorted(seed_owned), dict(means), players, rules)
    for p in seed_owned:
        warm[y[p]] = 1
        warm[xi[p]] = int(p in lineup.xi)
        warm[cap[p]] = int(p == lineup.captain)
    warm[hit] = max(0, state.transfers_made - state.free_transfers) if not unlimited else 0
    if min_transfers > 0 and not unlimited:
        # Build a legal exact-count seed before spending the optimization budget.
        # It is only an incumbent; the complete registered market stays in the MIP.
        tentative = set(owned)
        sold = []
        bought: list[int] = []
        for old in sorted(
            owned, key=lambda p: (-int(oldclubs[players[p].team] > rules.club_limit), -players[p].price, p)
        ):
            if len(bought) >= min_transfers:
                break
            trial = tentative - {old}
            possible = [
                p
                for p in ids
                if p not in owned
                and p not in bought
                and players[p].selectable
                and players[p].position == players[old].position
                and sum(players[k].team == players[p].team for k in trial) < rules.club_limit
            ]
            if not possible:
                continue
            new = min(possible, key=lambda p: (players[p].price, -means.get(p, 0), p))
            tentative = trial | {new}
            sold.append(old)
            bought.append(new)
        if len(bought) == min_transfers:
            proposed = np.zeros(len(model.costs))
            line = best_lineup(sorted(tentative), dict(means), players, rules)
            for p in tentative:
                proposed[y[p]] = 1
                proposed[xi[p]] = int(p in line.xi)
                proposed[cap[p]] = int(p == line.captain)
            proposed[hit] = max(0, len(bought) + state.transfers_made - state.free_transfers)
            warm = proposed
    rows = np.bincount(
        np.asarray(model.rows, dtype=int),
        weights=np.asarray(model.values) * warm[np.asarray(model.cols, dtype=int)],
        minlength=len(model.row_lower),
    )
    feasible = (
        np.all(warm >= np.asarray(model.lower) - 1e-7)
        and np.all(warm <= np.asarray(model.upper) + 1e-7)
        and np.all(rows >= np.asarray(model.row_lower) - 1e-7)
        and np.all(rows <= np.asarray(model.row_upper) + 1e-7)
    )
    seed = warm if feasible else None
    solution, info = model.solve(seconds, seed, relative_gap=relative_gap)
    chosen = {p for p in ids if solution[y[p]] > 0.5}
    action = Action(
        sell=sorted(owned - chosen),
        buy=sorted(chosen - owned),
        chip=cast(Chip | None, chip),
        lineup=best_lineup(sorted(chosen), dict(means), players, rules),
    )
    assert action.lineup is not None
    following, ledger = transition(state, action, players, rules)
    utility = sum(means.get(p, 0) for p in action.lineup.xi) + means.get(action.lineup.captain, 0)
    if chip == "3xc":
        utility += means.get(action.lineup.captain, 0)
    if chip == "bboost":
        utility += sum(means.get(p, 0) for p in action.lineup.bench)
    utility -= ledger["hit"]
    if chip != "freehit":
        utility += sum(squad_value.get(p, 0) for p in chosen)
    utility += bank_value * following.bank
    return Decision(action, following, ledger, info, float(utility))


def candidate_actions(
    state,
    players,
    rules,
    forecasts,
    *,
    seconds=0.2,
    chips=True,
    alternatives=2,
    squad_value=None,
    bank_value=0.0,
    ordinary_limit=15,
):
    """Full-market optimized alternatives across forecast sources, hits and chips."""
    results: dict[tuple, Decision] = {}
    errors = []
    for label, means in forecasts.items():
        modes = allowed_chips(state, rules) if chips else [None]
        for chip in modes:
            forbidden: list[list[int]] = []
            for i in range(alternatives if chip is None else 1):
                try:
                    d = solve_action(
                        state,
                        players,
                        rules,
                        means,
                        chip=chip,
                        seconds=seconds,
                        max_transfers=ordinary_limit,
                        forbidden=forbidden,
                        squad_value=squad_value,
                        bank_value=bank_value,
                    )
                    d.ledger["forecast_source"] = label
                    forbidden.append(d.action.buy)
                    key = (
                        tuple(d.action.sell),
                        tuple(d.action.buy),
                        chip,
                        tuple(d.action.lineup.xi),
                        d.action.lineup.captain,
                    )
                    results.setdefault(key, d)
                except ValueError as exc:
                    errors.append({"source": label, "chip": chip, "alternative": i, "reason": str(exc)})
        if label == next(iter(forecasts)):
            d = solve_action(state, players, rules, means, max_transfers=0, seconds=seconds)
            d.ledger["forecast_source"] = "roll"
            results[((), (), None, tuple(d.action.lineup.xi), d.action.lineup.captain)] = d
    if not results:
        raise ValueError("No legal candidate action: " + str(errors[:3]))
    return list(results.values()), errors
