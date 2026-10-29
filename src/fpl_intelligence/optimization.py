"""Full-market multiweek MIP with lot accounting, FT network, and joint chips.

Additive expected XI/captain scores generate candidates. Exported plans are then
independently transitioned and rescored in shared worlds with exact autosubs.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Mapping, cast

import highspy
import numpy as np
from scipy.sparse import coo_matrix

from .domain import Action, Chip, Lineup, TeamState
from .ingestion import load_market
from .rules import Rules, best_lineup, optimize_bench, quote, transition, validate_squad
from .simulation import score_plan
from .storage import Store, now


class MIP:
    def __init__(self):
        self.keys = {}
        self.lower, self.upper, self.costs, self.integer = [], [], [], []
        self.rows, self.cols, self.values, self.row_lower, self.row_upper = [], [], [], [], []

    def var(self, key, cost=0, lower=0, upper=1, integer=True):
        i = len(self.costs)
        self.keys[key] = i
        self.lower.append(lower)
        self.upper.append(upper)
        self.costs.append(cost)
        self.integer.append(integer)
        return i

    def row(self, terms, lower=-np.inf, upper=np.inf):
        row = len(self.row_lower)
        items = terms.items() if isinstance(terms, dict) else terms
        summed: dict[int, float] = defaultdict(float)
        for column, value in items:
            summed[column] += value
        for column, value in summed.items():
            if value:
                self.rows.append(row)
                self.cols.append(column)
                self.values.append(value)
        self.row_lower.append(lower)
        self.row_upper.append(upper)

    def equal(self, terms, value=0):
        self.row(terms, value, value)

    def and_var(self, key, a, b, cost=0):
        if self.upper[a] == 0 or self.upper[b] == 0:
            return self.var(key, cost, upper=0)
        v = self.var(key, cost)
        self.row({v: 1, a: -1}, upper=0)
        self.row({v: 1, b: -1}, upper=0)
        self.row({v: 1, a: -1, b: -1}, lower=-1)
        return v

    def solve(self, seconds, warm_start=None):
        count = len(self.costs)
        matrix = coo_matrix((self.values, (self.rows, self.cols)), shape=(len(self.row_lower), count)).tocsr()
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = count, len(self.row_lower)
        lp.col_cost_ = self.costs
        lp.col_lower_, lp.col_upper_ = self.lower, self.upper
        lp.row_lower_, lp.row_upper_ = self.row_lower, self.row_upper
        lp.sense_ = highspy.ObjSense.kMaximize
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = matrix.indptr
        lp.a_matrix_.index_ = matrix.indices
        lp.a_matrix_.value_ = matrix.data
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous for i in self.integer
        ]
        solver = highspy.Highs()
        solver.setOptionValue("output_flag", False)
        if os.getenv("FPL_SOLVER_LOG"):
            solver.setOptionValue("log_file", os.environ["FPL_SOLVER_LOG"])
            solver.setOptionValue("output_flag", True)
            solver.setOptionValue("log_to_console", False)
        solver.setOptionValue("time_limit", seconds)
        solver.setOptionValue("mip_rel_gap", 0.01)
        solver.setOptionValue("presolve", "on")
        solver.setOptionValue("random_seed", 11)
        solver.setOptionValue("threads", 2)
        model_status = solver.passModel(lp)
        if model_status != highspy.HighsStatus.kOk:
            raise ValueError(f"Solver rejected the formulation: {model_status}")
        if warm_start is not None:
            solution = highspy.HighsSolution()
            solution.col_value = warm_start
            solution.value_valid = True
            solver.setSolution(solution)
        solver.run()
        status = solver.modelStatusToString(solver.getModelStatus())
        solution, info = solver.getSolution(), solver.getInfo()
        if not solution.value_valid or info.primal_solution_status != 2:
            raise ValueError(f"Solver produced no feasible incumbent: {status}; increase time limit")
        return np.array(solution.col_value), {
            "status": status,
            "objective": info.objective_function_value,
            "best_bound": info.mip_dual_bound,
            "gap": info.mip_gap,
            "nodes": info.mip_node_count,
            "variables": count,
            "constraints": len(self.row_lower),
        }


def select_squad(state, players, rules, means, seconds=1, initial_budget=1000):
    """Full-market one-week squad oracle used only to seed the multiweek solve."""
    owned = {lot.player_id: lot for lot in state.lots} if state is not None else {}
    ids = [p for p in players if players[p].selectable or p in owned]
    costs = {p: quote(owned[p], players[p].price) if p in owned else players[p].price for p in ids}
    budget = state.bank + sum(costs[p] for p in owned) if state is not None else initial_budget
    model = MIP()
    y = {p: model.var(("y", p), cost=0.015 * means.get(p, 0)) for p in ids}
    x = {p: model.var(("x", p), cost=means.get(p, 0)) for p in ids}
    c = {p: model.var(("c", p), cost=means.get(p, 0)) for p in ids}
    model.equal([(y[p], 1) for p in ids], 15)
    model.equal([(x[p], 1) for p in ids], 11)
    model.equal([(c[p], 1) for p in ids], 1)
    model.row([(y[p], costs[p]) for p in ids], upper=budget)
    for pos, count in rules.composition.items():
        group = [p for p in ids if players[p].position == pos]
        model.equal([(y[p], 1) for p in group], count)
        model.row([(x[p], 1) for p in group], rules.minimum[pos], rules.maximum[pos])
    for club in {players[p].team for p in ids}:
        model.row([(y[p], 1) for p in ids if players[p].team == club], upper=3)
    for p in ids:
        model.row({x[p]: 1, y[p]: -1}, upper=0)
        model.row({c[p]: 1, x[p]: -1}, upper=0)
    warm = None
    if owned:
        warm = np.zeros(len(model.costs))
        lineup = best_lineup(list(owned), means, players, rules)
        for p in owned:
            warm[y[p]] = 1
            warm[x[p]] = int(p in lineup.xi)
            warm[c[p]] = int(p == lineup.captain)
    solution, _ = model.solve(seconds, warm)
    return [p for p in ids if solution[y[p]] > 0.5]


def greedy_bundle(state, players, rules, means, max_transfers=2):
    """A legal, bounded receding-horizon incumbent, never labeled an optimum."""
    owned = {lot.player_id for lot in state.lots}
    lots = {lot.player_id: lot for lot in state.lots}
    candidates = sorted(
        (p for p in players if p not in owned and players[p].selectable),
        key=lambda p: means.get(p, 0),
        reverse=True,
    )[:100]
    chosen = Action()
    bank = state.bank
    current = set(owned)
    for iteration in range(max_transfers):
        lineup = best_lineup(sorted(current), means, players, rules)
        base = sum(means.get(p, 0) for p in lineup.xi) + means.get(lineup.captain, 0)
        threshold = 0.4 if iteration < state.free_transfers - state.transfers_made else 4.4
        best = None
        for old in current & owned - set(chosen.buy):
            for new in candidates:
                if new in current or new in chosen.sell or players[new].position != players[old].position:
                    continue
                proceeds = quote(lots[old], players[old].price)
                if players[new].price > bank + proceeds:
                    continue
                trial = current - {old} | {new}
                if sum(players[p].team == players[new].team for p in trial) > 3:
                    continue
                next_lineup = best_lineup(sorted(trial), means, players, rules)
                value = sum(means.get(p, 0) for p in next_lineup.xi) + means.get(next_lineup.captain, 0)
                gain = value - base
                if gain > threshold and (best is None or gain > best[0]):
                    best = gain, old, new, proceeds
        if best is None:
            break
        _, old, new, proceeds = best
        chosen.sell.append(old)
        chosen.buy.append(new)
        current = current - {old} | {new}
        bank += proceeds - players[new].price
    return chosen


def optimize(
    store: Store,
    state_id: str,
    forecast_id: str,
    end_event: int,
    time_limit: float = 30,
    chips: Mapping[int, str] | None = None,
    max_transfers: int = 3,
    automatic_chips: bool = True,
    forced_first: Action | None = None,
    pin_chips: Mapping[int, str] | None = None,
    forced_sell: dict[int, list[int]] | None = None,
    disabled_chips: tuple[str, ...] = (),
) -> dict:
    started = time.monotonic()
    if time_limit <= 0 or not 0 <= max_transfers <= 15:
        raise ValueError("Solve time must be positive and per-event transfers must be between 0 and 15")
    players, market = load_market(store)
    state = TeamState.model_validate({k: v for k, v in store.get("state", state_id).items() if k != "id"})
    forecast = store.get("forecast", forecast_id)
    rules = Rules.from_bootstrap(store.latest("bootstrap")[0])
    validate_squad(state, players, rules)
    if forecast["market_id"] != market["id"] or forecast["rules_hash"] != rules.hash:
        raise ValueError("Forecast inputs are stale: refresh/recompute before solving")
    if not forecast["start_event"] <= state.event <= end_event <= forecast["end_event"]:
        raise ValueError("State/horizon is outside the forecast")
    if state.transfers_made and any(c == "freehit" for c in (chips or {}).values()):
        # Full prior-deadline restoration requires an explicitly frozen prior state.
        if state.deadline_lots is None or state.deadline_bank is None:
            raise ValueError("Free Hit needs prior deadline state after earlier transfers")
    ids = sorted(p for p in players if players[p].selectable or p in {lot.player_id for lot in state.lots})
    events = list(range(state.event, end_event + 1))
    for schedule in (chips, pin_chips):
        if schedule and any(
            t not in events or c not in ("wildcard", "freehit", "bboost", "3xc") for t, c in schedule.items()
        ):
            raise ValueError("Chip schedule is outside the plan horizon or contains an unknown chip")
    means = {(r["event"], r["player_id"]): max(0, r["mean"]) for r in forecast["players"]}
    lots = {lot.player_id: lot for lot in state.lots}
    old_ids = set(lots)
    sell_quotes = {p: quote(lot, players[p].price) for p, lot in lots.items()}
    # Difference between market cost and original sale proceeds is tracked until
    # that exact ownership lot is sold. A later repurchase creates a fresh lot.
    loss = {p: players[p].price - sell_quotes[p] for p in old_ids}
    mip = MIP()
    y, z, xi, captain, incoming, outgoing, original, sold_original = {}, {}, {}, {}, {}, {}, {}, {}
    bank, chip_vars, arcs, ft_next = {}, {}, {}, {}
    bb_x, bb_z, tc = {}, {}, {}
    terminal = {
        p: sum(means.get((e, p), 0) for e in range(end_event + 1, min(38, end_event + 4) + 1)) * 0.1
        for p in ids
    }
    for t in events:
        for chip in ("wildcard", "freehit", "bboost", "3xc"):
            allowed = False
            try:
                a, b = rules.chip_window(chip, t)
                allowed = not any(c.chip == chip and a <= c.event <= b for c in state.used_chips)
            except ValueError:
                pass
            if chip in disabled_chips:
                allowed = False
            if chip == "freehit" and state.transfers_made:
                # Keep prior-event transfer restoration out of the generator;
                # deterministic state engine supports it and validates imports.
                allowed = False
            if chips is not None or not automatic_chips:
                allowed = allowed and (chips or {}).get(t) == chip
            fixed = (chips or pin_chips or {}).get(t) == chip
            if fixed and not allowed:
                raise ValueError(f"Requested {chip} at GW{t} is unavailable")
            chip_vars[t, chip] = mip.var(("chip", t, chip), lower=int(fixed), upper=int(allowed))
        bank[t] = mip.var(("bank", t), upper=10000, integer=True)
        ft_next[t] = mip.var(("ft", t), upper=rules.ft_cap, integer=True)
        for p in ids:
            value = means.get((t, p), 0)
            y[t, p] = mip.var(("squad", t, p), cost=terminal[p] if t == end_event else 0)
            z[t, p] = mip.var(("playing_squad", t, p))
            xi[t, p] = mip.var(("xi", t, p), cost=value)
            captain[t, p] = mip.var(("captain", t, p), cost=value)
            incoming[t, p] = mip.var(("buy", t, p), upper=int(players[p].selectable))
            outgoing[t, p] = mip.var(("sell", t, p))
            if p in old_ids:
                original[t, p] = mip.var(("original", t, p))
                sold_original[t, p] = mip.var(("sold_original", t, p))
            bb_z[t, p] = mip.var(
                ("bb_squad", t, p), cost=value, integer=False, upper=mip.upper[chip_vars[t, "bboost"]]
            )
            bb_x[t, p] = mip.var(("bb_xi", t, p), upper=0, integer=False)
            if mip.upper[chip_vars[t, "bboost"]]:
                v = bb_z[t, p]
                mip.row({v: 1, z[t, p]: -1, xi[t, p]: 1}, upper=0)
                mip.row({v: 1, chip_vars[t, "bboost"]: -1}, upper=0)
                mip.row({v: 1, z[t, p]: -1, xi[t, p]: 1, chip_vars[t, "bboost"]: -1}, lower=-1)
            tc[t, p] = mip.and_var(("tc", t, p), captain[t, p], chip_vars[t, "3xc"], value)
    for t in events:
        previous = t - 1
        wc, fh = chip_vars[t, "wildcard"], chip_vars[t, "freehit"]
        mip.row([(chip_vars[t, c], 1) for c in ("wildcard", "freehit", "bboost", "3xc")], upper=1)
        if previous in events:
            mip.row({fh: 1, chip_vars[previous, "freehit"]: 1}, upper=1)
        elif any(c.chip == "freehit" and c.event == previous for c in state.used_chips):
            mip.equal({fh: 1}, 0)
        mip.equal([(y[t, p], 1) for p in ids], 15)
        mip.equal([(z[t, p], 1) for p in ids], 15)
        mip.equal([(xi[t, p], 1) for p in ids], 11)
        mip.equal([(captain[t, p], 1) for p in ids], 1)
        # Tight linking cuts avoid fractional chip copies across hundreds of
        # bench players in the root relaxation.
        mip.equal([(bb_z[t, p], 1) for p in ids] + [(chip_vars[t, "bboost"], -4)])
        mip.equal([(tc[t, p], 1) for p in ids] + [(chip_vars[t, "3xc"], -1)])
        mip.equal([(incoming[t, p], 1) for p in ids] + [(outgoing[t, p], -1) for p in ids])
        for pos, count in rules.composition.items():
            members = [p for p in ids if players[p].position == pos]
            mip.equal([(y[t, p], 1) for p in members], count)
            mip.equal([(z[t, p], 1) for p in members], count)
            mip.row([(xi[t, p], 1) for p in members], rules.minimum[pos], rules.maximum[pos])
        for club in market["teams"]:
            members = [p for p in ids if players[p].team == club["id"]]
            mip.row([(y[t, p], 1) for p in members], upper=rules.club_limit)
            mip.row([(z[t, p], 1) for p in members], upper=rules.club_limit)
        budget = [(bank[t], 1)]
        if previous in events:
            budget.append((bank[previous], -1))
        for p in ids:
            owned_prev = int(p in old_ids) if t == state.event else None
            terms = [(y[t, p], 1), (incoming[t, p], -1), (outgoing[t, p], 1)]
            if previous in events:
                terms.append((y[previous, p], -1))
            mip.equal(terms, owned_prev or 0)
            mip.row({incoming[t, p]: 1, outgoing[t, p]: 1}, upper=1)
            mip.row({incoming[t, p]: 1, fh: 1}, upper=1)
            mip.row({outgoing[t, p]: 1, fh: 1}, upper=1)
            # Actual scoring squad may differ from underlying ownership only on FH.
            mip.row({z[t, p]: 1, y[t, p]: -1, fh: -1}, upper=0)
            mip.row({y[t, p]: 1, z[t, p]: -1, fh: -1}, upper=0)
            mip.row({xi[t, p]: 1, z[t, p]: -1}, upper=0)
            mip.row({captain[t, p]: 1, xi[t, p]: -1}, upper=0)
            budget.extend([(incoming[t, p], players[p].price), (outgoing[t, p], -players[p].price)])
            if p in old_ids:
                mip.row({original[t, p]: 1, y[t, p]: -1}, upper=0)
                terms = [(original[t, p], 1), (sold_original[t, p], 1)]
                if previous in events:
                    terms.append((original[previous, p], -1))
                mip.equal(terms, 1 if t == state.event else 0)
                mip.row({sold_original[t, p]: 1, outgoing[t, p]: -1}, upper=0)
                # An original lot cannot disappear without a sale, or survive one.
                mip.row({original[t, p]: 1, outgoing[t, p]: 1}, upper=1)
                budget.append((sold_original[t, p], loss[p]))
        mip.equal(budget, state.bank if t == state.event else 0)
        # Exact temporary affordability with retained original lots valued at sell
        # quote on BOTH sides, rather than charging their current buy price.
        big_m = sum(sorted((players[p].price for p in ids), reverse=True)[:15])
        temporary = [(z[t, p], players[p].price) for p in ids] + [(fh, big_m)]
        available = state.bank + sum(sell_quotes.values()) if t == state.event else 0
        if previous in events:
            temporary.append((bank[previous], -1))
            temporary.extend((y[previous, p], -players[p].price) for p in ids)
            temporary.extend((original[previous, p], loss[p]) for p in old_ids)
        for p in old_ids:
            if previous in events:
                retained = mip.and_var(("fh_retained", t, p), original[previous, p], z[t, p])
                temporary.append((retained, -loss[p]))
            else:
                temporary.append((z[t, p], -loss[p]))
        mip.row(temporary, upper=big_m + available)
        # Exact FT state network, including capped rollover and chip preservation.
        event_arcs = []
        for f in range(rules.ft_cap + 1):
            for n in range(16):
                for preserve in (False, True):
                    if n > max_transfers and not preserve:
                        continue
                    if preserve and (not mip.upper[wc] and (not mip.upper[fh] or n > 0)):
                        continue
                    prior_made = state.transfers_made if t == state.event else 0
                    hit_paid = state.hit_paid if t == state.event else 0
                    hit = (
                        0
                        if preserve or (state.initial and t == state.event)
                        else max(0, n + prior_made - f) * rules.hit
                    ) - hit_paid
                    v = mip.var(("arc", t, f, n, preserve), cost=-hit)
                    next_f = f if preserve else min(rules.ft_cap, max(0, f - n - prior_made) + 1)
                    allocation = rules.event_overrides.get(t + 1, {}).get("rules", {}).get("free_transfers")
                    if allocation is not None:
                        next_f = int(allocation)
                    event_arcs.append((v, f, n, preserve, next_f))
        arcs[t] = event_arcs
        mip.equal([(v, 1) for v, _, _, _, _ in event_arcs], 1)
        ft_terms = [(v, f) for v, f, _, _, _ in event_arcs]
        if previous in events:
            ft_terms.append((ft_next[previous], -1))
        mip.equal(ft_terms, state.free_transfers if t == state.event else 0)
        mip.equal([(v, n) for v, _, n, _, _ in event_arcs] + [(incoming[t, p], -1) for p in ids])
        mip.equal([(v, int(pr)) for v, _, _, pr, _ in event_arcs] + [(wc, -1), (fh, -1)])
        mip.equal([(v, nf) for v, _, _, _, nf in event_arcs] + [(ft_next[t], -1)])
    for chip in ("wildcard", "freehit", "bboost", "3xc"):
        for window in {(c["start_event"], c["stop_event"]) for c in rules.chips if c["name"] == chip}:
            mip.row([(chip_vars[t, chip], 1) for t in events if window[0] <= t <= window[1]], upper=1)
    if forced_first:
        for p in ids:
            mip.equal({incoming[state.event, p]: 1}, int(p in forced_first.buy))
            mip.equal({outgoing[state.event, p]: 1}, int(p in forced_first.sell))
    for event, sell_ids in (forced_sell or {}).items():
        for p in sell_ids:
            mip.equal({outgoing[event, p]: 1}, 1)
    # Supply an independently feasible incumbent. Time limits must still return a
    # legal usable plan, even when the large joint-chip formulation has a weak LP.
    warm = np.zeros(len(mip.costs))
    warm_state = state
    surviving_original = set(old_ids)
    warm_chips = dict(chips or pin_chips or {})
    if chips is None and automatic_chips:
        # Whole-plan MIP may change these dates; these are incumbent seeds only.
        for kind in ("freehit", "bboost", "3xc"):
            if kind in warm_chips.values():
                continue
            allowed_events = [t for t in events if mip.upper[chip_vars[t, kind]] and t not in warm_chips]
            if not allowed_events:
                continue
            if kind == "freehit":
                scores = {}
                for t in allowed_events:
                    q = {p: means.get((t, p), 0) for p in ids}
                    regular = best_lineup(list(old_ids), q, players, rules)
                    # Relaxed score gain chooses a seed date; an actual affordable
                    # full-market squad is solved at that date below.
                    scores[t] = sum(sorted(q.values(), reverse=True)[:11]) - sum(q[p] for p in regular.xi)
                warm_chips[max(scores, key=lambda event: scores[event])] = kind
            elif kind == "bboost":

                def bench_value(t):
                    q = {p: means.get((t, p), 0) for p in ids}
                    return sum(q[p] for p in best_lineup(list(old_ids), q, players, rules).bench)

                warm_chips[max(allowed_events, key=bench_value)] = kind
            else:
                warm_chips[max(allowed_events, key=lambda t: max(means.get((t, p), 0) for p in old_ids))] = (
                    kind
                )
    for t in events:
        seed_chip = warm_chips.get(t)
        to_sell = (forced_sell or {}).get(t, [])
        action = (
            forced_first.model_copy(deep=True)
            if forced_first and t == state.event
            else Action(chip=cast(Chip | None, seed_chip))
        )
        action.chip = cast(Chip | None, seed_chip)
        current_ids = {lot.player_id for lot in warm_state.lots}
        reserved = set(current_ids) - set(to_sell)
        if not to_sell and not forced_first:
            if seed_chip in ("freehit", "wildcard"):
                picked = select_squad(
                    warm_state, players, rules, {p: means.get((t, p), 0) for p in ids}, seconds=1
                )
                action.sell = sorted(current_ids - set(picked))
                action.buy = sorted(set(picked) - current_ids)
            else:
                near = {
                    p: sum(
                        means.get((e, p), 0) * (0.85 ** (e - t)) for e in range(t, min(t + 3, end_event + 1))
                    )
                    for p in ids
                }
                bundle = greedy_bundle(warm_state, players, rules, near, min(2, max_transfers))
                action.sell, action.buy = bundle.sell, bundle.buy
        if to_sell and not action.buy:
            allowance = warm_state.bank + sum(
                quote(lot, players[lot.player_id].price)
                for lot in warm_state.lots
                if lot.player_id in to_sell
            )
            for leaving in to_sell:
                eligible = [
                    p
                    for p in ids
                    if p not in reserved
                    and players[p].selectable
                    and players[p].position == players[leaving].position
                    and players[p].price <= allowance
                    and sum(players[q].team == players[p].team for q in reserved) < 3
                ]
                if not eligible:
                    raise ValueError("No affordable warm-start replacement for forced sale")
                entering = max(eligible, key=lambda p: means.get((t, p), 0))
                action.sell.append(leaving)
                action.buy.append(entering)
                reserved.add(entering)
                allowance -= players[entering].price
        playing = sorted(current_ids - set(action.sell) | set(action.buy))
        action.lineup = best_lineup(playing, {p: means.get((t, p), 0) for p in ids}, players, rules)
        next_warm, warm_row = transition(warm_state, action, players, rules)
        permanent = {lot.player_id for lot in next_warm.lots}
        ordinary_buy = [] if seed_chip == "freehit" else action.buy
        ordinary_sell = [] if seed_chip == "freehit" else action.sell
        sold_old = surviving_original & set(ordinary_sell)
        surviving_original -= sold_old
        for p in ids:
            for kind, on in [
                ("squad", p in permanent),
                ("playing_squad", p in playing),
                ("xi", p in action.lineup.xi),
                ("captain", p == action.lineup.captain),
                ("buy", p in ordinary_buy),
                ("sell", p in ordinary_sell),
                ("bb_squad", seed_chip == "bboost" and p in playing and p not in action.lineup.xi),
                ("bb_xi", False),
                ("tc", seed_chip == "3xc" and p == action.lineup.captain),
            ]:
                warm[mip.keys[kind, t, p]] = int(on)
            if p in old_ids:
                warm[original[t, p]] = int(p in surviving_original)
                warm[sold_original[t, p]] = int(p in sold_old)
                retained_key = ("fh_retained", t, p)
                if retained_key in mip.keys:
                    warm[mip.keys[retained_key]] = int(p in (surviving_original | sold_old) and p in playing)
        warm[bank[t]] = next_warm.bank
        warm[ft_next[t]] = next_warm.free_transfers
        if seed_chip:
            warm[chip_vars[t, seed_chip]] = 1
        arc_key = (
            "arc",
            t,
            warm_state.free_transfers,
            len(ordinary_buy),
            seed_chip in ("wildcard", "freehit"),
        )
        if arc_key in mip.keys:
            warm[mip.keys[arc_key]] = 1
        warm_state = next_warm
    solution, diagnostics = mip.solve(time_limit, warm)
    ledger = []
    current = state
    with np.load(store.artifact_path(forecast["scenario_path"]), allow_pickle=False) as worlds:
        for t in events:
            fh = solution[chip_vars[t, "freehit"]] > 0.5
            playing = [p for p in ids if solution[z[t, p]] > 0.5]
            new_xi = [p for p in ids if solution[xi[t, p]] > 0.5]
            cap = next(p for p in ids if solution[captain[t, p]] > 0.5)
            vice = max((p for p in new_xi if p != cap), key=lambda p: means.get((t, p), 0))
            bench = sorted(
                (p for p in playing if p not in new_xi),
                key=lambda p: (players[p].position != 1, -means.get((t, p), 0)),
            )
            lineup = Lineup(xi=new_xi, bench=bench, captain=cap, vice=vice)
            lineup = optimize_bench(
                lineup, worlds[f"points_{t}"], worlds[f"minutes_{t}"], forecast["player_ids"], players, rules
            )
            old = {lot.player_id for lot in current.lots}
            sell = sorted(old - set(playing)) if fh else [p for p in ids if solution[outgoing[t, p]] > 0.5]
            buy = sorted(set(playing) - old) if fh else [p for p in ids if solution[incoming[t, p]] > 0.5]
            played_chip = next(
                (c for c in ("wildcard", "freehit", "bboost", "3xc") if solution[chip_vars[t, c]] > 0.5), None
            )
            action = Action(sell=sell, buy=buy, chip=cast(Chip | None, played_chip), lineup=lineup)
            current, row = transition(current, action, players, rules)
            if current.bank != round(solution[bank[t]]) and not fh:
                raise ValueError("Independent budget validator disagrees with solver")
            if current.free_transfers != round(solution[ft_next[t]]):
                raise ValueError("Independent FT validator disagrees with solver")
            row["expected_additive_net"] = (
                sum(means[t, p] for p in (playing if played_chip == "bboost" else new_xi))
                + means[t, cap] * (2 if played_chip == "3xc" else 1)
                - row["hit"]
            )
            row["assumptions"] = [
                "Constant market prices; refresh sell quotes before execution",
                "Future transfers conditional on refreshed minutes, availability and fixtures",
            ]
            ledger.append(row)
    diagnostics["runtime_seconds"] = round(time.monotonic() - started, 3)
    diagnostics["formulation"] = (
        "full-market multiweek MIP; additive XI/captain, FT network, original lots, joint chips"
    )
    diagnostics["market_players"] = len(ids)
    diagnostics["screened_players"] = 0
    diagnostics["terminal_value"] = (
        "0.1 × following four weeks squad xPts; sensitivity assumption, not learned value"
    )
    plan = {
        "created_at": now(),
        "state_id": state_id,
        "forecast_id": forecast_id,
        "market_id": market["id"],
        "start_event": state.event,
        "end_event": end_event,
        "ledger": ledger,
        "solver": diagnostics,
        "state_status": state.status,
        "execution_ready": state.status == "confirmed",
        "rules_hash": rules.hash,
        "input_hashes": forecast["input_hashes"],
        "final_state": current.model_dump(),
        "objective": "expected_points",
        "validation": {"legal": True, "affordable": True, "independent_transition": True},
        "contingencies": [
            {
                "trigger": "Player injury, benching or suspension confirmed",
                "action": "Recompute forecasts and re-solve before buying; retain hold as alternative",
            },
            {
                "trigger": "Price changes",
                "action": "Refresh buy and exact sell quotes, invalidate this price ledger and re-solve",
            },
            {
                "trigger": "Fixture postponed or double confirmed",
                "action": "Refresh fixtures and recompute every affected chip policy",
            },
            {
                "trigger": "New starter or role evidence",
                "action": "Add expiring evidence scenario and compare paired plan delta",
            },
        ],
        "limitations": [
            "MIP optimizes additive expectation; simulation rescores exact FPL autosubs",
            "Future prices and injury paths are assumptions; distant actions require re-optimization",
        ],
    }
    score_plan(store, plan, forecast)
    plan["id"] = store.put("plan", plan)
    return plan


def hold_plan(store: Store, state_id: str, forecast_id: str, end_event: int) -> dict:
    state = TeamState.model_validate({k: v for k, v in store.get("state", state_id).items() if k != "id"})
    players, market = load_market(store)
    forecast = store.get("forecast", forecast_id)
    rules = Rules.from_bootstrap(store.latest("bootstrap")[0])
    means = {(r["event"], r["player_id"]): r["mean"] for r in forecast["players"]}
    ledger = []
    for event in range(state.event, end_event + 1):
        lineup = best_lineup(
            [lot.player_id for lot in state.lots],
            {p: means.get((event, p), 0) for p in players},
            players,
            rules,
        )
        state, row = transition(state, Action(lineup=lineup), players, rules)
        ledger.append(row)
    plan = {
        "state_id": state_id,
        "forecast_id": forecast_id,
        "market_id": market["id"],
        "ledger": ledger,
        "label": "hold / roll",
        "created_at": now(),
        "validation": {"legal": True, "affordable": True},
    }
    score_plan(store, plan, forecast)
    plan["id"] = store.put("plan", plan)
    return plan
