"""Full-market multiweek MIP with lot accounting, FT network, and joint chips.

Additive expected XI/captain scores generate candidates. Exported plans are then
independently transitioned and rescored in shared worlds with exact autosubs.
"""

from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from typing import Mapping, cast

import highspy
import numpy as np
from scipy.sparse import coo_matrix

from .domain import Action, Chip, Lineup, TeamState
from .ingestion import load_market
from .rules import Rules, best_lineup, optimize_bench, quote, transition, validate_squad
from .simulation import score_plan
from .storage import Store, now


def incumbent_precedes_forecast(store, candidate, forecast):
    """A historical search cannot borrow an action generated with later inputs."""
    if (
        not forecast.get("cutoff")
        or not candidate.get("forecast_id")
        or candidate["forecast_id"] == forecast.get("id")
    ):
        return True
    from .availability import moment

    try:
        previous = store.get("forecast", candidate["forecast_id"])
    except ValueError:
        return False
    return bool(previous.get("cutoff") and moment(previous["cutoff"]) <= moment(forecast["cutoff"]))


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

    def solve(self, seconds, warm_start=None, relative_gap=0.01):
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
        solver.setOptionValue("mip_rel_gap", relative_gap)
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
            "best_bound": float(info.mip_dual_bound) if np.isfinite(info.mip_dual_bound) else None,
            "gap": float(info.mip_gap) if np.isfinite(info.mip_gap) else None,
            "bound_available": bool(np.isfinite(info.mip_dual_bound)),
            "nodes": info.mip_node_count,
            "variables": count,
            "constraints": len(self.row_lower),
        }


def select_squad(
    state,
    players,
    rules,
    means,
    seconds=1,
    initial_budget=1000,
    diagnostics=None,
    bench_weight=0.015,
    relative_gap=0.01,
    starting_eligible=None,
):
    """Full-market one-week squad oracle used only to seed the multiweek solve."""
    owned = {lot.player_id: lot for lot in state.lots} if state is not None else {}
    ids = [p for p in players if players[p].selectable or p in owned]
    costs = {p: quote(owned[p], players[p].price) if p in owned else players[p].price for p in ids}
    budget = state.bank + sum(costs[p] for p in owned) if state is not None else initial_budget
    model = MIP()
    y = {p: model.var(("y", p), cost=bench_weight * means.get(p, 0)) for p in ids}
    x = {p: model.var(("x", p), cost=means.get(p, 0)) for p in ids}
    if starting_eligible is not None:
        for p in ids:
            if p not in starting_eligible:
                model.upper[x[p]] = 0
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
    solution, info = model.solve(seconds, warm, relative_gap=relative_gap)
    if diagnostics is not None:
        diagnostics.update(info)
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
                if any(
                    sum(players[p].team == club for p in trial) > rules.club_limit
                    for club in {players[p].team for p in trial}
                ):
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


def feasible_incumbent(model, vector):
    """Check bounds, every row and integrality before offering a solver seed."""
    lhs = np.bincount(
        np.asarray(model.rows, dtype=int),
        weights=np.asarray(model.values) * vector[np.asarray(model.cols, dtype=int)],
        minlength=len(model.row_lower),
    )
    integer = np.asarray(model.integer, dtype=bool)
    return bool(
        np.isfinite(vector).all()
        and np.all(vector >= np.asarray(model.lower) - 1e-6)
        and np.all(vector <= np.asarray(model.upper) + 1e-6)
        and np.all(lhs >= np.asarray(model.row_lower) - 1e-6)
        and np.all(lhs <= np.asarray(model.row_upper) + 1e-6)
        and np.all(np.abs(vector[integer] - np.rint(vector[integer])) <= 1e-6)
    )


def relaxed_horizon_bound(players, rules, means, state, events, terminal, continuation, chip_allowed):
    """Transfer-free continuous squad relaxations bound the additive objective.

    Prices are fixed in this formulation. Original-lot market wealth therefore
    bounds every future squad, even before transfer/lot integrality is imposed.
    Free Hit/Wildcard are already relaxed by allowing a fresh squad every week.
    """
    ids = sorted(players)
    wealth = state.bank + sum(players[lot.player_id].price for lot in state.lots)
    extra = Counter(players[lot.player_id].team for lot in state.lots)
    totals, gains = [], defaultdict(list)
    for event in events:
        values = {}
        for mode in (None, "bboost", "3xc"):
            if mode and not chip_allowed.get((event, mode)):
                continue
            relaxation = MIP()
            y, x, c = {}, {}, {}
            for p in ids:
                value = means.get((event, p), 0)
                tail = terminal.get(p, 0) if event == events[-1] else 0
                y[p] = relaxation.var(("y", p), tail + (value if mode == "bboost" else 0), integer=False)
                x[p] = relaxation.var(("x", p), 0 if mode == "bboost" else value, integer=False)
                c[p] = relaxation.var(("c", p), value * (2 if mode == "3xc" else 1), integer=False)
                relaxation.row({x[p]: 1, y[p]: -1}, upper=0)
                relaxation.row({c[p]: 1, x[p]: -1}, upper=0)
            relaxation.equal([(y[p], 1) for p in ids], 15)
            relaxation.equal([(x[p], 1) for p in ids], 11)
            relaxation.equal([(c[p], 1) for p in ids], 1)
            relaxation.row([(y[p], players[p].price) for p in ids], upper=wealth)
            for pos, count in rules.composition.items():
                group = [p for p in ids if players[p].position == pos]
                relaxation.equal([(y[p], 1) for p in group], count)
                relaxation.row([(x[p], 1) for p in group], rules.minimum[pos], rules.maximum[pos])
            for club in {players[p].team for p in ids}:
                relaxation.row(
                    [(y[p], 1) for p in ids if players[p].team == club],
                    upper=max(rules.club_limit, extra[club]),
                )
            try:
                _, info = relaxation.solve(1)
            except ValueError:
                return None
            if info["status"] != "Optimal":
                return None
            values[mode] = float(info["objective"])
        totals.append(values[None])
        for mode in ("bboost", "3xc"):
            if mode in values:
                gains[mode, rules.chip_window(mode, event)].append(max(0, values[mode] - values[None]))
    reward = max(0, continuation["ft"]) * rules.ft_cap
    reward += max(0, continuation["bank"]) * min(wealth, continuation["bank_max"])
    reward += sum(max(0, v) for v in continuation["chip_use"].values())
    return sum(totals) + sum(max(v) for v in gains.values()) + reward


def optimize(
    store: Store,
    state_id: str,
    forecast_id: str,
    end_event: int,
    time_limit: float = 30,
    chips: Mapping[int, str] | None = None,
    max_transfers: int = 15,
    automatic_chips: bool = True,
    pin_chips: Mapping[int, str] | None = None,
    forced_sell: dict[int, list[int]] | None = None,
    disabled_chips: tuple[str, ...] = (),
    first_week_hit_budget: int | None = None,
    use_decision_critic: bool = True,
    relative_gap: float = 0.001,
    blocked_chips: Mapping[int, tuple[str, ...]] | None = None,
) -> dict:
    started = time.monotonic()
    if first_week_hit_budget is not None and first_week_hit_budget < 0:
        raise ValueError("Immediate hit budget cannot be negative")
    if time_limit <= 0 or not 0 <= max_transfers <= 15:
        raise ValueError("Solve time must be positive and per-event transfers must be between 0 and 15")
    players, market = load_market(store)
    state = TeamState.model_validate({k: v for k, v in store.get("state", state_id).items() if k != "id"})
    deadline = next((e.get("deadline_time") for e in market["events"] if e["id"] == state.event), None)
    if deadline and state.status != "dated_seed":
        from .availability import moment

        if moment(deadline) <= moment(now()):
            raise ValueError(
                "Account deadline has passed; refresh the revealed squad and next-gameweek forecasts before planning"
            )
    forecast = store.get("forecast", forecast_id)
    current_model = store.get("decision_model")
    if (
        not current_model.get("promoted")
        or forecast.get("model_version") != "learned-points-v1"
        or forecast.get("decision_model_id") != current_model["id"]
    ):
        raise ValueError("Forecast predates the validated current model; rebuild forecasts before solving")
    rules = Rules.from_bootstrap(store.latest("bootstrap")[0])
    validate_squad(state, players, rules)
    if forecast["market_id"] != market["id"] or forecast["rules_hash"] != rules.hash:
        raise ValueError("Forecast inputs are stale: refresh/recompute before solving")
    if not forecast["start_event"] <= state.event <= end_event <= forecast["end_event"]:
        raise ValueError("State/horizon is outside the forecast")
    requested_first_chip = (chips if chips is not None else pin_chips or {}).get(state.event)
    if state.active_chip is not None and requested_first_chip not in (None, state.active_chip):
        raise ValueError("Requested chip conflicts with an already activated chip")
    if state.active_chip == "freehit" or (requested_first_chip == "freehit" and state.transfers_made):
        if "freehit" in disabled_chips or "freehit" in (blocked_chips or {}).get(state.event, ()):
            raise ValueError("Requested Free Hit is disabled")
        if state.active_chip not in (None, "freehit"):
            raise ValueError("Requested chip conflicts with an already activated chip")
        if state.deadline_lots is None or state.deadline_bank is None:
            raise ValueError("Free Hit needs prior deadline state after earlier transfers")
        from .decision_runtime import conditional_plan
        from .strategy import solve_action

        current_means = {r["player_id"]: r["mean"] for r in forecast["players"] if r["event"] == state.event}
        first = solve_action(
            state,
            players,
            rules,
            current_means,
            chip="freehit",
            seconds=min(time_limit, 15),
            forced_sell=(forced_sell or {}).get(state.event, ()),
            relative_gap=relative_gap,
        )
        restored_plan = conditional_plan(
            store,
            state_id,
            forecast_id,
            end_event,
            first.action,
            time_limit,
            origin="Activated Free Hit with exact underlying restoration",
            continuation_options={
                "chips": None if chips is None else {t: c for t, c in chips.items() if t > state.event},
                "pin_chips": {t: c for t, c in (pin_chips or {}).items() if t > state.event},
                "forced_sell": {t: p for t, p in (forced_sell or {}).items() if t > state.event},
                "blocked_chips": {t: c for t, c in (blocked_chips or {}).items() if t > state.event},
                "disabled_chips": disabled_chips,
                "automatic_chips": automatic_chips,
                "max_transfers": max_transfers,
                "relative_gap": relative_gap,
            },
        )
        restored_plan["solver"]["first_action_solver"] = first.solver
        restored_plan["id"] = store.put("plan", {k: v for k, v in restored_plan.items() if k != "id"})
        return restored_plan
    if state.active_chip is not None:
        if (chips or pin_chips or {}).get(state.event, state.active_chip) != state.active_chip:
            raise ValueError("Requested chip conflicts with an already activated chip")
        pin_chips = {**(pin_chips or {}), state.event: state.active_chip}
    if (
        use_decision_critic
        and chips is None
        and pin_chips is None
        and not forced_sell
        and not blocked_chips
        and not disabled_chips
        and first_week_hit_budget is None
        and automatic_chips
        and store.list("decision_analysis", 1)
    ):
        analysis = store.get("decision_analysis")
        if (
            analysis.get("promotion_gate_passed")
            and analysis.get("state_id") == state_id
            and analysis.get("forecast_id") == forecast_id
            and analysis.get("market_id") == market["id"]
            and analysis.get("decision_critic_id") == store.get("decision_critic")["id"]
        ):
            from .decision_runtime import conditional_plan

            action = Action.model_validate(analysis["candidates"][analysis["selected_candidate"]]["action"])
            return conditional_plan(
                store, state_id, forecast_id, end_event, action, time_limit, origin=analysis["id"]
            )
    ids = sorted(p for p in players if players[p].selectable or p in {lot.player_id for lot in state.lots})
    events = list(range(state.event, end_event + 1))
    for schedule in (chips, pin_chips):
        if schedule and any(
            t not in events or c not in ("wildcard", "freehit", "bboost", "3xc") for t, c in schedule.items()
        ):
            raise ValueError("Chip schedule is outside the plan horizon or contains an unknown chip")
    if any(
        t not in events or any(c not in ("wildcard", "freehit", "bboost", "3xc") for c in names)
        for t, names in (blocked_chips or {}).items()
    ):
        raise ValueError("Blocked chip schedule is outside the plan horizon or contains an unknown chip")
    means = {(r["event"], r["player_id"]): r["mean"] for r in forecast["players"]}
    lots = {lot.player_id: lot for lot in state.lots}
    old_ids = set(lots)
    old_clubs = Counter(players[p].team for p in old_ids)
    grandfathered = {
        c: n - rules.club_limit
        for c, n in old_clubs.items()
        if n > rules.club_limit and c in state.grandfathered_clubs
    }
    changed_permanently: dict[int, int] = {}
    changed_scoring: dict[int, int] = {}
    sell_quotes = {p: quote(lot, players[p].price) for p, lot in lots.items()}
    # Difference between market cost and original sale proceeds is tracked until
    # that exact ownership lot is sold. A later repurchase creates a fresh lot.
    loss = {p: players[p].price - sell_quotes[p] for p in old_ids}
    mip = MIP()
    y, z, xi, captain, incoming, outgoing, original, sold_original = {}, {}, {}, {}, {}, {}, {}, {}
    bank, chip_vars, arcs, ft_next = {}, {}, {}, {}
    bb_x, bb_z, tc = {}, {}, {}
    from .decision_runtime import continuation_terms

    continuation = continuation_terms(store, end_event, means, players, rules, state)
    terminal = continuation["players"]
    for t in events:
        for chip in ("wildcard", "freehit", "bboost", "3xc"):
            allowed = False
            try:
                a, b = rules.chip_window(chip, t)
                allowed = not any(c.chip == chip and a <= c.event <= b for c in state.used_chips)
            except ValueError:
                pass
            if chip in disabled_chips or chip in (blocked_chips or {}).get(t, ()):
                allowed = False
            if chip == "freehit" and state.transfers_made and t == state.event:
                # Keep prior-event transfer restoration out of the generator;
                # deterministic state engine supports it and validates imports.
                allowed = False
            if chips is not None or not automatic_chips:
                allowed = allowed and (chips or {}).get(t) == chip
            fixed = (chips or pin_chips or {}).get(t) == chip
            if fixed and not allowed:
                raise ValueError(f"Requested {chip} at GW{t} is unavailable")
            chip_vars[t, chip] = mip.var(
                ("chip", t, chip),
                cost=continuation["chip_use"].get((t, chip), 0),
                lower=int(fixed),
                upper=int(allowed),
            )
        # With fixed market prices, selling/rebuying cannot create money.
        bank[t] = mip.var(("bank", t), cost=0, upper=state.bank + sum(sell_quotes.values()), integer=True)
        ft_next[t] = mip.var(
            ("ft", t), cost=continuation["ft"] if t == end_event else 0, upper=rules.ft_cap, integer=True
        )
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
        # Position-only score ceilings are valid even before the joint-chip
        # root relaxation finishes. They cut fractional/scattered chip rewards
        # without excluding any legal squad or relying on future outcomes.
        ranked = {
            pos: sorted((means.get((t, p), 0) for p in ids if players[p].position == pos), reverse=True)
            for pos in rules.composition
        }
        best_captain = max(means.get((t, p), 0) for p in ids)
        ordinary_ceiling = max(
            sum(ranked[1][:1])
            + sum(ranked[2][:d])
            + sum(ranked[3][:m])
            + sum(ranked[4][: 10 - d - m])
            + best_captain
            for d in range(rules.minimum[2], rules.maximum[2] + 1)
            for m in range(rules.minimum[3], rules.maximum[3] + 1)
            if rules.minimum[4] <= 10 - d - m <= rules.maximum[4]
        )
        bench_ceiling = (
            sum(sum(ranked[pos][:count]) for pos, count in rules.composition.items()) + best_captain
        )
        mip.row(
            [(xi[t, p], means.get((t, p), 0)) for p in ids]
            + [(captain[t, p], means.get((t, p), 0)) for p in ids]
            + [(bb_z[t, p], means.get((t, p), 0)) for p in ids]
            + [(tc[t, p], means.get((t, p), 0)) for p in ids]
            + [
                (chip_vars[t, "bboost"], ordinary_ceiling - bench_ceiling),
                (chip_vars[t, "3xc"], -best_captain),
            ],
            upper=ordinary_ceiling,
        )
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
            extra = grandfathered.get(club["id"], 0)
            if extra:
                if t not in changed_permanently:
                    moved = changed_permanently[t] = mip.var(("permanent_change", t))
                    scoring = changed_scoring[t] = mip.var(("scoring_change", t))
                    buys = [(incoming[t, p], 1) for p in ids]
                    mip.row(buys + [(moved, -15)], upper=0)
                    terms = [(moved, 1)] + [(incoming[t, p], -1) for p in ids]
                    if previous in changed_permanently:
                        prior = changed_permanently[previous]
                        mip.row({moved: 1, prior: -1}, lower=0)
                        terms.append((prior, -1))
                    mip.row(terms, upper=0)
                    different = [(z[t, p], 1) for p in ids if p not in old_ids]
                    mip.row(different + [(scoring, -15)], upper=0)
                    mip.row(different + [(scoring, -1)], lower=0)
                mip.row(
                    [(y[t, p], 1) for p in members] + [(changed_permanently[t], extra)],
                    upper=rules.club_limit + extra,
                )
                mip.row(
                    [(z[t, p], 1) for p in members] + [(changed_permanently[t], extra)],
                    upper=rules.club_limit + extra,
                )
                mip.row(
                    [(z[t, p], 1) for p in members] + [(changed_scoring[t], extra)],
                    upper=rules.club_limit + extra,
                )
            else:
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
                    if t == state.event and first_week_hit_budget is not None and hit > first_week_hit_budget:
                        continue
                    v = mip.var(("arc", t, f, n, preserve), cost=-hit)
                    next_f = (
                        f
                        if preserve
                        else (
                            1
                            if state.initial and t == state.event
                            else min(rules.ft_cap, max(0, f - n - prior_made) + 1)
                        )
                    )
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
    if continuation["bank"] > 0 and continuation["bank_max"] > 0:
        reward = mip.var(
            ("terminal_bank",), cost=continuation["bank"], upper=continuation["bank_max"], integer=False
        )
        mip.row({reward: 1, bank[end_event]: -1}, upper=0)
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

    def fill_warm(vector, t, before, action, after, row, survivors):
        warm = vector
        warm_state = before
        next_warm = after
        surviving_original = survivors
        seed_chip = action.chip
        playing = row["playing_squad"]
        permanent = {lot.player_id for lot in next_warm.lots}
        ordinary_buy = [] if seed_chip == "freehit" else action.buy
        ordinary_sell = [] if seed_chip == "freehit" else action.sell
        sold_old = surviving_original & set(ordinary_sell)
        surviving_original -= sold_old
        if t in changed_permanently:
            before_changed = (
                bool(vector[changed_permanently[t - 1]]) if t - 1 in changed_permanently else False
            )
            warm[changed_permanently[t]] = int(before_changed or bool(ordinary_buy))
            warm[changed_scoring[t]] = int(bool(set(playing) - old_ids))
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
        if t == end_event and ("terminal_bank",) in mip.keys:
            warm[mip.keys[("terminal_bank",)]] = min(next_warm.bank, continuation["bank_max"])
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

    for t in events:
        seed_chip = warm_chips.get(t)
        to_sell = (forced_sell or {}).get(t, [])
        action = Action(chip=cast(Chip | None, seed_chip))
        action.chip = cast(Chip | None, seed_chip)
        current_ids = {lot.player_id for lot in warm_state.lots}
        reserved = set(current_ids) - set(to_sell)
        if not to_sell:
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
                warm_limit = min(2, max_transfers)
                if t == state.event and first_week_hit_budget is not None and not state.initial:
                    warm_limit = min(
                        warm_limit,
                        max(
                            0,
                            warm_state.free_transfers
                            - warm_state.transfers_made
                            + (first_week_hit_budget + warm_state.hit_paid) // rules.hit,
                        ),
                    )
                bundle = greedy_bundle(warm_state, players, rules, near, warm_limit)
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
        fill_warm(warm, t, warm_state, action, next_warm, warm_row, surviving_original)
        warm_state = next_warm
    reused_plan = None
    objective_before = float(np.dot(mip.costs, warm))
    warm_value = objective_before
    exact_seed_count = 0
    # A time-limited joint-chip search often never improves its first incumbent.
    # Offer full-market action solves as additional legal seeds. Their complete
    # trajectories must improve THIS objective before they can replace a seed.
    from .strategy import solve_action

    seed_schedules = [dict(chips or pin_chips or {})]
    if warm_chips != seed_schedules[0]:
        seed_schedules.append(dict(warm_chips))
    wildcard_dates = [t for t in events if mip.upper[chip_vars[t, "wildcard"]] and t not in warm_chips]
    if chips is None and automatic_chips and wildcard_dates:
        seed_schedules.append({**warm_chips, wildcard_dates[0]: "wildcard"})
    for schedule in seed_schedules:
        vector = np.zeros(len(mip.costs))
        trial = state.model_copy(deep=True)
        survivors = set(old_ids)
        try:
            for t in events:
                seed_kind = schedule.get(t)
                limit = min(max_transfers, max(2, len((forced_sell or {}).get(t, []))))
                if t == state.event and first_week_hit_budget is not None and not state.initial:
                    limit = min(
                        limit,
                        max(
                            0,
                            trial.free_transfers
                            - trial.transfers_made
                            + (first_week_hit_budget + trial.hit_paid) // rules.hit,
                        ),
                    )
                future = {
                    p: 0.15 * sum(means.get((e, p), 0) for e in range(t + 1, min(t + 3, end_event) + 1))
                    for p in ids
                }
                selected = solve_action(
                    trial,
                    players,
                    rules,
                    {p: means.get((t, p), 0) for p in ids},
                    chip=seed_kind,
                    max_transfers=limit,
                    seconds=0.3,
                    squad_value=future,
                    forced_sell=(forced_sell or {}).get(t, []),
                )
                fill_warm(vector, t, trial, selected.action, selected.state, selected.ledger, survivors)
                trial = selected.state
            if not feasible_incumbent(mip, vector):
                continue
        except ValueError:
            continue
        exact_seed_count += 1
        value = float(np.dot(mip.costs, vector))
        if value > warm_value + 1e-7:
            warm, warm_value = vector, value
    candidate_count = 0
    for cached_kind, item in [
        ("plan", item) for item in store.list_matching("plan", {"state_id": state_id}, 100)
    ] + [("warm_start", item) for item in store.list_matching("warm_start", {"state_id": state_id}, 30)]:
        cached = store.get(cached_kind, item["id"])
        if not incumbent_precedes_forecast(store, cached, forecast):
            continue
        prior = cached.get("ledger", [])
        if (
            cached.get("state_id") != state_id
            or not cached.get("validation", {}).get("legal")
            or [r["event"] for r in prior] != events
        ):
            continue
        vector = np.zeros(len(mip.costs))
        trial = state.model_copy(deep=True)
        survivors = set(old_ids)
        try:
            for t, row in zip(events, prior):
                action = Action(
                    sell=[r["player_id"] for r in row["sell"]],
                    buy=[r["player_id"] for r in row["buy"]],
                    chip=row["chip"],
                )
                playing = sorted({lot.player_id for lot in trial.lots} - set(action.sell) | set(action.buy))
                action.lineup = best_lineup(playing, {p: means.get((t, p), 0) for p in ids}, players, rules)
                following, checked = transition(trial, action, players, rules)
                fill_warm(vector, t, trial, action, following, checked, survivors)
                trial = following
            if not feasible_incumbent(mip, vector):
                continue
        except (ValueError, KeyError):
            continue
        candidate_count += 1
        value = float(np.dot(mip.costs, vector))
        if value >= warm_value - 1e-7:
            warm = vector
            warm_value = value
            reused_plan = item["id"]
    relaxation_bound = relaxed_horizon_bound(
        {p: players[p] for p in ids},
        rules,
        means,
        state,
        events,
        terminal,
        continuation,
        {key: bool(mip.upper[value]) for key, value in chip_vars.items()},
    )
    solution, diagnostics = mip.solve(time_limit, warm, relative_gap=relative_gap)
    if relaxation_bound is not None:
        if relaxation_bound < diagnostics["objective"] - 1e-5:
            raise ValueError("Independent relaxation bound disagrees with the feasible incumbent")
        # Numerical tolerances cannot make a valid incumbent exceed its bound.
        relaxation_bound = max(relaxation_bound, diagnostics["objective"])
        reported = diagnostics["best_bound"]
        diagnostics["raw_solver_best_bound"] = reported
        diagnostics["independent_relaxation_bound"] = relaxation_bound
        diagnostics["best_bound"] = (
            min(reported, relaxation_bound) if reported is not None else relaxation_bound
        )
        diagnostics["gap"] = max(
            0, (diagnostics["best_bound"] - diagnostics["objective"]) / max(1, abs(diagnostics["objective"]))
        )
        diagnostics["bound_available"] = True
    diagnostics.update(
        initial_warm_objective=objective_before,
        selected_warm_objective=warm_value,
        reused_incumbent_plan_id=reused_plan,
        validated_cached_incumbents=candidate_count,
        validated_full_market_seeds=exact_seed_count,
    )
    ledger = []
    current = state
    from .worlds import load_worlds

    with load_worlds(store, forecast) as worlds:
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
            cap = lineup.captain
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
    diagnostics["terminal_value"] = continuation["method"]
    diagnostics["state_value_model_id"] = continuation["model_id"]
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
        "execution_ready": state.status == "confirmed"
        and forecast.get("model_version") == "learned-points-v1",
        "rules_hash": rules.hash,
        "input_hashes": forecast["input_hashes"],
        "final_state": current.model_dump(),
        "objective": "expected_points",
        "first_week_hit_budget": first_week_hit_budget,
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
    if (
        state.transfers_made
        and state.deadline_lots is not None
        and state.deadline_bank is not None
        and chips is None
        and automatic_chips
        and state.event not in (pin_chips or {})
        and "freehit" not in disabled_chips
        and "freehit" not in (blocked_chips or {}).get(state.event, ())
    ):
        from .strategy import allowed_chips

        if "freehit" in allowed_chips(state, rules):
            restored = optimize(
                store,
                state_id,
                forecast_id,
                end_event,
                time_limit,
                max_transfers=max_transfers,
                pin_chips={**(pin_chips or {}), state.event: "freehit"},
                forced_sell=forced_sell,
                disabled_chips=disabled_chips,
                first_week_hit_budget=first_week_hit_budget,
                use_decision_critic=False,
                relative_gap=relative_gap,
                blocked_chips=blocked_chips,
            )
            if restored["simulation"]["mean"] > plan["simulation"]["mean"]:
                return restored
    return plan


def hold_plan(store: Store, state_id: str, forecast_id: str, end_event: int) -> dict:
    state = TeamState.model_validate({k: v for k, v in store.get("state", state_id).items() if k != "id"})
    players, market = load_market(store)
    forecast = store.get("forecast", forecast_id)
    rules = Rules.from_bootstrap(store.latest("bootstrap")[0])
    means = {(r["event"], r["player_id"]): r["mean"] for r in forecast["players"]}
    ledger = []
    from .worlds import load_worlds

    with load_worlds(store, forecast) as worlds:
        for event in range(state.event, end_event + 1):
            lineup = best_lineup(
                [lot.player_id for lot in state.lots],
                {p: means.get((event, p), 0) for p in players},
                players,
                rules,
            )
            lineup = optimize_bench(
                lineup,
                worlds[f"points_{event}"],
                worlds[f"minutes_{event}"],
                forecast["player_ids"],
                players,
                rules,
            )
            state, row = transition(state, Action(lineup=lineup), players, rules)
            row["expected_additive_net"] = (
                sum(
                    means.get((event, p), 0)
                    for p in (row["playing_squad"] if row["chip"] == "bboost" else lineup.xi)
                )
                + means.get((event, lineup.captain), 0) * (2 if row["chip"] == "3xc" else 1)
                - row["hit"]
            )
            ledger.append(row)
    plan = {
        "state_id": state_id,
        "forecast_id": forecast_id,
        "market_id": market["id"],
        "ledger": ledger,
        "label": "hold / roll",
        "start_event": ledger[0]["event"],
        "end_event": end_event,
        "rules_hash": rules.hash,
        "input_hashes": forecast["input_hashes"],
        "state_status": state.status,
        "execution_ready": state.status == "confirmed",
        "final_state": state.model_dump(),
        "objective": "hold_squad_with_weekly_lineup_optimization",
        "solver": {
            "status": "Fixed hold policy; weekly legal lineup search",
            "gap": None,
            "best_bound": None,
            "runtime_seconds": 0,
            "scope": "A comparator policy, with no claim of a global transfer/chip bound",
        },
        "created_at": now(),
        "validation": {"legal": True, "affordable": True},
    }
    score_plan(store, plan, forecast)
    plan["id"] = store.put("plan", plan)
    return plan
