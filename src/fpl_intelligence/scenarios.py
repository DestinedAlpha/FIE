"""Observable scenario branches with shared prefixes and balanced price ledgers."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from pydantic import Field, model_validator

from .domain import Action, Lineup, Strict, TeamState
from .ingestion import load_market
from .optimization import greedy_bundle, select_squad
from .rules import Rules, best_lineup, selling_price, transition
from .simulation import forecast, score_plan
from .storage import Store, atomic_json, now


class PlayerChange(Strict):
    player_id: int = Field(ge=1)
    from_event: int = Field(ge=1, le=38)
    through_event: int = Field(default=38, ge=1, le=38)
    availability: float | None = Field(default=None, ge=0, le=1)
    parameters: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid(self):
        bounds = {
            "p_start": (0, 1),
            "p_cameo": (0, 1),
            "p_long_start": (0, 1),
            "start_minutes": (1, 90),
            "minutes_cap": (1, 90),
            "goals90": (0, 2),
            "assists90": (0, 2),
        }
        if self.through_event < self.from_event:
            raise ValueError("Invalid scenario time interval")
        if any(k not in bounds or not bounds[k][0] <= v <= bounds[k][1] for k, v in self.parameters.items()):
            raise ValueError("Unbounded or unknown scenario parameter")
        return self


class PriceChange(Strict):
    event: int = Field(ge=1, le=38)
    player_id: int = Field(ge=1)
    price: int = Field(ge=0, le=300)


class FixtureChange(Strict):
    fixture_id: int = Field(ge=1)
    event: int | None = Field(default=None, ge=1, le=38)


class ScenarioSpec(Strict):
    name: str = Field(min_length=3, max_length=150)
    reveal_event: int = Field(ge=1, le=38)
    probability: float | None = Field(default=None, gt=0, le=1)
    players: list[PlayerChange] = Field(default_factory=list)
    prices: list[PriceChange] = Field(default_factory=list)
    fixture_events: list[FixtureChange] = Field(default_factory=list)
    sale_assumption: Literal["conservative", "optimistic"] = "conservative"
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def nonanticipative(self):
        if any(c.from_event < self.reveal_event for c in self.players) or any(
            c.event < self.reveal_event for c in self.prices
        ):
            raise ValueError("Scenario changes cannot precede their observation event")
        if any(c.event is not None and c.event < self.reveal_event for c in self.fixture_events):
            raise ValueError("A rearranged fixture cannot occur before the scenario is observed")
        return self


def sale_bounds(lot, current_price: int, new_price: int) -> tuple[int, int]:
    """All possible future sale values consistent with the exact CURRENT quote."""
    if lot.purchase_price is not None:
        value = selling_price(lot.purchase_price, new_price)
        return value, value
    if lot.sell_price == current_price:
        if new_price <= current_price:
            return new_price, new_price
        return selling_price(current_price, new_price), new_price
    candidates = [2 * lot.sell_price - current_price, 2 * lot.sell_price - current_price + 1]
    valid = [p for p in candidates if p >= 0 and selling_price(p, current_price) == lot.sell_price]
    if not valid:
        raise ValueError("Supplied quote cannot establish a lawful gain-sharing scenario")
    values = [selling_price(p, new_price) for p in valid]
    return min(values), max(values)


def conditional_lots(state, players, spec):
    """Select a consistent purchase-lot hypothesis; never label it actual acquisition history."""
    maximum = {
        p: max([player.price] + [c.price for c in spec.prices if c.player_id == p])
        for p, player in players.items()
    }
    lots = []
    assumptions = []
    for lot in state.lots:
        if lot.purchase_price is not None:
            lots.append(lot)
            continue
        current = players[lot.player_id].price
        if lot.sell_price == current:
            assumed = current if spec.sale_assumption == "conservative" else maximum[lot.player_id]
        else:
            possible = [2 * lot.sell_price - current, 2 * lot.sell_price - current + 1]
            possible = [p for p in possible if p >= 0 and selling_price(p, current) == lot.sell_price]
            if not possible:
                raise ValueError("Current sale quote is inconsistent with price-scenario assumptions")
            assumed = min(possible) if spec.sale_assumption == "conservative" else max(possible)
        lots.append(
            lot.model_copy(
                update={"purchase_price": assumed, "provenance": "conditional_purchase_lot_hypothesis"}
            )
        )
        assumptions.append(
            {
                "player_id": lot.player_id,
                "assumed_purchase": assumed,
                "actual_purchase_known": False,
                "current_exact_sell": lot.sell_price,
                "mode": spec.sale_assumption,
            }
        )
    return state.model_copy(update={"lots": lots}), assumptions


def action_from_row(row):
    return Action(
        sell=[v["player_id"] for v in row["sell"]],
        buy=[v["player_id"] for v in row["buy"]],
        chip=row["chip"],
        lineup=Lineup.model_validate(row["lineup"]),
    )


def rollout(store: Store, base: dict, branch_forecast: dict, spec: ScenarioSpec) -> dict:
    players, market = load_market(store, base["market_id"])
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    state = TeamState.model_validate(
        {k: v for k, v in store.get("state", base["state_id"]).items() if k != "id"}
    )
    state, lot_assumptions = conditional_lots(state, players, spec)
    means = {(r["event"], r["player_id"]): r["mean"] for r in branch_forecast["players"]}
    current_players = dict(players)
    ledger = []
    for original in base["ledger"]:
        event = original["event"]
        for change in spec.prices:
            if change.event == event:
                if change.player_id not in current_players:
                    raise ValueError("Unknown scenario price identity")
                current_players[change.player_id] = current_players[change.player_id].model_copy(
                    update={"price": change.price}
                )
        quotes = [
            lot
            if current_players[lot.player_id].price == lot.quote_market_price
            else lot.model_copy(
                update={
                    "sell_price": selling_price(
                        cast(int, lot.purchase_price), current_players[lot.player_id].price
                    ),
                    "quote_market_price": current_players[lot.player_id].price,
                }
            )
            for lot in state.lots
        ]
        state = state.model_copy(update={"lots": quotes})
        if event < spec.reveal_event:
            action = action_from_row(original)
        else:
            near = {
                p: sum(
                    means.get((e, p), 0) * (0.85 ** (e - event))
                    for e in range(event, min(event + 3, base["end_event"] + 1))
                )
                for p in players
            }
            chip = original["chip"]
            action = greedy_bundle(state, current_players, rules, near, 2)
            action.chip = chip
            if chip in ("freehit", "wildcard"):
                chosen = select_squad(
                    state, current_players, rules, {p: means.get((event, p), 0) for p in players}, seconds=1
                )
                owned = {lot.player_id for lot in state.lots}
                action.sell = sorted(owned - set(chosen))
                action.buy = sorted(set(chosen) - owned)
            squad = sorted({lot.player_id for lot in state.lots} - set(action.sell) | set(action.buy))
            action.lineup = best_lineup(
                squad, {p: means.get((event, p), 0) for p in players}, current_players, rules
            )
        state, row = transition(state, action, current_players, rules)
        row["information"] = (
            "shared original information"
            if event < spec.reveal_event
            else f"{spec.name} observed from GW{spec.reveal_event}"
        )
        ledger.append(row)
    result = {
        "at": now(),
        "state_id": base["state_id"],
        "forecast_id": branch_forecast["id"],
        "market_id": market["id"],
        "start_event": base["start_event"],
        "end_event": base["end_event"],
        "ledger": ledger,
        "scenario": spec.model_dump(),
        "purchase_lot_assumptions": lot_assumptions,
        "validation": {"legal": True, "affordable": True, "shared_prefix": True},
        "policy": "fixed feasible prefix; receding-horizon transfer policy after observable information",
        "final_state": state.model_dump(),
    }
    for before, after in zip(base["ledger"], ledger):
        if before["event"] >= spec.reveal_event:
            break
        if action_from_row(before) != action_from_row(after):
            raise ValueError("Scenario policy violates information nonanticipativity")
    score_plan(store, result, branch_forecast)
    result["id"] = store.put("scenario_plan", result)
    return result


def evaluate_scenarios(store: Store, plan_id: str, specs: list[ScenarioSpec]) -> dict:
    if not specs:
        raise ValueError("At least one scenario is required")
    base = store.get("plan", plan_id)
    original = store.get("forecast", base["forecast_id"])
    if base["market_id"] != store.get("market")["id"]:
        raise ValueError("Scenario evaluation requires a plan computed for the current market version")
    probabilities = [s.probability for s in specs]
    if any(p is not None for p in probabilities) and (
        any(p is None for p in probabilities)
        or abs(sum(p for p in probabilities if p is not None) - 1) > 1e-6
    ):
        raise ValueError("Explicit scenario probabilities must cover all branches and sum to one")
    results = []
    controls = {}
    for spec in specs:
        if not base["start_event"] < spec.reveal_event <= base["end_event"]:
            raise ValueError("Future scenario must be observed after the immediate decision, within the plan")
        for change in spec.fixture_events:
            old = next(
                (
                    f
                    for f in store.get("market", base["market_id"])["fixtures"]
                    if f["id"] == change.fixture_id
                ),
                None,
            )
            if (
                old is None
                or old.get("finished")
                or (old.get("event") is not None and old["event"] < spec.reveal_event)
            ):
                raise ValueError("Cannot revise a fixture preceding scenario observation")
        for identity in spec.evidence_ids:
            store.get("claim", identity)
        branch = original
        if spec.players or spec.fixture_events:
            branch = forecast(
                store,
                start_event=original["start_event"],
                end_event=original["end_event"],
                worlds=original["worlds"],
                seed=original["seed"],
                scenario=spec.model_dump(exclude_none=True),
            )
        result = full_recourse(store, base, branch, spec)
        control_key = (spec.reveal_event, spec.sale_assumption)
        if control_key not in controls:
            controls[control_key] = full_recourse(
                store,
                base,
                original,
                ScenarioSpec(
                    name="Unmodified adaptive control",
                    reveal_event=spec.reveal_event,
                    sale_assumption=spec.sale_assumption,
                ),
            )
        control = controls[control_key]
        delta = score_plan(store, result, branch) - score_plan(store, control, original)
        error = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        results.append(
            {
                "name": spec.name,
                "probability": spec.probability,
                "plan_id": result["id"],
                "simulation": result["simulation"],
                "control_mean": control["simulation"]["mean"],
                "paired_scenario_delta": float(delta.mean()),
                "paired_mc_interval_95": [
                    float(delta.mean() - 1.96 * error),
                    float(delta.mean() + 1.96 * error),
                ],
                "reveal_event": spec.reveal_event,
                "validation": result["validation"],
                "price_lot_assumptions": result["purchase_lot_assumptions"],
            }
        )
    report = {
        "at": now(),
        "base_plan_id": plan_id,
        "branches": results,
        "probabilities_supplied": all(p is not None for p in probabilities),
        "expected_net_points": sum(
            cast(float, p) * r["simulation"]["mean"] for p, r in zip(probabilities, results)
        )
        if all(p is not None for p in probabilities)
        else None,
        "worst_branch_mean": min(r["simulation"]["mean"] for r in results),
        "method": "Observable branches with identical action prefix; no implicit probability or private purchase-price invention",
    }
    report["id"] = store.put("scenario_evaluation", report)
    atomic_json(store.root / "reports" / "scenario-evaluation.json", report)
    return report


def full_recourse(store: Store, base: dict, branch_forecast: dict, spec: ScenarioSpec, seconds=30):
    """Shared prefix, then full-horizon re-optimization at each new price observation."""
    import copy
    import shutil
    import tempfile
    from pathlib import Path

    from .ingestion import normalize
    from .optimization import optimize
    from .rules import quote
    from .storage import digest

    players, market = load_market(store, base["market_id"])
    original_bootstrap = store.latest("bootstrap", content_hash=market["source_hash"])[0]
    rules = Rules.from_bootstrap(original_bootstrap)
    state = TeamState.model_validate(
        {k: v for k, v in store.get("state", base["state_id"]).items() if k != "id"}
    )
    state, assumptions = conditional_lots(state, players, spec)
    current_players = dict(players)
    ledger = []
    recourse = []
    tail = {}
    source_forecast = branch_forecast
    for original in base["ledger"]:
        event = original["event"]
        changes = [c for c in spec.prices if c.event == event]
        for c in changes:
            if c.player_id not in current_players:
                raise ValueError("Unknown price-scenario player")
            current_players[c.player_id] = current_players[c.player_id].model_copy(update={"price": c.price})
        state = state.model_copy(
            update={
                "lots": [
                    lot.model_copy(
                        update={
                            "sell_price": quote(lot, current_players[lot.player_id].price),
                            "quote_market_price": current_players[lot.player_id].price,
                        }
                    )
                    for lot in state.lots
                ]
            }
        )
        if event < spec.reveal_event:
            action = action_from_row(original)
        else:
            if event == spec.reveal_event or changes:
                with tempfile.TemporaryDirectory(prefix="fie-recourse-") as temporary:
                    local = Store(temporary)
                    # Copy only the frozen world file; Windows does not require symlink privileges.
                    scenario_file = store.artifact_path(source_forecast["scenario_path"])
                    (Path(temporary) / "scenarios").mkdir()
                    shutil.copy2(scenario_file, Path(temporary) / "scenarios" / scenario_file.name)
                    bootstrap = copy.deepcopy(original_bootstrap)
                    for p in bootstrap["elements"]:
                        p["now_cost"] = current_players[p["id"]].price
                    fixtures = copy.deepcopy(market["fixtures"])
                    for change in spec.fixture_events:
                        for fixture in fixtures:
                            if fixture["id"] == change.fixture_id:
                                fixture["event"] = change.event
                    local.snapshot(
                        "bootstrap",
                        "scenario://conditional-bootstrap/" + digest(spec.model_dump()),
                        bootstrap,
                    )
                    local.snapshot(
                        "fixtures", "scenario://conditional-fixtures/" + digest(spec.model_dump()), fixtures
                    )
                    normalize(local, bootstrap, fixtures)
                    local_market = local.get("market")
                    active = store.get("decision_model")
                    active_id = local.put("decision_model", {k: v for k, v in active.items() if k != "id"})
                    if store.list("state_value_model", 1):
                        local.put(
                            "state_value_model",
                            {k: v for k, v in store.get("state_value_model").items() if k != "id"},
                        )
                    hypothetical = state.model_copy(update={"status": "dated_seed"})
                    identity = local.put("state", hypothetical.model_dump())
                    f = {k: v for k, v in source_forecast.items() if k != "id"}
                    f.update(market_id=local_market["id"], decision_model_id=active_id, rules_hash=rules.hash)
                    fid = local.put("forecast", f)
                    seed = {
                        **base,
                        "state_id": identity,
                        "forecast_id": fid,
                        "market_id": local_market["id"],
                        "start_event": event,
                        "ledger": [r for r in base["ledger"] if r["event"] >= event],
                    }
                    local.put("plan", {k: v for k, v in seed.items() if k != "id"})
                    solved = optimize(
                        local, identity, fid, base["end_event"], seconds, use_decision_critic=False
                    )
                    tail = {r["event"]: r for r in solved["ledger"]}
                    context_id = store.put(
                        "conditional_market",
                        {
                            "market": local_market,
                            "scenario": spec.model_dump(),
                            "event": event,
                            "source_market_id": market["id"],
                            "status": "Conditional hypothesis, never account truth",
                        },
                    )
                    recourse.append(
                        {"event": event, "conditional_market_id": context_id, "solver": solved["solver"]}
                    )
            action = action_from_row(tail[event])
        state, row = transition(state, action, current_players, rules)
        row["information"] = "Frozen common prefix" if event < spec.reveal_event else spec.name + " observed"
        ledger.append(row)
    result = {
        "at": now(),
        "state_id": base["state_id"],
        "forecast_id": branch_forecast["id"],
        "market_id": market["id"],
        "start_event": base["start_event"],
        "end_event": base["end_event"],
        "ledger": ledger,
        "scenario": spec.model_dump(),
        "purchase_lot_assumptions": assumptions,
        "recourse": recourse,
        "final_state": state.model_dump(),
        "validation": {
            "legal": True,
            "affordable": True,
            "shared_prefix": True,
            "full_market_recourse": True,
        },
        "policy": "Full remaining-horizon MIP after observation; price changes trigger exact repricing and renewed recourse",
        "limits": [
            "Finite MIP bounds remain visible; later unobserved prices remain conditional",
            "Purchase-lot hypotheses are declared rather than inferred as actual account history",
        ],
    }
    for before, after in zip(base["ledger"], ledger):
        if before["event"] < spec.reveal_event and action_from_row(before) != action_from_row(after):
            raise ValueError("Anticipatory scenario action rejected")
    score_plan(store, result, branch_forecast)
    result["id"] = store.put("scenario_plan", result)
    return result
