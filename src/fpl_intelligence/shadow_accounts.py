"""Synchronized prospective, counterfactual accounts with known purchase lots.

These are explicitly hypothetical equal-£100m accounts, not private account
reconstructions. Frozen finances advance after the deadline; scores wait for
official finalization. A
precommitted hold instruction handles an installation missing a deadline.
"""

from __future__ import annotations

import json
from collections import defaultdict

from .availability import moment
from .decision import canonical_rows
from .decision_training import reprice
from .domain import Action, Lineup, Lot, TeamState
from .ingestion import load_market
from .optimization import optimize, select_squad
from .pipeline_jobs import replay_lock
from .replay_benchmarks import baseline_signals
from .rules import Rules, best_lineup, score_gameweek, transition
from .storage import Store, digest, now
from .strategy import solve_action


class ShadowStore(Store):
    """Keep hypothetical selection receipts separate from the real account."""

    def put(self, kind, body):
        return super().put("shadow_policy_selection" if kind == "policy_selection" else kind, body)


def save_capture(store, cohort, registry):
    # A crash cannot leave a first cohort without its matching account registry.
    with store.db() as db:
        for kind, body in (("shadow_cohort", cohort), ("shadow_registry", registry)):
            db.execute(
                "INSERT OR IGNORE INTO objects VALUES(?,?,?,?)",
                (kind, digest(body), now(), json.dumps(body, allow_nan=False)),
            )
    return digest(cohort)


def latest_cohorts(store, registry):
    result: dict = {}
    for item in store.list("shadow_cohort", 4000):
        record = store.get("shadow_cohort", item["id"])
        if record["registry_key"] == registry["registry_key"]:
            result.setdefault(record["event"], record)
    return result


def actual_events(store, season):
    rows, _, _ = canonical_rows(store)
    points: dict = defaultdict(lambda: defaultdict(float))
    minutes: dict = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row["season"] == season:
            event, player = int(row["round"]), int(row["element"])
            points[event][player] += float(row.get("total_points") or 0)
            minutes[event][player] += int(row.get("minutes") or 0)
    return points, minutes


def _advance_shadow_accounts(store: Store):
    if not store.list("shadow_registry", 1):
        return {"status": "No prospective account cohort yet"}
    registry = store.get("shadow_registry")
    players, market = load_market(store)
    if registry["season"] != market["season"]:
        return {"status": "Prior-season cohort retained; new season needs a new seed"}
    finalized = {e["id"] for e in market["events"] if e.get("data_checked")}
    next_event = next((e["id"] for e in market["events"] if e.get("is_next")), 39)
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    points, minutes = actual_events(store, market["season"])
    cohorts = latest_cohorts(store, registry)
    accounts = {name: TeamState.model_validate(body) for name, body in registry["accounts"].items()}
    event = min(s.event for s in accounts.values())
    advanced = []
    while event < next_event and event > registry.get(
        "transitioned_through", registry.get("finalized_through", 0)
    ):
        cohort = cohorts.get(event)
        if cohort is None:
            previous = max((e for e in cohorts if e < event), default=None)
            if previous is None:
                break
            prior = cohorts[previous]
            actions = {}
            for name, state in accounts.items():
                standing = Lineup.model_validate(prior["actions"][name]["standing_lineup"])
                action = Action(lineup=standing)
                actions[name] = {
                    "state_before": state.model_dump(),
                    "action": action.model_dump(),
                    "standing_lineup": standing.model_dump(),
                }
            cohort = {
                "registry_key": registry["registry_key"],
                "season": market["season"],
                "event": event,
                "recorded_at": now(),
                "frozen_at": prior["frozen_at"],
                "market_id": prior["market_id"],
                "forecast_id": prior["forecast_id"],
                "actions": actions,
                "track": "Previously frozen standing hold instruction; no new prediction issued",
                "standing_source_id": prior["id"],
            }
            cohort["id"] = store.put("shadow_cohort", cohort)
            cohorts[event] = cohort
        frozen_players, _ = load_market(store, cohort["market_id"])
        for name, row in cohort["actions"].items():
            before = TeamState.model_validate(row["state_before"])
            action = Action.model_validate(row["action"])
            following, _ = transition(before, action, frozen_players, rules)
            accounts[name] = following
        advanced.append(event)
        if event == 38:
            break
        event += 1
    # Recompute scores for every finalized cohort, including official revisions.
    outcomes = []
    totals = {name: 0.0 for name in accounts}
    for event, cohort in sorted(cohorts.items()):
        if event not in finalized:
            continue
        if event not in points:
            raise ValueError(
                f"Finalized GW{event} history is missing; refresh player histories before scoring"
            )
        deadline = next(e["deadline_time"] for e in market["events"] if e["id"] == event)
        if moment(cohort["frozen_at"]) >= moment(deadline):
            raise ValueError("A post-deadline cohort cannot be counted as prospective")
        frozen_players, _ = load_market(store, cohort["market_id"])
        for name, row in cohort["actions"].items():
            before = TeamState.model_validate(row["state_before"])
            action = Action.model_validate(row["action"])
            assert action.lineup is not None
            _, ledger = transition(before, action, frozen_players, rules)
            result = (
                score_gameweek(
                    points[event], minutes[event], action.lineup, frozen_players, rules, action.chip
                )
                - ledger["hit"]
            )
            totals[name] += result
            outcomes.append(
                {
                    "event": event,
                    "policy": name,
                    "net_points": result,
                    "hit": ledger["hit"],
                    "chip": action.chip,
                    "cohort_id": cohort["id"],
                    "track": cohort["track"],
                }
            )
    for name, state in accounts.items():
        state.points = int(totals[name])
    body = {k: v for k, v in registry.items() if k != "id"}
    body.update(
        accounts={name: s.model_dump() for name, s in accounts.items()},
        transitioned_through=max(
            [registry.get("transitioned_through", registry.get("finalized_through", 0)), *advanced]
        ),
        updated_at=now(),
    )
    body["id"] = store.put("shadow_registry", body)
    report = {
        "at": now(),
        "registry_key": registry["registry_key"],
        "advanced_events": advanced,
        "outcomes": outcomes,
        "total_net_points": totals,
        "finalized_deadlines": len({r["event"] for r in outcomes}),
        "pending_deadlines": sorted(set(cohorts) - finalized),
        "scope": "Prospective continuous hypothetical accounts, equal initial £100m purchase lots, actual final points and legal FT/chip/bank transitions; no submitted account changes",
    }
    report["id"] = store.put("shadow_evaluation", report)
    return report


def advance_shadow_accounts(store: Store):
    with replay_lock(store.root / "prospective-shadow.lock"):
        return _advance_shadow_accounts(store)


def propose_accounts(store, states, forecast, players, rules, seconds):
    from .objectives import choose_recommendation

    store = ShadowStore(store.root)
    event = forecast["start_event"]
    end = 19 if event <= 19 else 38
    scheduled = {
        r["player_id"] for r in forecast["players"] if r["event"] == event and r["fixture_count"] > 0
    }
    choices = {}
    primary = states["primary_multiweek"]
    state_id = store.put("planning_state", primary.model_dump())
    plan = optimize(store, state_id, forecast["id"], end, seconds, use_decision_critic=False)
    plan, _ = choose_recommendation(store, [plan], forecast)
    row = plan["ledger"][0]
    choices["primary_multiweek"] = Action(
        sell=[p["player_id"] for p in row["sell"]],
        buy=[p["player_id"] for p in row["buy"]],
        chip=row["chip"],
        lineup=Lineup.model_validate(row["lineup"]),
    )
    for name, baseline in (
        ("ownership_no_hits", "ownership"),
        ("official_expected_no_hits", "official_expected"),
    ):
        state = states[name]
        signal = baseline_signals(players, baseline, scheduled)
        choices[name] = solve_action(
            state,
            players,
            rules,
            signal,
            max_transfers=max(0, state.free_transfers - state.transfers_made),
            seconds=min(seconds, 10),
        ).action
    return choices


def freeze_shadow_accounts(store: Store, seconds=60):
    """Freeze synchronized policies without moving the real account or plan."""
    with replay_lock(store.root / "prospective-shadow.lock"):
        _advance_shadow_accounts(store)
        players, market = load_market(store)
        forecast = store.get("forecast")
        model = store.get("decision_model")
        event = next((e["id"] for e in market["events"] if e.get("is_next")), None)
        if event is None:
            return {"status": "Season complete"}
        deadline = next(e["deadline_time"] for e in market["events"] if e["id"] == event)
        if moment(now()) >= moment(deadline):
            raise ValueError("Cannot freeze prospective actions after their deadline")
        if (
            forecast["market_id"] != market["id"]
            or forecast["start_event"] != event
            or forecast["decision_model_id"] != model["id"]
        ):
            raise ValueError("Refresh the current validated forecast before freezing shadow accounts")
        try:
            registry = store.get("shadow_registry")
        except ValueError:
            registry = None
        if registry is None or registry["season"] != market["season"]:
            scheduled = {
                r["player_id"] for r in forecast["players"] if r["event"] == event and r["fixture_count"] > 0
            }
            seed = select_squad(
                None,
                players,
                Rules.from_bootstrap(store.latest("bootstrap")[0]),
                baseline_signals(players, "ownership", scheduled),
                initial_budget=1000,
                seconds=min(seconds, 10),
                bench_weight=0,
                relative_gap=0,
            )
            state = TeamState(
                season=market["season"],
                event=event,
                bank=1000 - sum(players[p].price for p in seed),
                free_transfers=1,
                status="dated_seed",
                known_at=now(),
                lots=[
                    Lot(
                        player_id=p,
                        purchase_price=players[p].price,
                        sell_price=players[p].price,
                        quote_market_price=players[p].price,
                    )
                    for p in seed
                ],
            )
            registry = {
                "season": market["season"],
                "seed_event": event,
                "created_at": now(),
                "registry_key": digest({"market": market["id"], "seed": state.model_dump()}),
                "accounts": {
                    name: state.model_dump()
                    for name in ("primary_multiweek", "ownership_no_hits", "official_expected_no_hits")
                },
                "seed_scope": "Hypothetical equal £100m midseason accounts with exact purchase lots and one FT; not the user's private account",
                "missed_capture_policy": "Hold permanent roster with the last precommitted standing lineup, without transfers or chips",
            }
        cohorts = latest_cohorts(store, registry)
        if event in cohorts and cohorts[event].get("forecast_id") == forecast["id"]:
            return cohorts[event]
        states = {name: TeamState.model_validate(body) for name, body in registry["accounts"].items()}
        if any(state.event != event for state in states.values()):
            raise ValueError("Refresh released deadlines before advancing prospective accounts")
        rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
        for name, state in states.items():
            states[name] = reprice(state, players, rules)
            states[name].known_at = now()
        before = {
            k: store.get(k)["id"] for k in ("state", "selected_plan", "market", "forecast", "decision_model")
        }
        choices = propose_accounts(store, states, forecast, players, rules, seconds)
        frozen_at = now()
        if moment(frozen_at) >= moment(deadline):
            raise ValueError("The deadline passed during calculation; no late shadow decision was recorded")
        if store.get("forecast")["id"] != forecast["id"]:
            raise ValueError("Forecast changed during shadow calculation; retry with the new information set")
        actions = {}
        future = {r["player_id"]: r["mean"] for r in forecast["players"] if r["event"] == min(38, event + 1)}
        for name, action in choices.items():
            following, ledger = transition(states[name], action, players, rules)
            permanent = [lot.player_id for lot in following.lots]
            standing = best_lineup(permanent, future, players, rules)
            actions[name] = {
                "state_before": states[name].model_dump(),
                "action": action.model_dump(),
                "ledger": ledger,
                "standing_lineup": standing.model_dump(),
            }
        cohort = {
            "registry_key": registry["registry_key"],
            "season": market["season"],
            "event": event,
            "deadline": deadline,
            "recorded_at": frozen_at,
            "frozen_at": frozen_at,
            "market_id": market["id"],
            "forecast_id": forecast["id"],
            "model_id": model["id"],
            "track": "Synchronized prospective continuous shadow accounts",
            "actions": actions,
            "real_account_unchanged": all(store.get(k)["id"] == v for k, v in before.items()),
            "seed_scope": registry["seed_scope"],
        }
        if not cohort["real_account_unchanged"]:
            raise ValueError("Inputs changed while freezing shadow accounts; no cohort recorded")
        registry = {k: v for k, v in registry.items() if k != "id"}
        registry["accounts"] = {name: state.model_dump() for name, state in states.items()}
        cohort["id"] = save_capture(store, cohort, registry)
        return cohort
