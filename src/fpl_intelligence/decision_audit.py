"""Evaluate choices after forecasting; hindsight is confined to diagnostic oracles."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from .domain import Action, Lot, Player, TeamState
from .optimization import greedy_bundle, select_squad
from .rules import Rules, best_lineup, score_gameweek, selling_price, transition
from .storage import Store, atomic_json, now


def event_market(rows, season):
    clubs = {name: i + 1 for i, name in enumerate(sorted({r["team"] for r in rows}))}
    players = {}
    actual: dict[int, int] = defaultdict(int)
    minutes: dict[int, int] = defaultdict(int)
    means: dict[str, dict[int, float]] = {
        "learned": defaultdict(float),
        "shrinkage": defaultdict(float),
        "neural": defaultdict(float),
    }
    for row in rows:
        identity = row["player_id"]
        players[identity] = Player(
            id=identity,
            code=identity,
            season=season,
            name=str(identity),
            web_name=str(identity),
            team=clubs[row["team"]],
            position=cast(Any, int(row["position"])),
            price=int(round(row["price_lag"])),
            ownership=0,
            status="a",
            news="",
            raw={},
        )
        actual[identity] += int(row["actual"])
        minutes[identity] += int(row["minutes"])
        means["learned"][identity] += row["predicted"]
        means["shrinkage"][identity] += row["baseline"]
        means["neural"][identity] += row["network"] if row.get("network") is not None else row["predicted"]
    return players, dict(actual), dict(minutes), means


def evaluate_choices(store: Store, seconds=1.0):
    spec = store.get("decision_model")
    groups = defaultdict(list)
    for row in spec["predictions"]:
        groups[row["deadline_group"]].append(row)
    report = {
        "at": now(),
        "decision_model_id": spec["id"],
        "track": "out-of-fold historical decisions",
        "oracle_use": "Diagnostic upper bound only; no hindsight target is added to model features or live fitting",
        "budget": 1000,
        "weeks": [],
        "captains": [],
        "price_cohorts": {},
        "limits": [
            "Archive registrations and prices are retrospective; not a strict historical account reconstruction",
            "Perfect-hindsight teams intentionally overfit realized match variance and are not attainable forecasts",
        ],
    }
    for group, rows in sorted(groups.items()):
        season = rows[0]["season"]
        players, actual, minutes, means = event_market(rows, season)
        # Consistent club IDs within this event; single-week budget comparison.
        rules = Rules(season=season)
        eligible = {
            p: player
            for p, player in players.items()
            if any(r["player_id"] == p and r["price_observed"] for r in rows)
        }
        candidates = sorted(eligible, key=lambda p: (-means["learned"][p], p))
        if len(candidates) < 15:
            continue
        top2 = candidates[:2]
        premium = sorted(eligible, key=lambda p: (-eligible[p].price, p))[:2]
        shrunk = sorted(eligible, key=lambda p: (-means["shrinkage"][p], p))[:2]

        def bonus(selection):
            return next((actual[p] for p in selection if minutes[p] > 0), 0)

        report["captains"].append(
            {
                "deadline_group": group,
                "learned_player": top2[0],
                "learned_position": eligible[top2[0]].position,
                "learned_bonus": bonus(top2),
                "price_baseline_bonus": bonus(premium),
                "shrinkage_bonus": bonus(shrunk),
                "hindsight_best": max(actual[p] for p in eligible),
            }
        )
        # Burn in nine weeks before a budgeted team audit; the captain diagnostic
        # above also covers opening weeks with known returning-player prices.
        if rows[0]["event"] < 9:
            continue
        result = {"deadline_group": group}
        for name, target in [
            ("learned", means["learned"]),
            ("shrinkage", means["shrinkage"]),
            ("oracle", actual),
        ]:
            diagnostics: dict[str, Any] = {}
            chosen = select_squad(
                None,
                eligible,
                rules,
                target,
                seconds=max(seconds, 3) if name == "oracle" else seconds,
                initial_budget=1000,
                diagnostics=diagnostics,
                bench_weight=0,
                relative_gap=0,
            )
            lineup = best_lineup(chosen, target, eligible, rules)
            result[name] = {
                "realized_points": score_gameweek(actual, minutes, lineup, eligible, rules),
                "cost": sum(eligible[p].price for p in chosen),
                "solver": diagnostics,
                "squad": chosen,
                "lineup": lineup.model_dump(),
            }
        result["hindsight_regret"] = (
            result["oracle"]["realized_points"] - result["learned"]["realized_points"]
        )
        report["weeks"].append(result)
    captains = report["captains"]
    report["captain_totals"] = {
        k: sum(r[k] for r in captains)
        for k in ["learned_bonus", "price_baseline_bonus", "shrinkage_bonus", "hindsight_best"]
    }
    report["captain_totals"]["defender_captain_weeks"] = sum(r["learned_position"] == 2 for r in captains)
    report["team_totals"] = {
        k: sum(r[k]["realized_points"] for r in report["weeks"]) for k in ("learned", "shrinkage", "oracle")
    }
    for label, low, high in [("budget", 0, 60), ("midprice", 60, 90), ("premium", 90, 1000)]:
        cohort = [r for r in spec["predictions"] if low <= r["price_lag"] < high and r["prior_start"] >= 0.6]
        report["price_cohorts"][label] = {
            "n": len(cohort),
            "predicted": float(np.mean([r["predicted"] for r in cohort])),
            "actual": float(np.mean([r["actual"] for r in cohort])),
            "bias": float(np.mean([r["predicted"] - r["actual"] for r in cohort])),
        }
    report["id"] = store.put("decision_audit", report)
    atomic_json(
        Path("artifacts/evaluations/decision_audit.json"),
        {k: v for k, v in report.items() if k not in ("weeks", "captains")},
    )
    return report


def replay_policy(
    store: Store,
    season="2025/26",
    start=9,
    end=38,
    spec=None,
    artifact_kind="decision_policy",
    reference_transfers=False,
):
    spec = spec or store.get("decision_model")
    groups = defaultdict(list)
    for row in spec["predictions"]:
        if row["season"] == season:
            groups[row["event"]].append(row)
    # Stable club IDs for the entire evolving account.
    club_ids = {
        name: i + 1 for i, name in enumerate(sorted({r["team"] for rows in groups.values() for r in rows}))
    }
    states: dict[str, TeamState] = {}
    known: dict[int, Player] = {}
    ledgers: dict[str, list[dict]] = {
        name: [] for name in ("transfer-learned", "transfer-shrinkage", "hold-learned", "hold-shrinkage")
    }
    rules = Rules(season=season)
    for event in range(1, end + 1):
        rows = groups[event]
        current, actual, minutes, means = event_market(rows, season)
        for row in rows:
            current[row["player_id"]].team = club_ids[row["team"]]
        known = {p: player.model_copy(update={"selectable": p in current}) for p, player in known.items()}
        known.update(current)
        if event < start:
            continue
        if not states:
            chosen = select_squad(None, known, rules, means["shrinkage"], seconds=3, initial_budget=1000)
            shared = TeamState(
                season=season,
                event=event,
                bank=1000 - sum(known[p].price for p in chosen),
                free_transfers=1,
                lots=[
                    Lot(
                        player_id=p,
                        purchase_price=known[p].price,
                        sell_price=known[p].price,
                        quote_market_price=known[p].price,
                        provenance="historical matched start",
                    )
                    for p in chosen
                ],
                known_at=now(),
            )
            states = {name: shared.model_copy(deep=True) for name in ledgers}
        for name, state in states.items():
            state = state.model_copy(
                update={
                    "lots": [
                        lot.model_copy(
                            update={
                                "sell_price": selling_price(
                                    cast(int, lot.purchase_price), known[lot.player_id].price
                                ),
                                "quote_market_price": known[lot.player_id].price,
                            }
                        )
                        for lot in state.lots
                    ]
                }
            )
            counts = Counter(known[lot.player_id].team for lot in state.lots)
            state.grandfathered_clubs = [club for club, count in counts.items() if count > 3]
            forecast = means["learned" if name.endswith("learned") else "shrinkage"]
            transfer_target = means["shrinkage"] if reference_transfers else forecast
            action = (
                greedy_bundle(state, known, rules, transfer_target, 2)
                if name.startswith("transfer")
                else Action()
            )
            squad = sorted({lot.player_id for lot in state.lots} - set(action.sell) | set(action.buy))
            action.lineup = best_lineup(squad, forecast, known, rules)
            following, row = transition(state, action, known, rules)
            row["realized_net_points"] = (
                score_gameweek(actual, minutes, action.lineup, known, rules) - row["hit"]
            )
            states[name] = following
            ledgers[name].append(row)
    report = {
        "at": now(),
        "decision_model_id": spec["id"],
        "season": season,
        "events": end - start + 1,
        "shared_initial_state": True,
        "invalid_exported_actions": 0,
        "track": "out-of-fold evolving-account replay; no test outcome used in decisions",
        "policies": {
            name: {
                "realized_net_points": sum(r["realized_net_points"] for r in rows),
                "hits": sum(r["hit"] for r in rows),
                "transfers": sum(len(r["buy"]) for r in rows),
                "final_bank": states[name].bank,
                "ledger": rows,
            }
            for name, rows in ledgers.items()
        },
        "limits": [
            "No chips, to isolate forecast/transfer effects",
            "Historical archived quotes are not guaranteed captured before deadline",
            "No guarantee of rank or future returns",
        ],
    }
    report["id"] = store.put(artifact_kind, report)
    atomic_json(
        Path("artifacts/evaluations/" + artifact_kind + ".json"),
        {
            **report,
            "policies": {
                name: {k: v for k, v in row.items() if k != "ledger"}
                for name, row in report["policies"].items()
            },
        },
    )
    return report
