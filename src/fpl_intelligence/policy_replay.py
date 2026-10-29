"""Continuous evolving-account replay against observed historical outcomes."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import polars as pl

from .domain import Action, Lot, Player, TeamState
from .features import FEATURE_NAMES
from .history import archived_rows
from .optimization import greedy_bundle, select_squad
from .rules import Rules, best_lineup, score_gameweek, selling_price, transition, validate_squad
from .storage import Store, atomic_json, now


def continuous_replay(store: Store, season: str = "2025/26", start: int = 9, end: int = 38) -> dict:
    if not 2 <= start <= end <= 38:
        raise ValueError("Replay needs a historical burn-in and valid horizon")
    protocol: dict[str, Any] = {
        "version": "continuous-policy-v1",
        "season": season,
        "start": start,
        "end": end,
        "budget": 1000,
        "policies": ["hold-shrinkage", "hold-direct", "transfer-shrinkage", "transfer-direct"],
        "model_training": "strictly earlier deadline groups; fixed 100-tree direct model",
        "rules": "shared initial state, integer account ledgers, autosubs and captain fallback; no chips",
        "track": "historical_statistical_policy_replay",
        "price_provenance": "archived fixture value; no strict capture-time guarantee",
    }
    protocol_id = store.put("protocol", protocol)
    rows = [r for r in archived_rows(store) if r["season"] == season]
    if not rows:
        raise ValueError("Requested season archive is not imported")
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["round"])].append(row)
    if not all(event in groups for event in range(start, end + 1)):
        raise ValueError("Archive has incomplete event coverage")
    clubs = sorted({r["team"] for r in rows})
    club_ids = {name: i + 1 for i, name in enumerate(clubs)}
    features = store.get("features")
    frame = pl.read_parquet(store.artifact_path(features["path"]))
    states: dict[str, TeamState] = {}
    totals = {name: 0.0 for name in protocol["policies"]}
    ledgers: dict[str, list[dict[str, Any]]] = {name: [] for name in totals}
    known_players: dict[int, Player] = {}
    rejections = {name: 0 for name in totals}
    rules = Rules(season=season)
    for event in range(1, end + 1):
        current = {}
        for row in groups[event]:
            identity = row["element"]
            current[identity] = Player(
                id=identity,
                code=row.get("code") or identity,
                season=season,
                name=row["name"],
                web_name=row["name"],
                team=club_ids[row["team"]],
                position=row["position"],
                price=int(float(row["value"])),
                ownership=0,
                status="a",
                news="",
                raw={},
            )
        known_players = {
            p: v.model_copy(update={"selectable": p in current}) for p, v in known_players.items()
        }
        known_players.update(current)
        if event < start:
            continue
        group = f"{season}-GW{event:02d}"
        past = frame.filter(pl.col("deadline_group") < group)
        present = frame.filter(pl.col("deadline_group") == group)
        if past.is_empty() or present.is_empty():
            raise ValueError("Missing lagged replay features")
        model = lgb.LGBMRegressor(
            n_estimators=100,
            num_leaves=12,
            learning_rate=0.04,
            min_child_samples=45,
            reg_lambda=5,
            verbosity=-1,
            n_jobs=1,
            random_state=47,
        )
        model.fit(past.select(FEATURE_NAMES).to_numpy(), past["target"].to_numpy())
        direct = np.maximum(0, model.predict(present.select(FEATURE_NAMES).to_numpy()))
        baseline: dict[int, float] = defaultdict(float)
        learned: dict[int, float] = defaultdict(float)
        for row, prediction in zip(present.to_dicts(), direct):
            baseline[row["player_id"]] += row["baseline"]
            learned[row["player_id"]] += float(prediction)
        if not states:
            chosen = select_squad(None, known_players, rules, baseline, seconds=5, initial_budget=1000)
            shared = TeamState(
                season=season,
                event=event,
                bank=1000 - sum(known_players[p].price for p in chosen),
                free_transfers=1,
                lots=[
                    Lot(
                        player_id=p,
                        purchase_price=known_players[p].price,
                        sell_price=known_players[p].price,
                        quote_market_price=known_players[p].price,
                        provenance="historical_shared_start",
                    )
                    for p in chosen
                ],
                known_at=now(),
            )
            validate_squad(shared, known_players, rules)
            states = {name: shared.model_copy(deep=True) for name in totals}
        actual: dict[int, int] = defaultdict(int)
        minutes: dict[int, int] = defaultdict(int)
        for row in groups[event]:
            actual[row["element"]] += int(row["total_points"])
            minutes[row["element"]] += int(row["minutes"])
        for name, state in states.items():
            state = state.model_copy(
                update={
                    "lots": [
                        lot.model_copy(
                            update={
                                "sell_price": selling_price(
                                    cast(int, lot.purchase_price), known_players[lot.player_id].price
                                ),
                                "quote_market_price": known_players[lot.player_id].price,
                            }
                        )
                        for lot in state.lots
                    ]
                }
            )
            counts = Counter(known_players[lot.player_id].team for lot in state.lots)
            state.grandfathered_clubs = [club for club, count in counts.items() if count > 3]
            means = learned if name.endswith("direct") else baseline
            action = (
                greedy_bundle(state, known_players, rules, means, 2)
                if name.startswith("transfer")
                else Action()
            )
            squad = sorted({lot.player_id for lot in state.lots} - set(action.sell) | set(action.buy))
            action.lineup = best_lineup(squad, means, known_players, rules)
            try:
                following, ledger = transition(state, action, known_players, rules)
            except ValueError:
                rejections[name] += 1
                action = Action(
                    lineup=best_lineup([lot.player_id for lot in state.lots], means, known_players, rules)
                )
                following, ledger = transition(state, action, known_players, rules)
            assert action.lineup is not None
            realized = score_gameweek(actual, minutes, action.lineup, known_players, rules) - ledger["hit"]
            totals[name] += realized
            states[name] = following
            ledger.update(realized_net_points=realized, cumulative_points=totals[name])
            ledgers[name].append(ledger)
    result = {
        "at": now(),
        "protocol_id": protocol_id,
        "season": season,
        "events": end - start + 1,
        "track": protocol["track"],
        "shared_initial_state": True,
        "invalid_exported_actions": 0,
        "policies": {
            name: {
                "realized_net_points": totals[name],
                "hits": sum(r["hit"] for r in ledgers[name]),
                "transfers": sum(len(r["buy"]) for r in ledgers[name]),
                "candidate_rejections": rejections[name],
                "final_bank": states[name].bank,
                "ledger": ledgers[name],
            }
            for name in totals
        },
        "limitations": [
            "Historical statistical replay; archival prices are not strict pre-deadline snapshots",
            "These four policies exclude chips to isolate forecast and transfer effects",
            "No overall-rank or competitor superiority claim follows from this replay",
        ],
    }
    result["id"] = store.put("policy_evaluation", result)
    atomic_json(
        Path("artifacts/evaluations/continuous_policy.json"),
        {
            **result,
            "policies": {
                k: {f: v for f, v in p.items() if f != "ledger"} for k, p in result["policies"].items()
            },
        },
    )
    return result
