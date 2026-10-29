"""Counterfactual transfer-value learning on independent legal account states."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import polars as pl

from .context_archive import load_context
from .decision import canonical_rows
from .domain import Action, Lot, Player, TeamState
from .optimization import greedy_bundle
from .rules import Rules, best_lineup, score_gameweek, transition, validate_squad
from .storage import Store, atomic_json, digest, now

NAMES = (
    "nextweek_gain",
    "reference_gain",
    "fourweek_gain",
    "hit",
    "transfers",
    "free_transfers",
    "bank_before",
    "bank_after",
    "mean_in_price",
    "mean_out_price",
    "forecast_disagreement",
)
PARAMETERS: dict[str, Any] = {
    "n_estimators": 160,
    "num_leaves": 10,
    "min_child_samples": 100,
    "learning_rate": 0.03,
    "reg_lambda": 30,
    "random_state": 83,
    "n_jobs": 1,
    "verbosity": -1,
}


def random_state(players, means, rules, event, season, rng):
    pools = {
        pos: [p for p in players if players[p].position == pos and players[p].selectable]
        for pos in range(1, 5)
    }
    for _ in range(5000):
        owned: list[int] = []
        for pos, count in rules.composition.items():
            pool = pools[pos]
            weights = np.array([max(0.1, means.get(p, 0)) + 0.3 for p in pool]) ** 1.5
            weights /= weights.sum()
            owned.extend(int(p) for p in rng.choice(pool, count, replace=False, p=weights))
        cost = sum(players[p].price for p in owned)
        if (
            not 950 <= cost <= 1000
            or max(Counter(players[p].team for p in owned).values()) > rules.club_limit
        ):
            continue
        state = TeamState(
            season=season,
            event=event,
            bank=1000 - cost,
            free_transfers=int(rng.integers(1, 6)),
            lots=[
                Lot(
                    player_id=p,
                    purchase_price=players[p].price,
                    sell_price=players[p].price,
                    quote_market_price=players[p].price,
                    provenance="Counterfactual account at observed pre-deadline quote; no accrued profit",
                )
                for p in owned
            ],
            known_at=now(),
        )
        validate_squad(state, players, rules)
        return state
    raise ValueError("Random legal-team sampler exhausted its budget")


def expected(lineup, means):
    return sum(means.get(p, 0) for p in lineup.xi) + means.get(lineup.captain, 0)


def action_features(state, action, players, rules, short, reference, long):
    owned = {r.player_id for r in state.lots}
    following = (owned - set(action.sell)) | set(action.buy)
    original = best_lineup(sorted(owned), short, players, rules)
    nextline = best_lineup(sorted(following), short, players, rules)
    oldref = best_lineup(sorted(owned), reference, players, rules)
    newref = best_lineup(sorted(following), reference, players, rules)
    oldlong = best_lineup(sorted(owned), long, players, rules)
    newlong = best_lineup(sorted(following), long, players, rules)
    priced = sum(players[p].price for p in action.buy) - sum(players[p].price for p in action.sell)
    hit = rules.hit * max(0, len(action.buy) - state.free_transfers)
    short_gain = expected(nextline, short) - expected(original, short)
    reference_gain = expected(newref, reference) - expected(oldref, reference)
    long_gain = expected(newlong, long) - expected(oldlong, long)
    return [
        short_gain,
        reference_gain,
        long_gain,
        hit,
        len(action.buy),
        state.free_transfers,
        state.bank / 10,
        (state.bank - priced) / 10,
        np.mean([players[p].price for p in action.buy]) / 10 if action.buy else 0,
        np.mean([players[p].price for p in action.sell]) / 10 if action.sell else 0,
        abs(short_gain - reference_gain),
    ]


def build_policy_data(store: Store, states_per_week=24):
    horizon = store.get("horizon_model")
    rows = pl.read_parquet(store.artifact_path(horizon["prediction_path"])).to_dicts()
    groups = defaultdict(list)
    for row in rows:
        groups[row["deadline_group"]].append(row)
    context = load_context(store)
    registry: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for (season, event, _), packet in context.items():
        registry[season, event][packet["id"]] = packet
    raw, _, _ = canonical_rows(store)
    actual: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    minutes: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in raw:
        key = (row["season"], int(row["round"]))
        actual[key][int(row["element"])] += int(row.get("total_points") or 0)
        minutes[key][int(row["element"])] += int(row.get("minutes") or 0)
    del raw, context
    cache = store.root / "policy-checkpoints" / (horizon["id"][:20] + "-" + str(states_per_week))
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    states = 0
    oracle = []
    for group, current in sorted(groups.items()):
        season = current[0]["season"]
        event = current[0]["event"]
        packets = registry[season, event]
        if event == 1:
            continue
        checkpoint = cache / (group.replace("/", "-") + ".parquet")
        if checkpoint.exists():
            completed = pl.read_parquet(checkpoint).to_dicts()
            records.extend(completed)
            states += states_per_week
            print("Restored completed counterfactual deadline", group, flush=True)
            continue
        rng = np.random.default_rng([83, int(digest(group)[:8], 16)])
        week_records = []
        # Restrict this controlled experiment to observed current-fixture rows;
        # blank-week transfer policies remain covered by the main planner.
        ids = {r["player_id"] for r in current}
        players = {
            p: Player(
                id=p,
                code=int(packets[p]["code"]),
                season=season,
                name=str(p),
                web_name=str(p),
                team=int(packets[p]["team"]),
                position=cast(Any, int(packets[p]["element_type"])),
                price=int(packets[p]["now_cost"]),
                ownership=0,
                status=packets[p].get("status", "a"),
                news="",
                selectable=packets[p].get("status") != "u",
                raw={},
            )
            for p in ids
            if p in packets
        }
        short = {r["player_id"]: r["nextweek_tree"] for r in current}
        reference = {r["player_id"]: r["nextweek_reference"] for r in current}
        long = {r["player_id"]: r[horizon["selected"]] for r in current}
        rules = Rules(season=season)
        for sample in range(states_per_week):
            state = random_state(players, reference, rules, event, season, rng)
            owned = sorted(lot.player_id for lot in state.lots)
            options = [("roll", Action())]
            for source, means in [("reference", reference), ("short", short), ("long", long)]:
                for maximum in (1, 2):
                    options.append(
                        (source + str(maximum), greedy_bundle(state, players, rules, means, maximum))
                    )
            unique: dict[tuple, tuple[str, Action]] = {}
            for label, action in options:
                unique.setdefault((tuple(sorted(action.sell)), tuple(sorted(action.buy))), (label, action))
            base_line = best_lineup(owned, short, players, rules)
            base = sum(
                score_gameweek(actual[season, g], minutes[season, g], base_line, players, rules)
                for g in range(event, event + 4)
            )
            state_records = []
            for label, action in unique.values():
                squad = sorted(set(owned) - set(action.sell) | set(action.buy))
                action.lineup = best_lineup(squad, short, players, rules)
                _, ledger = transition(state, action, players, rules)
                value = (
                    sum(
                        score_gameweek(actual[season, g], minutes[season, g], action.lineup, players, rules)
                        for g in range(event, event + 4)
                    )
                    - ledger["hit"]
                )
                # Also score the reference's own lineup on the identical account.
                refline = best_lineup(squad, reference, players, rules)
                refvalue = (
                    sum(
                        score_gameweek(actual[season, g], minutes[season, g], refline, players, rules)
                        for g in range(event, event + 4)
                    )
                    - ledger["hit"]
                )
                state_records.append(
                    {
                        "deadline_group": group,
                        "label_end": f"{season}-GW{event + 3:02d}",
                        "state_id": f"{group}-{sample}",
                        "x": action_features(state, action, players, rules, short, reference, long),
                        "actual_gain": value - base,
                        "actual_score": value,
                        "reference_lineup_score": refvalue,
                        "action_source": label,
                        "sell": action.sell,
                        "buy": action.buy,
                        "hit": ledger["hit"],
                    }
                )
            reference_action = next(a for name, a in options if name == "reference2")
            refkey = (tuple(sorted(reference_action.sell)), tuple(sorted(reference_action.buy)))
            reference_row = next(r for key, r in zip(unique, state_records) if key == refkey)
            for record in state_records:
                record["reference_score"] = reference_row["reference_lineup_score"]
            records.extend(state_records)
            week_records.extend(state_records)
            states += 1
            oracle.append(
                {
                    "state_id": state_records[0]["state_id"],
                    "best_searched_score": max(r["actual_score"] for r in state_records),
                }
            )
        temporary = checkpoint.with_suffix(".partial")
        pl.DataFrame(week_records, infer_schema_length=None).write_parquet(temporary)
        temporary.replace(checkpoint)
        print("Counterfactual legal accounts generated", group, states, flush=True)
    path = (
        store.root
        / "features"
        / ("policy-" + digest({"horizon": horizon["id"], "states": states_per_week})[:20] + ".parquet")
    )
    pl.DataFrame(records, infer_schema_length=None).write_parquet(path)
    result = {
        "at": now(),
        "horizon_model_id": horizon["id"],
        "states_per_deadline": states_per_week,
        "states": states,
        "action_rows": len(records),
        "invalid_actions": 0,
        "path": str(path),
        "hash": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "features": NAMES,
        "hindsight_use": "Best observed action among the generated legal candidates is diagnostic only; it is not claimed to be the global optimal transfer",
        "limits": [
            "Random squads are plausible synthetic states, not sampled private accounts",
            "Four-week fixed-lineup hold isolates transfer value; it does not model all subsequent actions or chips",
            "No accrued price profits; starting quotes are genuine pre-deadline registered prices",
            "Controlled experiment excludes blank-week player rows without current fixture features",
        ],
    }
    result["id"] = store.put("policy_features", result)
    return result


def train_policy(store: Store, states_per_week=24):
    dataset = build_policy_data(store, states_per_week)
    rows = pl.read_parquet(store.artifact_path(dataset["path"])).to_dicts()
    train = [r for r in rows if r["label_end"] < "2025/26-GW20"]
    cal = [r for r in rows if "2025/26-GW20" <= r["deadline_group"] <= "2025/26-GW24"]
    test = [r for r in rows if r["deadline_group"] >= "2025/26-GW28"]

    def fit(records):
        x = np.array([r["x"] for r in records])
        y = np.array([r["actual_gain"] - (r["x"][2] - r["x"][3]) for r in records])
        return lgb.LGBMRegressor(**PARAMETERS).fit(x, y)

    def evaluate(model, records, weight):
        correction = model.predict(np.array([r["x"] for r in records]))
        groups = defaultdict(list)
        for i, row in enumerate(records):
            groups[row["state_id"]].append((row, float(row["x"][2] - row["x"][3] + weight * correction[i])))
        decisions = []
        for state, choices in groups.items():
            chosen = max(choices, key=lambda v: (v[1], -v[0]["hit"], -len(v[0]["buy"])))[0]
            decisions.append(
                {
                    "state_id": state,
                    "deadline_group": chosen["deadline_group"],
                    "actual": 0,
                    "selected": chosen["actual_score"],
                    "baseline": chosen["reference_score"],
                    "gain": chosen["actual_score"] - chosen["reference_score"],
                    "best_searched": max(r["actual_score"] for r, _ in choices),
                    "action_source": chosen["action_source"],
                    "hit": chosen["hit"],
                }
            )
        return decisions

    model = fit(train)
    weights = (0, 0.25, 0.5, 1.0)
    calibration = {str(w): float(np.mean([r["gain"] for r in evaluate(model, cal, w)])) for w in weights}
    weight = max(weights, key=lambda w: calibration[str(w)])
    decisions = evaluate(model, test, weight)
    # block_interval compares RMSE; bootstrap direct score differences instead.
    byweek = defaultdict(list)
    for row in decisions:
        byweek[row["deadline_group"]].append(row["gain"])
    values = np.array([np.mean(byweek[g]) for g in sorted(byweek)])
    rng = np.random.default_rng(83)
    n = len(values)
    starts = rng.integers(0, n, (2000, (n + 3) // 4))
    indices = ((starts[:, :, None] + np.arange(4)) % n).reshape(2000, -1)[:, :n]
    ci = np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist()
    result = {
        "at": now(),
        "dataset_id": dataset["id"],
        "horizon_model_id": dataset["horizon_model_id"],
        "protocol": "Train label windows end before GW20; calibrate GW20–24; test starts GW28 after a three-week purge; current season GW1–5 also held out",
        "features": NAMES,
        "parameters": PARAMETERS,
        "training_action_rows": len(train),
        "calibration_action_rows": len(cal),
        "test_decisions": len(decisions),
        "shrinkage_weight": weight,
        "calibration_gain_by_weight": calibration,
        "mean_heldout_gain": float(np.mean([r["gain"] for r in decisions])),
        "four_deadline_block_gain_95ci": ci,
        "selected_mean": float(np.mean([r["selected"] for r in decisions])),
        "reference_mean": float(np.mean([r["baseline"] for r in decisions])),
        "best_searched_mean": float(np.mean([r["best_searched"] for r in decisions])),
        "promoted": bool(ci[0] > 0),
        "invalid_actions": 0,
        "limits": dataset["limits"]
        + [
            "Overlapping four-week labels use deadline blocks, not independent player-row confidence intervals",
            "Any selected final weights are fitted on all subsequently observed training windows only after this fixed held-out comparison",
        ],
    }
    final = fit(rows)
    directory = (
        store.root
        / "models"
        / ("policy-" + digest({"dataset": dataset["id"], "parameters": PARAMETERS})[:20])
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "critic.txt"
    final.booster_.save_model(str(path))
    result.update(
        directory=directory.name, model_hash=__import__("hashlib").sha256(path.read_bytes()).hexdigest()
    )
    result["id"] = store.put("transfer_policy_model", result)
    atomic_json(Path("artifacts/evaluations/transfer_policy_model.json"), result)
    return result


def personalized_analysis(store: Store):
    import hashlib

    from .horizon import live_values
    from .rules import quote

    model = store.get("transfer_policy_model")
    values, players, market, horizon = live_values(store)
    if model["horizon_model_id"] != horizon["id"]:
        raise ValueError("Transfer critic must be retrained for the current horizon model")
    saved = store.get("state")
    state = TeamState.model_validate({k: v for k, v in saved.items() if k != "id"})
    rules = Rules.from_bootstrap(store.latest("bootstrap")[0])
    forecast = store.get("forecast")
    reference = {r["player_id"]: r["mean"] for r in forecast["players"] if r["event"] == state.event}
    short = {p: v["short"] for p, v in values.items()}
    long = {p: v["fourweek"] for p, v in values.items()}
    options = [("roll", Action())]
    for name, target in (("reference", reference), ("context", short), ("fourweek", long)):
        for maximum in (1, 2):
            options.append((name + str(maximum), greedy_bundle(state, players, rules, target, maximum)))
    unique: dict[tuple, tuple[str, Action]] = {}
    for name, action in options:
        unique.setdefault((tuple(sorted(action.sell)), tuple(sorted(action.buy))), (name, action))
    path = store.root / "models" / model["directory"] / "critic.txt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != model["model_hash"]:
        raise ValueError("Transfer critic integrity failed")
    regressor = lgb.Booster(model_file=str(path))
    lots = {r.player_id: r for r in state.lots}
    rows = []
    for name, action in unique.values():
        squad = sorted({lot.player_id for lot in state.lots} - set(action.sell) | set(action.buy))
        action.lineup = best_lineup(squad, short, players, rules)
        _, ledger = transition(state, action, players, rules)
        x = action_features(state, action, players, rules, short, reference, long)
        # Counterfactual training assumes quotes equal market prices. Correct the
        # live private-account budget features using its exact selling quotes.
        proceeds = sum(quote(lots[p], players[p].price) for p in action.sell)
        x[7] = ledger["bank_after"] / 10
        x[9] = proceeds / max(1, len(action.sell)) / 10
        residual = float(regressor.predict(np.array([x]), num_threads=1)[0])
        value = x[2] - x[3] + model["shrinkage_weight"] * residual
        rows.append(
            {
                "source": name,
                "sell": action.sell,
                "buy": action.buy,
                "transfers": [
                    {"out": players[a].web_name, "in": players[b].web_name}
                    for a, b in zip(action.sell, action.buy)
                ],
                "estimated_fourweek_gain": value,
                "hit": ledger["hit"],
                "bank": ledger["bank_after"],
                "lineup": action.lineup.model_dump(),
            }
        )
    roll = next((r["estimated_fourweek_gain"] for r in rows if not r["buy"]), 0)
    for row in rows:
        row["estimated_fourweek_gain"] -= roll
    from datetime import datetime, timezone

    from .research import active_claims

    evidence = [
        claim
        for claim in active_claims(store)
        if datetime.fromisoformat(claim["expires_at"]) > datetime.now(timezone.utc)
    ]
    for row in rows:
        held = {lot.player_id for lot in state.lots} - set(row["sell"]) | set(row["buy"])
        relevant = [claim for claim in evidence if claim["player_id"] in held]
        row["evidence"] = [
            {k: claim[k] for k in ("player_id", "claim_type", "statement", "source_url", "status")}
            for claim in relevant
        ]
        concerns = {
            claim["player_id"]
            for claim in relevant
            if claim["claim_type"] in ("unavailable", "injury_concern", "suspension", "minutes_restriction")
        }
        stressed = short.copy()
        for player in concerns:
            stressed[player] = 0
        base_line = best_lineup(sorted(held), short, players, rules)
        stress_line = best_lineup(sorted(held), stressed, players, rules)
        row["confirmed_absence_lineup_sensitivity"] = expected(stress_line, stressed) - expected(
            base_line, short
        )
        row["evidence_policy"] = (
            "If concerns become confirmed absences before the deadline, reselect the XI. This conditional stress has no assigned probability or automatic points uplift."
        )
    result = {
        "at": now(),
        "state_id": saved["id"],
        "market_id": market["id"],
        "policy_model_id": model["id"],
        "horizon_model_id": horizon["id"],
        "event": state.event,
        "status": "validated research ranking"
        if model["promoted"]
        else "challenger; confidence gate not passed",
        "automatic_execution": False,
        "candidates": sorted(rows, key=lambda r: -r["estimated_fourweek_gain"]),
        "heldout_mean_gain": model["mean_heldout_gain"],
        "heldout_gain_95ci": model["four_deadline_block_gain_95ci"],
        "limits": [
            "The historical confidence interval describes average counterfactual gains, not this individual transfer",
            "No future chips or intervening transfers inside this four-week critic; the full planner handles chip policies",
            "Exact current selling quotes are used; future price changes are not known",
            "A ranked candidate is not a proof of globally optimal FPL performance",
        ],
    }
    result["id"] = store.put("personalized_policy_analysis", result)
    return result
