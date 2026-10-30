"""Decision-focused, evolving-account learning with disjoint outcome windows.

Hindsight is a training label and audit ceiling. Every rollout chooses later
moves from that later deadline's out-of-fold predictions, never its outcomes.
"""

from __future__ import annotations

import hashlib
import json
import lzma
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import httpx
import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.optimize import lsq_linear

from .context_archive import MANIFEST, load_context
from .decision import canonical_rows
from .domain import ChipUse, Lineup, Player
from .horizon import block_interval
from .policy_learning import random_state
from .rules import Rules, score_gameweek, selling_price
from .storage import Store, atomic_json, digest, now
from .strategy import candidate_actions, solve_action

VERSION = "evolving-decisions-v3"
STATE_FEATURES = (
    "squad_future_points",
    "bank_millions",
    "free_transfers",
    "wildcard",
    "freehit",
    "bboost",
    "3xc",
)
ACTION_FEATURES = (
    "reference_points",
    "tree_points",
    "network_points",
    "fourweek_points",
    "hit",
    "purchases",
    "bank_after",
    "ft_after",
    "wildcard",
    "freehit",
    "bboost",
    "3xc",
    "squad_future_points",
    "expiring_chip",
    "zero_chance_owned",
    "disagreement",
)


def season_rules(store, season):
    if season == store.get("market")["season"]:
        return Rules.from_bootstrap(store.latest("bootstrap")[0])
    source = "season_rules_" + season.replace("/", "-")
    try:
        bootstrap = store.latest(source)[0]
    except ValueError:
        row = next(r for r in json.loads(MANIFEST.read_text())["snapshots"] if r["season"] == season)
        response = httpx.get(row["url"], timeout=30)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != row["sha256"]:
            raise ValueError("Historical rules archive hash changed")
        original = json.loads(lzma.decompress(response.content, memlimit=128_000_000))
        bootstrap = {k: original[k] for k in ("game_settings", "element_types", "chips", "events")}
        bootstrap["game_config"] = original.get("game_config", {})
        # Preserve the official season-identifying path without inventing a year.
        bootstrap["element_stats"] = original.get("element_stats", [])
        bootstrap["total_players"] = original.get("total_players")
        bootstrap["elements"] = original["elements"][:1]
        store.snapshot(source, row["url"], bootstrap, row["source_captured_at"])
    rules = Rules.from_bootstrap(bootstrap)
    if rules.season != season:
        raise ValueError("Archived rules season mismatch")
    return rules


def training_markets(store):
    spec = store.get("fusion_model")
    heldout = pl.read_parquet(store.artifact_path(spec["prediction_path"])).to_dicts()
    forecasts: dict[tuple, dict[str, dict]] = defaultdict(
        lambda: {k: defaultdict(float) for k in ("reference", "tree", "network")}
    )
    for row in heldout:
        key = (row["season"], row["event"])
        for kind, field in (("reference", "baseline"), ("tree", "tree"), ("network", "network")):
            forecasts[key][kind][int(row["player_id"])] += float(row[field])
    long_model = store.get("horizon_model")
    longs: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in pl.read_parquet(store.artifact_path(long_model["prediction_path"])).to_dicts():
        longs[r["season"], r["event"]][r["player_id"]] = float(r[long_model["selected"]])
    actual: dict[tuple, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    minutes: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    rows, _, _ = canonical_rows(store)
    for r in rows:
        key = (r["season"], int(r["round"]))
        p = int(r["element"])
        actual[key][p] += float(r.get("total_points") or 0)
        minutes[key][p] += int(r.get("minutes") or 0)
    del rows
    registry: dict[tuple, dict[int, Player]] = defaultdict(dict)
    for (season, event, _), packet in load_context(store).items():
        if (season, event) not in forecasts:
            continue
        p = int(packet["id"])
        registry[season, event][p] = Player(
            id=p,
            code=int(packet["code"]),
            season=season,
            name=str(p),
            web_name=str(p),
            team=int(packet["team"]),
            position=cast(Any, int(packet["element_type"])),
            price=int(packet["now_cost"]),
            selectable=packet.get("status") != "u",
            ownership=float(packet.get("selected_by_percent") or 0),
            status=packet.get("status") or "a",
            chance=float(packet["chance_of_playing_next_round"]) / 100
            if packet.get("chance_of_playing_next_round") is not None
            else None,
            news=packet.get("news") or "",
            raw=packet,
        )
    return registry, forecasts, longs, actual, minutes


def reprice(state, players, rules):
    """Observed next-deadline prices alter lots; no private acquisition guessing."""
    missing = {lot.player_id for lot in state.lots} - players.keys()
    if missing:
        raise ValueError("Historical registration missing owned players: " + str(sorted(missing)))
    lots = [
        lot.model_copy(
            update={
                "sell_price": selling_price(cast(int, lot.purchase_price), players[lot.player_id].price),
                "quote_market_price": players[lot.player_id].price,
            }
        )
        for lot in state.lots
    ]
    counts = Counter(players[p.player_id].team for p in lots)
    return state.model_copy(
        update={"lots": lots, "grandfathered_clubs": [c for c, n in counts.items() if n > rules.club_limit]}
    )


def state_features(state, players, rules, long, short):
    future = sum(
        max(-10, long.get(lot.player_id, 4 * short.get(lot.player_id, 0)) - short.get(lot.player_id, 0))
        for lot in state.lots
    )
    flags = []
    for chip in ("wildcard", "freehit", "bboost", "3xc"):
        try:
            a, b = rules.chip_window(chip, state.event)
            unused = not any(u.chip == chip and a <= u.event <= b for u in state.used_chips)
            flags.append(float(unused) * min(1, (b - state.event + 1) / 4))
        except ValueError:
            flags.append(0.0)
    return [future, state.bank / 10, state.free_transfers, *flags]


def action_features(decision, before, players, rules, forecasts, long):
    line = decision.action.lineup
    assert line is not None
    vals = []
    for name in ("reference", "tree", "network"):
        q = forecasts[name]
        score = sum(q.get(p, 0) for p in line.xi) + q.get(line.captain, 0)
        if decision.action.chip == "3xc":
            score += q.get(line.captain, 0)
        if decision.action.chip == "bboost":
            score += sum(q.get(p, 0) for p in line.bench)
        vals.append(score - decision.ledger["hit"])
    chips = [float(decision.action.chip == c) for c in ("wildcard", "freehit", "bboost", "3xc")]
    longvalue = sum(long.get(p, 4 * forecasts["reference"].get(p, 0)) for p in line.xi) + long.get(
        line.captain, 4 * forecasts["reference"].get(line.captain, 0)
    )
    expiring = 0.0
    if decision.action.chip:
        _, end = rules.chip_window(decision.action.chip, before.event)
        expiring = 1 / max(1, end - before.event + 1)
    return [
        *vals,
        longvalue,
        decision.ledger["hit"],
        len(decision.action.buy),
        decision.state.bank / 10,
        decision.state.free_transfers,
        *chips,
        state_features(decision.state, players, rules, long, forecasts["reference"])[0],
        expiring,
        sum(players[p].chance == 0 for p in decision.ledger["permanent_squad"]),
        max(vals) - min(vals),
    ]


def evolve(store: Store, states_per_week=24, horizon=4, seconds=0.15, max_transfers=15):
    registry, forecasts, longs, outcomes, minutes = training_markets(store)
    rules_by_season = {s: season_rules(store, s) for s, _ in registry}
    lineage = {
        "version": VERSION,
        "fusion": store.get("fusion_model")["id"],
        "horizon_model": store.get("horizon_model")["id"],
        "context": store.get("context_archive")["id"],
        "samples": states_per_week,
        "horizon": horizon,
        "seconds": seconds,
        "ordinary_transfer_limit": max_transfers,
        "rules": {s: r.hash for s, r in rules_by_season.items()},
    }
    directory = store.root / "decision-checkpoints" / digest(lineage)[:24]
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    continuations = []
    diagnostics = []
    for season, event in sorted(registry):
        if (
            event == 1
            or (season, event) not in longs
            or not all((season, g) in registry for g in range(event, event + horizon))
        ):
            continue
        group = f"{season}-GW{event:02d}"
        checkpoint = directory / (group.replace("/", "-") + ".json")
        if checkpoint.exists():
            cached = json.loads(checkpoint.read_text())
            valid = all(
                len({u["event"] for u in r["state"]["used_chips"]}) == len(r["state"]["used_chips"])
                for r in cached["actions"]
            )
            if not valid:
                checkpoint.rename(checkpoint.with_suffix(".rejected"))
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            records += saved["actions"]
            continuations += saved["states"]
            diagnostics += saved["diagnostics"]
            print("Restored evolving deadline", group, flush=True)
            continue
        players = registry[season, event]
        q = forecasts[season, event]
        rules = rules_by_season[season]
        rng = np.random.default_rng([117, int(digest(group)[:8], 16)])
        action_rows = []
        state_rows = []
        audit = []
        for sample in range(states_per_week):
            state = random_state(players, q["reference"], rules, event, season, rng)
            uses = []
            if event > 2:
                for chip in ("wildcard", "freehit", "bboost", "3xc"):
                    try:
                        a, _ = rules.chip_window(chip, event)
                        if a < event and rng.random() < 0.5:
                            uses.append(ChipUse(chip=chip, event=int(rng.integers(a, event))))
                    except ValueError:
                        pass
            # A real account can activate only one chip at a deadline.
            dates = set()
            lawful = []
            for usage in uses:
                a, _ = rules.chip_window(usage.chip, event)
                choices = [g for g in range(a, event) if g not in dates]
                if choices:
                    chosen = usage.event if usage.event in choices else int(rng.choice(choices))
                    lawful.append(usage.model_copy(update={"event": chosen}))
                    dates.add(chosen)
            state = state.model_copy(update={"used_chips": lawful})
            candidates, failures = candidate_actions(
                state, players, rules, q, seconds=seconds, alternatives=1, ordinary_limit=max_transfers
            )
            oracle = solve_action(
                state,
                players,
                rules,
                outcomes[season, event],
                seconds=max(0.5, seconds),
                max_transfers=max_transfers,
                relative_gap=0,
            )
            for number, d in enumerate(candidates):
                immediate = (
                    score_gameweek(
                        outcomes[season, event],
                        minutes[season, event],
                        d.action.lineup,
                        players,
                        rules,
                        d.action.chip,
                    )
                    - d.ledger["hit"]
                )
                following = d.state
                future = 0.0
                ledger = [d.ledger]
                skipped = None
                for g in range(event + 1, event + horizon):
                    market = registry[season, g]
                    pred = forecasts[season, g]["reference"]
                    try:
                        following = reprice(following, market, rules)
                        # Future information is revealed only at its own deadline.
                        choices, _ = candidate_actions(
                            following,
                            market,
                            rules,
                            {"reference": pred},
                            seconds=max(0.025, seconds / 4),
                            alternatives=1,
                            chips=True,
                            ordinary_limit=max_transfers,
                        )
                        step = max(choices, key=lambda c: c.utility)
                    except ValueError as error:
                        skipped = str(error)
                        break
                    future += (
                        score_gameweek(
                            outcomes[season, g],
                            minutes[season, g],
                            step.action.lineup,
                            market,
                            rules,
                            step.action.chip,
                        )
                        - step.ledger["hit"]
                    )
                    following = step.state
                    ledger.append(step.ledger)
                if skipped:
                    audit.append({"group": group, "sample": sample, "excluded": skipped})
                    continue
                features = action_features(d, state, players, rules, q, longs[season, event])
                key = f"{group}-{sample}"
                action_rows.append(
                    {
                        "group": group,
                        "label_end": f"{season}-GW{event + horizon - 1:02d}",
                        "account": key,
                        "candidate": number,
                        "x": features,
                        "immediate": float(immediate),
                        "return": float(immediate + future),
                        "predicted_reference": features[0],
                        "chip": d.action.chip or "none",
                        "action": d.action.model_dump(),
                        "ledger": ledger,
                        "solver": d.solver,
                        "state": state.model_dump(),
                        "oracle_immediate": float(
                            score_gameweek(
                                outcomes[season, event],
                                minutes[season, event],
                                oracle.action.lineup,
                                players,
                                rules,
                            )
                            - oracle.ledger["hit"]
                        ),
                        "oracle_gap": oracle.solver["gap"],
                    }
                )
                state_rows.append(
                    {
                        "group": group,
                        "label_end": f"{season}-GW{event + horizon - 1:02d}",
                        "x": state_features(d.state, players, rules, longs[season, event], q["reference"]),
                        "target": float(future),
                    }
                )
            audit.append(
                {
                    "group": group,
                    "sample": sample,
                    "candidate_count": len(candidates),
                    "search_exclusions": failures,
                }
            )
        saved = {"actions": action_rows, "states": state_rows, "diagnostics": audit}
        atomic_json(checkpoint, saved)
        records += action_rows
        continuations += state_rows
        diagnostics += audit
        print(
            "Completed evolving deadline",
            group,
            "accounts",
            states_per_week,
            "actions",
            len(action_rows),
            flush=True,
        )
    if not records:
        raise ValueError("No complete evolving-account episodes")
    rescore_records(records, continuations, registry, outcomes, minutes, rules_by_season)
    path = store.root / "features" / ("evolving-" + digest(lineage)[:24] + "-scored-v2.json.gz")
    import gzip

    path.write_bytes(
        gzip.compress(
            json.dumps({"actions": records, "states": continuations}, allow_nan=False).encode(), mtime=0
        )
    )
    report = {
        "at": now(),
        "scoring_version": "all-deadline-chips-v2",
        "lineage": lineage,
        "path": str(path),
        "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "accounts": len({r["account"] for r in records}),
        "actions": len(records),
        "horizon": horizon,
        "groups": sorted({r["group"] for r in records}),
        "rules": "Each action replays observed prices, lots, bank, FT, autosubs, captain fallback and chips; later policies only use their own deadline forecasts",
        "exclusions": [d for d in diagnostics if "excluded" in d],
        "oracle": "Independent full-market hindsight additive transfer solve, exact realized rescoring and solver gap retained; not a multiweek clairvoyant optimum",
        "timing": "Only completed checkpoints are reused; hindsight outcomes and oracle choices are labels, never features",
    }
    report["id"] = store.put("evolving_dataset", report)
    atomic_json(Path("artifacts/evaluations/evolving_dataset.json"), report)
    return report


def load_episodes(store):
    import gzip

    spec = store.get("evolving_dataset")
    path = store.artifact_path(spec["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["hash"]:
        raise ValueError("Evolving dataset checksum mismatch")
    data = json.loads(gzip.decompress(path.read_bytes()))
    if spec.get("scoring_version") != "all-deadline-chips-v2":
        registry, _, _, outcomes, minutes = training_markets(store)
        rules = {season: season_rules(store, season) for season, _ in registry}
        rescore_records(data["actions"], data["states"], registry, outcomes, minutes, rules)
        repaired = path.with_name(path.name.replace(".json.gz", "-scored-v2.json.gz"))
        repaired.write_bytes(gzip.compress(json.dumps(data, allow_nan=False).encode(), mtime=0))
        spec = {k: v for k, v in spec.items() if k != "id"}
        spec.update(
            at=now(),
            path=str(repaired),
            hash=hashlib.sha256(repaired.read_bytes()).hexdigest(),
            scoring_version="all-deadline-chips-v2",
        )
        spec["id"] = store.put("evolving_dataset", spec)
        atomic_json(Path("artifacts/evaluations/evolving_dataset.json"), spec)
    return data, spec


def rescore_records(actions, states, registry, outcomes, minutes, rules):
    """Independent outcome replay includes every future chip, including cached runs."""
    if len(actions) != len(states):
        raise ValueError("Action and post-state training rows must match")
    for action, post in zip(actions, states):
        season = action["group"].split("-GW")[0]
        scores = []
        for row in action["ledger"]:
            key = season, row["event"]
            scores.append(
                float(
                    score_gameweek(
                        outcomes[key],
                        minutes[key],
                        Lineup.model_validate(row["lineup"]),
                        registry[key],
                        rules[season],
                        row["chip"],
                    )
                    - row["hit"]
                )
            )
        action["immediate"] = scores[0]
        action["return"] = sum(scores)
        post["target"] = sum(scores[1:])


def partition(rows):
    train = [r for r in rows if r["label_end"] < "2025/26-GW20"]
    cal = [r for r in rows if "2025/26-GW20" <= r["group"] <= "2025/26-GW24"]
    test = [r for r in rows if r["group"] >= "2025/26-GW28"]
    if not all((train, cal, test)):
        raise ValueError("Purged train/calibration/test partitions require more completed episodes")
    return train, cal, test


def gain_interval(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["group"]].append(r["gain"])
    values = np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    rng = np.random.default_rng(117)
    n = len(values)
    starts = rng.integers(0, n, (2000, (n + 3) // 4))
    indices = ((starts[:, :, None] + np.arange(4)) % n).reshape(2000, -1)[:, :n]
    return np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist()


def fit_critic(train, cal, seed=117):
    """Listwise regret learning: hindsight return differences supervise choices."""
    import torch
    from torch import nn

    torch.set_num_threads(3)
    torch.manual_seed(seed)
    x = np.array([r["x"] for r in train], dtype=np.float32)
    mu = x.mean(0)
    sd = np.maximum(x.std(0), 0.01)
    model = nn.Sequential(
        nn.Linear(len(ACTION_FEATURES), 64),
        nn.SiLU(),
        nn.LayerNorm(64),
        nn.Linear(64, 32),
        nn.SiLU(),
        nn.Linear(32, 1),
    )
    with torch.no_grad():
        cast(torch.Tensor, cast(nn.Linear, model[-1]).bias).fill_(
            float(np.mean([r["return"] - r["predicted_reference"] for r in train]))
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    groups = defaultdict(list)
    for i, r in enumerate(train):
        groups[r["account"]].append(i)
    xx = torch.tensor((x - mu) / sd)
    target = torch.tensor([r["return"] for r in train], dtype=torch.float32)
    baseline = torch.tensor([r["predicted_reference"] for r in train], dtype=torch.float32)
    best = None
    bestloss = float("inf")
    patience = 0
    cx = torch.tensor((np.array([r["x"] for r in cal], dtype=np.float32) - mu) / sd)
    cy = np.array([r["return"] for r in cal])
    cb = np.array([r["predicted_reference"] for r in cal])
    for epoch in range(250):
        model.train()
        optimizer.zero_grad()
        pred = model(xx).squeeze(1) + baseline
        loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=10)
        for idx in groups.values():
            if len(idx) > 1:
                # Both realized winner and near-winners contribute; no inference sees them.
                preference = torch.softmax((target[idx] - target[idx].max()) / 5, dim=0)
                loss = loss - (preference * torch.log_softmax(pred[idx] / 5, dim=0)).sum() / len(groups) * 5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            raw = model(cx).squeeze(1).numpy() + cb
        value = float(np.mean((raw - cy) ** 2))
        if value < bestloss:
            bestloss = value
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 6:
            break
    assert best is not None
    model.load_state_dict(best)
    return model, mu, sd, epoch + 1


def train_decision_models(store):
    import torch

    data, dataset = load_episodes(store)
    train, cal, test = partition(data["actions"])
    tree = lgb.LGBMRegressor(
        n_estimators=200,
        num_leaves=12,
        min_child_samples=50,
        reg_lambda=30,
        learning_rate=0.03,
        n_jobs=1,
        verbosity=-1,
        random_state=117,
    )
    tree.fit([r["x"] for r in train], [r["return"] - r["predicted_reference"] for r in train])
    network, mu, sd, epochs = fit_critic(train, cal)

    def predictions(rows, kind):
        x = np.array([r["x"] for r in rows], dtype=np.float32)
        if kind == "tree":
            return tree.predict(x, num_threads=1)
        network.eval()
        with torch.no_grad():
            return network(torch.tensor((x - mu) / sd)).squeeze(1).numpy()

    def decisions(rows, residual, weight):
        groups = defaultdict(list)
        for r, v in zip(rows, residual):
            groups[r["account"]].append((r, float(v)))
        result = []
        for choices in groups.values():
            baseline = max(choices, key=lambda a: a[0]["predicted_reference"])[0]
            selected = max(choices, key=lambda a: a[0]["predicted_reference"] + weight * a[1])[0]
            result.append(
                {
                    "group": selected["group"],
                    "account": selected["account"],
                    "gain": selected["return"] - baseline["return"],
                    "score": selected["return"],
                    "baseline": baseline["return"],
                    "best_searched": max(r["return"] for r, _ in choices),
                    "first_chip": selected["chip"],
                }
            )
        return result

    candidates = {}
    weights = {}
    for kind in ("tree", "network"):
        cp = predictions(cal, kind)
        choice = {
            str(w): float(np.mean([r["gain"] for r in decisions(cal, cp, w)])) for w in (0, 0.25, 0.5, 1)
        }
        weight = max((0, 0.25, 0.5, 1), key=lambda w: choice[str(w)])
        weights[kind] = weight
        result = decisions(test, predictions(test, kind), weight)
        ci = gain_interval(result)
        candidates[kind] = {
            "calibration_gain_by_weight": choice,
            "weight": weight,
            "accounts": len(result),
            "mean_gain": float(np.mean([r["gain"] for r in result])),
            "block_gain_95ci": ci,
            "mean_points": float(np.mean([r["score"] for r in result])),
            "reference_points": float(np.mean([r["baseline"] for r in result])),
            "best_searched_points": float(np.mean([r["best_searched"] for r in result])),
            "promoted": bool(ci[0] > 0 and weight > 0),
        }
    calibrated_kind = max(
        candidates, key=lambda k: candidates[k]["calibration_gain_by_weight"][str(candidates[k]["weight"])]
    )
    selected = calibrated_kind if candidates[calibrated_kind]["promoted"] else None
    directory = store.root / "models" / ("decision-critic-" + dataset["id"][:20])
    directory.mkdir(parents=True, exist_ok=True)
    tree_path = directory / "tree.txt"
    tree.booster_.save_model(str(tree_path))
    neural_path = directory / "network.pt"
    torch.save(network.state_dict(), neural_path)
    report = {
        "at": now(),
        "dataset_id": dataset["id"],
        "version": VERSION,
        "features": ACTION_FEATURES,
        "protocol": "Labels end before GW20; calibration GW20–24; later tests GW28+, purging overlapping four-week windows. Candidate regret and forecast residual losses share the same legal decisions.",
        "training_rows": len(train),
        "calibration_rows": len(cal),
        "test_rows": len(test),
        "candidates": candidates,
        "selected": selected,
        "architecture_selected_on_calibration": calibrated_kind,
        "neural": {
            "epochs": epochs,
            "mean": mu.tolist(),
            "scale": sd.tolist(),
            "parameters": sum(p.numel() for p in network.parameters()),
        },
        "models": {
            k: {"path": str(p), "hash": hashlib.sha256(p.read_bytes()).hexdigest()}
            for k, p in [("tree", tree_path), ("network", neural_path)]
        },
        "invalid_actions": 0,
        "all_market": True,
        "outcome_features": False,
        "limits": [
            "Finite candidate and solver budgets are exposed; oracle is an outcome-only audit, not future knowledge",
            "Historical news and price archive limitations remain; model selection is frozen before later test outcomes",
            "The final critic retains the earlier calibrated fit rather than silently retraining on its own test data",
        ],
    }
    report["id"] = store.put("decision_critic", report)
    atomic_json(Path("artifacts/evaluations/decision_critic.json"), report)
    train_state_value(store, data["states"], dataset, data["actions"])
    return report


def train_state_value(store, rows, dataset, actions=None):
    train, cal, test = partition(rows)

    def fit(items, ridge):
        x = np.array([r["x"] for r in items])
        y = np.array([r["target"] for r in items])
        sd = np.maximum(x.std(0), 1)
        z = np.column_stack([np.ones(len(x)), x / sd])
        reg = np.diag([0] + [ridge] * x.shape[1])
        fit = lsq_linear(
            np.vstack([z, reg]),
            np.concatenate([y, np.zeros(z.shape[1])]),
            bounds=([-np.inf] + [0] * x.shape[1], [np.inf] * z.shape[1]),
        )
        return float(fit.x[0]), fit.x[1:] / sd

    best = None
    for ridge in (1, 5, 20):
        intercept, coef = fit(train, ridge)
        error = np.mean([(intercept + np.dot(r["x"], coef) - r["target"]) ** 2 for r in cal])
        if best is None or error < best[0]:
            best = (error, ridge, intercept, coef)
    assert best is not None
    _, ridge, intercept, coef = best
    baseline_offset = float(np.mean([r["target"] - 0.1 * r["x"][0] for r in train]))
    pred = np.array([intercept + np.dot(r["x"], coef) for r in test])
    y = np.array([r["target"] for r in test])
    prior = np.array([baseline_offset + 0.1 * r["x"][0] for r in test])
    comparisons = [
        {"deadline_group": r["group"], "actual": r["target"], "learned": float(v), "baseline": float(b)}
        for r, v, b in zip(test, pred, prior)
    ]
    ci = block_interval(comparisons, "learned")
    policy_review = None
    if actions is not None:
        choices = defaultdict(list)
        for action, post in zip(actions, rows):
            if action["group"] >= "2025/26-GW28":
                choices[action["account"]].append((action, post))
        comparison = []
        for options in choices.values():
            base = max(options, key=lambda pair: pair[0]["predicted_reference"] + 0.1 * pair[1]["x"][0])[0]
            chosen = max(
                options, key=lambda pair: pair[0]["predicted_reference"] + np.dot(pair[1]["x"], coef)
            )[0]
            comparison.append({"group": chosen["group"], "gain": chosen["return"] - base["return"]})
        interval = gain_interval(comparison)
        policy_review = {
            "accounts": len(comparison),
            "mean_gain": float(np.mean([r["gain"] for r in comparison])),
            "block_gain_95ci": interval,
            "passed": bool(interval[0] > 0),
        }
    report = {
        "at": now(),
        "dataset_id": dataset["id"],
        "features": STATE_FEATURES,
        "feature_bounds": [
            [float(min(r["x"][i] for r in train)), float(max(r["x"][i] for r in train))]
            for i in range(len(STATE_FEATURES))
        ],
        "policy_review": policy_review,
        "intercept": intercept,
        "coefficients": coef.tolist(),
        "ridge": ridge,
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "baseline_rmse": float(np.sqrt(np.mean((prior - y) ** 2))),
        "paired_rmse_95ci": ci,
        "promoted": bool(ci[1] < 0 and policy_review and policy_review["passed"]),
        "training_rows": len(train),
        "test_rows": len(test),
        "protocol": "Nonnegative regularized continuation-value coefficients; fit earlier labels, select ridge on calibration, report untouched later-deadline block interval",
        "baseline": "Fixed 0.1 squad future points with training-only fitted intercept; bank, FT and unused chips zero",
        "scope": "Learned four-week continuation under the explicitly replayed future policy; intercept cancels within a fixed planning horizon",
    }
    report["id"] = store.put("state_value_model", report)
    atomic_json(Path("artifacts/evaluations/state_value_model.json"), report)
    return report
