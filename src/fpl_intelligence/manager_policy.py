"""Deadline-separated learning of manager purchases, sales and chip decisions."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.special import logit, softmax
from sklearn.metrics import log_loss

from .manager_behavior import behavior_rows, context_packets
from .storage import atomic_json, now

CHIPS = ("none", "wildcard", "freehit", "bboost", "3xc")
PLAYER_FEATURES = (
    "price",
    "position",
    "ownership",
    "incoming",
    "outgoing",
    "chance",
    "chance_known",
    "injury",
    "reference_points",
    "tree_points",
    "network_points",
    "club_owned",
    "owned",
    "previous_captain",
    "recent_purchase",
    "recent_sale",
    "bank",
    "free_transfers",
    "roster_injuries",
    "event",
)


def choice_features(packet, p, owned, previous, transfers, bank, ft, event, packets, points):
    chance = packet.get("chance_of_playing_next_round")
    return [
        float(packet["now_cost"]) / 10,
        float(packet["element_type"]),
        float(packet.get("selected_by_percent") or 0) / 100,
        np.log1p(float(packet.get("transfers_in_event") or 0)),
        np.log1p(float(packet.get("transfers_out_event") or 0)),
        float(chance) / 100 if chance is not None else 1.0,
        float(chance is not None),
        float(packet.get("status") in ("i", "s", "d", "u")),
        *[points.get(p, {}).get(k, 0) for k in ("baseline", "tree", "network")],
        sum(packets.get(q, {}).get("team") == packet["team"] for q in owned),
        float(p in owned),
        float(any(r["element"] == p and r.get("is_captain") for r in previous)),
        sum(r["element_in"] == p for r in transfers),
        sum(r["element_out"] == p for r in transfers),
        bank / 10,
        ft,
        sum(packets.get(q, {}).get("status") in ("d", "i", "s", "u") for q in owned),
        event / 38,
    ]


def manager_inputs(manager, event, packets, pointmap, ft):
    history = manager.get("history") or {}
    chipmap = {r["event"]: r["name"] for r in history.get("chips", [])}
    roster = event - 2 if chipmap.get(event - 1) == "freehit" else event - 1
    previous = manager["picks"].get(str(roster), {}).get("picks", [])
    owned = {r["element"] for r in previous}
    finances: dict = next((r for r in history.get("current", []) if r["event"] == roster), {})
    transfers = [r for r in manager.get("transfers", []) if event - 3 <= r["event"] < event]
    context = {p: v for (e, p), v in packets.items() if e == event}
    values = {
        p: choice_features(
            v,
            p,
            owned,
            previous,
            transfers,
            finances.get("bank", 0),
            ft,
            event,
            context,
            pointmap.get(event, {}),
        )
        for p, v in context.items()
    }
    return values, owned, context


def train_manager_policy(store):
    import polars as pl

    report = store.get("top_manager_history")
    packets = context_packets(store)
    season = report["season"]
    members = {m["entry"]: m for m in report["members"]}
    pointmap: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
    fusion = store.get("fusion_model")
    for r in pl.read_parquet(store.artifact_path(fusion["prediction_path"])).to_dicts():
        if r["season"] == season:
            d = pointmap[r["event"]].setdefault(
                r["player_id"], {k: 0.0 for k in ("baseline", "tree", "network")}
            )
            for k in d:
                d[k] += float(r[k])
    decisions = [r for r in behavior_rows(report, packets) if r["label"] is not None]
    last = max(r["event"] for r in decisions)
    rng = np.random.default_rng(143)
    training: dict[str, list[dict[str, Any]]] = {"buy": [], "sell": []}
    evaluation: dict[str, list[dict[str, Any]]] = {"buy": [], "sell": []}
    for i, r in enumerate(decisions):
        m = members[r["entry"]]
        event = r["event"]
        values, owned, context = manager_inputs(m, event, packets, pointmap, int(r["x"][1]))
        transfers = [t for t in m.get("transfers", []) if t["event"] == event]
        for kind, field, pool in (
            ("buy", "element_in", set(values) - owned),
            ("sell", "element_out", owned & values.keys()),
        ):
            positives = {t[field] for t in transfers} & pool
            if not positives:
                continue
            negatives = sorted(pool - positives)
            chosen = sorted(positives) + [
                int(p) for p in rng.choice(negatives, min(len(negatives), 20 * len(positives)), replace=False)
            ]
            if event < last - 3:
                training[kind].extend(
                    {
                        "x": values[p],
                        "y": int(p in positives),
                        "weight": 1
                        if p in positives
                        else len(negatives) / max(1, len(chosen) - len(positives)),
                    }
                    for p in chosen
                )
            else:
                evaluation[kind].append(
                    {
                        "event": event,
                        "entry": m["entry"],
                        "pool": sorted(pool),
                        "x": [values[p] for p in sorted(pool)],
                        "positive": positives,
                        "baseline": [
                            max(0.001, float(context[p].get("selected_by_percent") or 0) / 100)
                            if kind == "buy"
                            else max(
                                0.001,
                                1
                                - float(
                                    100
                                    if context[p].get("chance_of_playing_next_round") is None
                                    else context[p]["chance_of_playing_next_round"]
                                )
                                / 100,
                            )
                            for p in sorted(pool)
                        ],
                    }
                )
        if (i + 1) % 1000 == 0:
            print("Prepared manager choice decisions", i + 1, flush=True)
    models = {}
    metrics = {}
    directory = store.root / "models" / ("manager-policy-" + report["id"][:20])
    directory.mkdir(parents=True, exist_ok=True)
    for kind in ("buy", "sell"):
        rows = training[kind]
        model = lgb.LGBMClassifier(
            n_estimators=160,
            num_leaves=14,
            min_child_samples=100,
            reg_lambda=30,
            learning_rate=0.035,
            n_jobs=1,
            verbosity=-1,
            random_state=143,
        )
        model.fit([r["x"] for r in rows], [r["y"] for r in rows], sample_weight=[r["weight"] for r in rows])
        diagnostics = []
        # Temperature is picked on the first two later deadlines, never test weeks.
        for item in evaluation[kind]:
            raw = np.asarray(model.predict_proba(item["x"]))[:, 1]
            item["logits"] = logit(np.clip(raw, 1e-6, 1 - 1e-6))

        def score(items, t):
            vals = []
            for item in items:
                probs = softmax(item["logits"] / t)
                b = np.array(item["baseline"])
                b /= b.sum()
                indices = [j for j, p in enumerate(item["pool"]) if p in item["positive"]]
                vals.append(
                    {
                        "event": item["event"],
                        "loss": float(-np.log(probs[indices]).mean()),
                        "baseline": float(-np.log(b[indices]).mean()),
                        "top5_recall": len(
                            set(np.array(item["pool"])[np.argsort(probs)[-5:]]) & item["positive"]
                        )
                        / len(item["positive"]),
                    }
                )
            return vals

        cal = [r for r in evaluation[kind] if r["event"] < last - 1]
        test = [r for r in evaluation[kind] if r["event"] >= last - 1]
        temperature = min((0.5, 1.0, 2.0, 4.0), key=lambda t: np.mean([r["loss"] for r in score(cal, t)]))
        diagnostics = score(test, temperature)
        loss = float(np.mean([r["loss"] for r in diagnostics]))
        base = float(np.mean([r["baseline"] for r in diagnostics]))
        metrics[kind] = {
            "temperature": temperature,
            "test_decisions": len(diagnostics),
            "conditional_log_loss": loss,
            "baseline_log_loss": base,
            "top5_recall": float(np.mean([r["top5_recall"] for r in diagnostics])),
            "promoted": loss < base,
        }
        path = directory / (kind + ".txt")
        model.booster_.save_model(str(path))
        models[kind] = {"path": str(path), "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    chiptrain = [r for r in decisions if r["event"] < last - 3]
    chipcal = [r for r in decisions if last - 3 <= r["event"] < last - 1]
    chiptest = [r for r in decisions if r["event"] >= last - 1]
    chip = lgb.LGBMClassifier(
        n_estimators=120,
        num_leaves=10,
        min_child_samples=100,
        reg_lambda=30,
        n_jobs=1,
        verbosity=-1,
        random_state=144,
    )
    chip.fit([r["x"] for r in chiptrain], [CHIPS.index(r["chip"]) for r in chiptrain])
    frequency = (np.bincount([CHIPS.index(r["chip"]) for r in chiptrain], minlength=5) + 1) / (
        len(chiptrain) + 5
    )

    def chipprob(rows):
        p = np.full((len(rows), 5), 1e-8)
        p[:, chip.classes_.astype(int)] = chip.predict_proba([r["x"] for r in rows])
        return p / p.sum(1, keepdims=True)

    cp = chipprob(chipcal)
    cy = [CHIPS.index(r["chip"]) for r in chipcal]
    weight = min(
        (0, 0.25, 0.5, 0.75, 1),
        key=lambda w: log_loss(cy, w * cp + (1 - w) * frequency, labels=list(range(5))),
    )
    tp = weight * chipprob(chiptest) + (1 - weight) * frequency
    ty = [CHIPS.index(r["chip"]) for r in chiptest]
    loss = float(log_loss(ty, tp, labels=list(range(5))))
    base = float(log_loss(ty, np.tile(frequency, (len(ty), 1)), labels=list(range(5))))
    metrics["chip"] = {
        "weight": weight,
        "frequency": frequency.tolist(),
        "classes": chip.classes_.astype(int).tolist(),
        "log_loss": loss,
        "baseline_log_loss": base,
        "promoted": loss < base,
        "test_decisions": len(ty),
    }
    path = directory / "chip.txt"
    chip.booster_.save_model(str(path))
    models["chip"] = {"path": str(path), "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {
        "at": now(),
        "ordinary_transfer_tail": [
            int(r["event_transfers"])
            for m in report["members"]
            for r in (m.get("history") or {}).get("current", [])
            if r.get("event_transfers", 0) >= 3
            and not any(
                c["event"] == r["event"] and c["name"] in ("wildcard", "freehit")
                for c in (m.get("history") or {}).get("chips", [])
            )
        ]
        or [3],
        "population_id": report["id"],
        "fusion_model_id": fusion["id"],
        "models": models,
        "player_features": PLAYER_FEATURES,
        "metrics": metrics,
        "test_events": [last - 1, last],
        "calibration_events": [last - 3, last - 2],
        "trained_before_event": last - 3,
        "protocol": "Earlier deadline training, next two deadlines for mixture/temperature, final two untouched for choices and chips; full registered candidate pools at evaluation",
        "limits": [
            "Current top-1,000 selection is retrospective and test spans only two deadlines",
            "Conditional choice ranks are not knowledge of hidden moves; resulting branches must pass exact portfolio/FT/chip constraints",
            "Negative sampling uses inverse sampling weights; identity, current rank and later team choices are not predictors",
        ],
    }
    result["id"] = store.put("manager_policy", result)
    atomic_json(Path("artifacts/evaluations/manager_policy.json"), result)
    return result


def policy_models(store):
    spec = store.get("manager_policy")
    models = {}
    for name, item in spec["models"].items():
        path = store.artifact_path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["hash"]:
            raise ValueError("Manager policy checksum mismatch")
        models[name] = lgb.Booster(model_file=str(path))
    return spec, models


def action_preferences(store, state, players, means, models=None):
    spec, bundle = models or policy_models(store)
    owned = {r.player_id for r in state.lots}
    packets = {p: q.raw for p, q in players.items()}
    previous: list[dict] = []
    points = {
        p: {"baseline": means.get(p, 0), "tree": means.get(p, 0), "network": means.get(p, 0)} for p in players
    }
    ids = sorted(players)
    x = np.array(
        [
            choice_features(
                packets[p],
                p,
                owned,
                previous,
                [],
                state.bank,
                state.free_transfers,
                state.event,
                packets,
                points,
            )
            for p in ids
        ]
    )
    preference = {p: 0.0 for p in ids}
    for name in ("buy", "sell"):
        if not spec["metrics"][name]["promoted"]:
            continue
        probs = bundle[name].predict(x, num_threads=1)
        scores = logit(np.clip(probs, 1e-6, 1 - 1e-6)) / spec["metrics"][name]["temperature"]
        for i, p in enumerate(ids):
            # Retaining a likely sale has the opposite preference to purchasing it.
            if (name == "buy" and p not in owned) or (name == "sell" and p in owned):
                preference[p] += (1 if name == "buy" else -1) * float(np.clip(scores[i], -8, 8))
    return preference, spec


def rollout_action(store, member, state, players, rules, means, history, branch, step, loaded):
    """Learned action type, legal chip state, conditional player choices and exact search."""
    from .strategy import allowed_chips, solve_action

    spec, bundle = loaded
    from .manager_behavior import behavior_rows

    record = bundle.get("__record")
    if record is None:
        selections = {}
        for event in sorted(r["event"] for r in history["current"])[-2:]:
            try:
                selections[str(event)] = store.latest(f"picks_{member['entry']}_{event}")[0]
            except ValueError:
                pass
        try:
            transfers = store.latest(f"transfers_{member['entry']}")[0]
        except ValueError:
            transfers = []
        record = {"entry": member["entry"], "history": history, "picks": selections, "transfers": transfers}
        bundle["__record"] = record
    packets = {(state.event, p): player.raw for p, player in players.items()}
    future = next(r for r in behavior_rows({"members": [record]}, packets) if r["label"] is None)
    x = future["x"]
    x[0] = state.event
    x[1] = state.free_transfers
    x[2] = state.bank / 10
    baseprob = np.zeros(4)
    behavior = store.get("manager_behavior")
    if not behavior["promoted_over_frequency"]:
        baseprob[:] = behavior["aggregate_probabilities"]
    else:
        for name, weight in (("basic", 1 - behavior["rich_weight"]), ("rich", behavior["rich_weight"])):
            key = "__count_" + name
            if key not in bundle:
                config = behavior["models"][name]
                path = store.artifact_path(config["path"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != config["hash"]:
                    raise ValueError("Manager count checksum mismatch")
                bundle[key] = lgb.Booster(model_file=str(path))
            config = behavior["models"][name]
            values = np.asarray(bundle[key].predict(np.array([x[: config["width"]]]), num_threads=1))[0]
            baseprob[np.array(config["classes"])] += weight * values
    rng = np.random.default_rng([151, member["entry"], branch, step])
    if member.get("population_count_prior") is not None:
        baseprob = np.array(member["population_count_prior"])
    count = int(rng.choice(4, p=baseprob / baseprob.sum()))
    if count == 3:
        # Represent 3+ with the observed ordinary-transfer tail, excluding chip churn.
        if "__tail" not in bundle:
            bundle["__tail"] = spec.get("ordinary_transfer_tail", [3])
        count = min(int(rng.choice(bundle["__tail"])), rules.transfer_cap - state.transfers_made)
    masks = allowed_chips(state, rules)
    chipfrequency = np.array(spec["metrics"]["chip"]["frequency"])
    probs = chipfrequency.copy()
    if spec["metrics"]["chip"]["promoted"]:
        raw = np.asarray(bundle["chip"].predict(np.array([x]), num_threads=1)).reshape(-1)
        conditional = np.zeros(5)
        conditional[np.array(spec["metrics"]["chip"]["classes"])] = raw
        w = spec["metrics"]["chip"]["weight"]
        probs = w * conditional + (1 - w) * chipfrequency
    eligible = np.array([(None if c == "none" else c) in masks for c in CHIPS])
    probs *= eligible
    probs /= probs.sum()
    label = CHIPS[int(rng.choice(5, p=probs))]
    chip = None if label == "none" else label
    preference, _ = action_preferences(store, state, players, means, loaded)
    # Choice-model log odds express manager preferences, never our own player xPts.
    count_constraint_relaxed = False
    try:
        d = solve_action(
            state,
            players,
            rules,
            means,
            chip=chip,
            max_transfers=count,
            min_transfers=count if chip not in ("freehit", "wildcard") else 0,
            transfer_bias={p: 0.35 * v for p, v in preference.items()},
            seconds=0.3,
        )
    except ValueError:
        try:
            d = solve_action(
                state,
                players,
                rules,
                means,
                chip=chip,
                max_transfers=count,
                min_transfers=count if chip not in ("freehit", "wildcard") else 0,
                transfer_bias={p: 0.35 * v for p, v in preference.items()},
                seconds=2,
            )
        except ValueError:
            d = solve_action(
                state,
                players,
                rules,
                means,
                chip=chip,
                max_transfers=count,
                transfer_bias={p: 0.35 * v for p, v in preference.items()},
                seconds=1,
            )
            count_constraint_relaxed = True
    d.ledger["behavior"] = {
        "manager_policy_id": spec["id"],
        "transfer_count_probabilities": baseprob.tolist(),
        "chip_probabilities": probs.tolist(),
        "sampled_count": count,
        "count_constraint_relaxed": count_constraint_relaxed,
        "actual_transfers": len(d.action.buy),
        "sampled_chip": label,
        "player_preferences": True,
        "solver": d.solver,
        "information": "Released history plus this branch’s earlier actions; future private moves unknown",
    }
    record["history"]["current"].append(
        {
            "event": state.event,
            "event_transfers": len(d.action.buy),
            "event_transfers_cost": d.ledger["event_total_hit"],
            "points": d.utility + d.ledger["hit"],
            "bank": d.state.bank,
            "value": sum(players[lot.player_id].price for lot in d.state.lots),
        }
    )
    if d.action.chip:
        record["history"]["chips"].append({"event": state.event, "name": d.action.chip})
    line = d.action.lineup
    assert line is not None
    record["picks"][str(state.event)] = {
        "picks": [
            {"element": p, "position": i + 1, "is_captain": p == line.captain}
            for i, p in enumerate(line.xi + line.bench)
        ]
    }
    record["transfers"].extend(
        {
            "event": state.event,
            "element_out": old,
            "element_in": new,
            "element_out_cost": players[old].price,
            "element_in_cost": players[new].price,
        }
        for old, new in zip(d.action.sell, d.action.buy)
    )
    return d
