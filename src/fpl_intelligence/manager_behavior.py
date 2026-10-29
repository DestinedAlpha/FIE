"""Released top-1,000 selections, transfer behavior and feasible action branches."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss

from .ingestion import Official, load_market
from .storage import Store, atomic_json, now


async def collect_top_managers(store: Store, limit=1000):
    if not 1 <= limit <= 1000:
        raise ValueError("Rank cohort budget is 1–1,000 actual entries")
    _, market = load_market(store)
    events = [e["id"] for e in market["events"] if e.get("data_checked")]
    try:
        allow_history_cache = store.get("top_manager_history")["events"] == events
    except ValueError:
        allow_history_cache = False
    official = Official(store)
    members = []
    records = []
    misses = []
    season_start = min(e["deadline_time"] for e in market["events"])
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        for page in range(1, 1 + (limit + 49) // 50):
            body = await official.fetch(
                client, f"leagues-classic/314/standings/?page_standings={page}", f"topk_standings_{page}"
            )
            if not body:
                break
            if body["league"]["name"].casefold() != "overall":
                raise ValueError("League 314 is not the current overall competition")
            members.extend(body["standings"]["results"])
            if not body["standings"]["has_next"]:
                break
        members = list({r["entry"]: r for r in members}.values())[:limit]

        async def cached(path, source, aliases=()):
            for name in (source, *aliases):
                try:
                    value, meta = store.latest(name)
                    captured = datetime.fromisoformat(meta["captured_at"].replace("Z", "+00:00"))
                    if meta["captured_at"] >= season_start and (
                        "/event/" in path
                        or (
                            allow_history_cache and datetime.now(timezone.utc) - captured < timedelta(hours=6)
                        )
                    ):
                        return value
                except ValueError:
                    pass
            return await official.fetch(client, path, source)

        async def one(member):
            entry = member["entry"]
            history, transfers, *picks = await asyncio.gather(
                cached(
                    f"entry/{entry}/history/",
                    f"topk_history_{entry}",
                    (f"elite_history_{entry}", f"entry_history_{entry}"),
                ),
                cached(f"entry/{entry}/transfers/", f"topk_transfers_{entry}", (f"transfers_{entry}",)),
                *(
                    cached(
                        f"entry/{entry}/event/{event}/picks/",
                        f"topk_picks_{entry}_{event}",
                        (
                            f"elite_picks_{entry}_{event}",
                            f"cohort_picks_{entry}_{event}",
                            f"picks_{entry}_{event}",
                        ),
                    )
                    for event in events
                ),
            )
            released = {
                str(event): body
                for event, body in zip(events, picks)
                if body and len(body.get("picks", [])) == 15
            }
            if history:
                history = {
                    **history,
                    "current": [r for r in history["current"] if r["event"] in events],
                    "chips": [r for r in history["chips"] if r["event"] in events],
                }
            record = {
                "entry": entry,
                "rank_at_selection": member["rank"],
                "history": history,
                "transfers": [r for r in (transfers or []) if r.get("event") in events],
                "picks": released,
            }
            records.append(record)
            if len(released) != len(events) or history is None or transfers is None:
                misses.append(
                    {
                        "entry": entry,
                        "missing_events": sorted(set(events) - {int(e) for e in released}),
                        "history_missing": history is None,
                        "transfers_missing": transfers is None,
                    }
                )
            if len(records) % 100 == 0:
                print(f"Top manager histories retrieved: {len(records)}/{len(members)}", flush=True)

        await asyncio.gather(*(one(member) for member in members))
    result = {
        "at": now(),
        "season": market["season"],
        "selection": "First 1,000 entries in current official overall-rank order; ties and actual retrieval counts retained",
        "selection_event": max(events),
        "requested": limit,
        "retrieved_managers": len(records),
        "retrieved_manager_gameweeks": sum(len(r["picks"]) for r in records),
        "complete_managers": len(records) - len(misses),
        "events": events,
        "members": sorted(records, key=lambda r: (r["rank_at_selection"], r["entry"])),
        "misses": misses,
        "errors": official.errors,
        "temporal_track": "Current-ranked retrospective analysis; never presented as pre-season expertise or an unbiased past top-1,000 cohort",
        "feature_rule": "At target GW g use only released picks/history/transfers with event < g; current rank never enters numeric features",
        "future_transfers": "Unknown; model outputs conditional feasible branches, not private account actions",
    }
    result["id"] = store.put("top_manager_history", result)
    atomic_json(
        Path("artifacts/evaluations/top_manager_coverage.json"),
        {k: v for k, v in result.items() if k not in ("members", "misses", "errors")},
    )
    return result


def behavior_rows(report, packets=None):
    packets = packets or {}
    result = []
    for manager in report["members"]:
        history = manager.get("history") or {}
        weeks = {r["event"]: r for r in history.get("current", [])}
        chips = {r["event"]: r["name"] for r in history.get("chips", [])}
        ft = 1
        past: list[dict] = []
        for event in sorted(weeks):
            row = weeks[event]
            transfers = int(row.get("event_transfers", 0))
            chip = chips.get(event)
            if past:
                previous = weeks[event - 1] if event - 1 in weeks else past[-1]
                finances = weeks.get(event - 2, previous) if chips.get(event - 1) == "freehit" else previous
                picks = manager["picks"].get(str(event - 1), {}).get("picks", [])
                x = [
                    event,
                    ft,
                    finances.get("bank", 0) / 10,
                    finances.get("value", 1000) / 100,
                    previous.get("points", 0) / 100,
                    np.mean([r.get("event_transfers", 0) for r in past[-3:]]),
                    sum(r.get("event_transfers_cost", 0) for r in past) / 4,
                    float(chips.get(event - 1) == "freehit"),
                    len(chips.keys() & {r["event"] for r in past}),
                    sum(p.get("is_captain", False) for p in picks),
                ]
                result.append(
                    {
                        "entry": manager["entry"],
                        "event": event,
                        "x": x + decision_context(manager, event, packets),
                        "label": min(transfers, 3),
                        "chip": chip or "none",
                    }
                )
            if event == 1:
                ft = 1
            elif chip not in ("wildcard", "freehit"):
                ft = min(5, max(0, ft - transfers) + 1)
            past.append(row)
        if past:
            event = max(weeks) + 1
            previous = past[-1]
            finances = weeks.get(event - 2, previous) if chips.get(event - 1) == "freehit" else previous
            x = [
                event,
                ft,
                finances.get("bank", 0) / 10,
                finances.get("value", 1000) / 100,
                previous.get("points", 0) / 100,
                np.mean([r.get("event_transfers", 0) for r in past[-3:]]),
                sum(r.get("event_transfers_cost", 0) for r in past) / 4,
                float(chips.get(event - 1) == "freehit"),
                len(chips),
                1,
            ]
            result.append(
                {
                    "entry": manager["entry"],
                    "event": event,
                    "x": x + decision_context(manager, event, packets),
                    "label": None,
                    "chip": None,
                }
            )
    return result


def train_behavior(store: Store):
    report = store.get("top_manager_history")
    rows = behavior_rows(report, context_packets(store))
    observed = [r for r in rows if r["label"] is not None]
    tests = []

    def fit(training, width):
        x = np.array([r["x"][:width] for r in training])
        y = np.array([r["label"] for r in training])
        model = lgb.LGBMClassifier(
            n_estimators=90,
            num_leaves=10,
            min_child_samples=100,
            reg_lambda=25,
            learning_rate=0.04,
            random_state=61,
            n_jobs=1,
            verbosity=-1,
        ).fit(x, y)
        return model

    def predict(model, records, width):
        values = model.predict_proba(np.array([r["x"][:width] for r in records]))
        probability = np.zeros((len(records), 4))
        probability[:, model.classes_.astype(int)] = values
        return probability

    last_event = max(r["event"] for r in observed)
    if last_event < 8:
        raise ValueError("Eight completed gameweeks are required for the calibrated behavior comparison")
    for event in range(last_event - 3, last_event + 1):
        training = [r for r in observed if r["event"] < event]
        test = [r for r in observed if r["event"] == event]
        if not test:
            continue
        simple = predict(fit(training, 10), test, 10)
        rich = predict(fit(training, 23), test, 23)
        counts = np.bincount([r["label"] for r in training], minlength=4) + 1
        baseline = counts / counts.sum()
        for i, row in enumerate(test):
            tests.append(
                {
                    "event": event,
                    "actual": row["label"],
                    "basic": simple[i].tolist(),
                    "rich": rich[i].tolist(),
                    "baseline": baseline.tolist(),
                }
            )
    cal = [r for r in tests if r["event"] < last_event - 1]
    test = [r for r in tests if r["event"] >= last_event - 1]

    def loss(records, weight):
        y = [r["actual"] for r in records]
        prob = np.array([(1 - weight) * np.array(r["basic"]) + weight * np.array(r["rich"]) for r in records])
        return float(log_loss(y, prob, labels=[0, 1, 2, 3]))

    weights = (0, 0.25, 0.5, 0.75, 1)
    calibration = {str(w): loss(cal, w) for w in weights}
    weight = min(weights, key=lambda w: calibration[str(w)])
    y = np.array([r["actual"] for r in test])
    p = np.array([(1 - weight) * np.array(r["basic"]) + weight * np.array(r["rich"]) for r in test])
    b = np.array([r["baseline"] for r in test])
    metrics = {
        "rows": len(test),
        "events": [last_event - 1, last_event],
        "model_log_loss": float(log_loss(y, p, labels=[0, 1, 2, 3])),
        "frequency_log_loss": float(log_loss(y, b, labels=[0, 1, 2, 3])),
        "model_accuracy": float((p.argmax(1) == y).mean()),
        "frequency_accuracy": float((b.argmax(1) == y).mean()),
        "basic_log_loss": loss(test, 0),
        "rich_log_loss": loss(test, 1),
    }
    promoted = metrics["model_log_loss"] < metrics["frequency_log_loss"]
    future = [r for r in rows if r["label"] is None]
    models = {}
    probs = {}
    directory = store.root / "models" / "manager-behavior"
    directory.mkdir(parents=True, exist_ok=True)
    for name, width in (("basic", 10), ("rich", 23)):
        model = fit(observed, width)
        path = directory / (
            name
            + "-"
            + __import__("hashlib").sha256(model.booster_.model_to_string().encode()).hexdigest()[:24]
            + ".txt"
        )
        model.booster_.save_model(str(path))
        models[name] = {
            "path": str(path),
            "hash": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "width": width,
            "classes": model.classes_.astype(int).tolist(),
        }
        probs[name] = predict(model, future, width)
    pred = (1 - weight) * probs["basic"] + weight * probs["rich"]
    if not promoted:
        counts = np.bincount([r["label"] for r in observed], minlength=4) + 1
        pred[:] = counts / counts.sum()
    last = str(report["selection_event"])
    purchases: Counter[int] = Counter()
    sales: Counter[int] = Counter()
    ownership: Counter[int] = Counter()
    captains: Counter[int] = Counter()
    for manager in report["members"]:
        for transfer in manager.get("transfers", []):
            if transfer["event"] >= report["selection_event"] - 2:
                purchases[transfer["element_in"]] += 1
                sales[transfer["element_out"]] += 1
        for pick in manager["picks"].get(last, {}).get("picks", []):
            ownership[pick["element"]] += 1
            captains[pick["element"]] += int(pick["is_captain"])
    output = [
        {"entry": row["entry"], "event": row["event"], "probabilities": values.tolist()}
        for row, values in zip(future, pred)
    ]
    result = {
        "at": now(),
        "population_id": report["id"],
        "selection_event": report["selection_event"],
        "event": max(r["event"] for r in future),
        "classes": ["roll", "one", "two", "three_or_more"],
        "metrics": metrics,
        "protocol": (
            f"Expanding earlier-event fits; choose basic/richer mixture on GW{last_event - 3}–{last_event - 2}, "
            f"report GW{last_event - 1}–{last_event} separately; final refit uses all {last_event} released weeks"
        ),
        "calibration_loss_by_rich_weight": calibration,
        "rich_weight": weight,
        "promoted_over_frequency": promoted,
        "predictions": output,
        "aggregate_probabilities": pred.mean(0).tolist(),
        "models": models,
        "recent_purchases": [{"player_id": p, "managers": n} for p, n in purchases.most_common(20)],
        "recent_sales": [{"player_id": p, "managers": n} for p, n in sales.most_common(20)],
        "ownership": [
            {
                "player_id": p,
                "fraction": n / len(report["members"]),
                "captain_fraction": captains[p] / len(report["members"]),
            }
            for p, n in ownership.most_common(30)
        ],
        "input_groups": [
            "Lagged FT/bank/value/points and transfers/hits",
            "Released squads, injury/premium/club exposure, captain continuity and turnover",
            "Past incoming/outgoing prices, spending and rapid resales",
        ],
        "limits": [
            "Current top-1,000 selection uses present ranks; these are conditional retrospective behavior diagnostics, not unbiased elite-skill validation",
            "Only two later test deadlines; prospective transfer calibration remains unproven",
            "Transfer-count probabilities and feasible forecast-guided branches cannot reveal private future purchases",
            "Richer source features only affect probabilities to the extent selected on earlier calibration weeks",
        ],
    }
    result["id"] = store.put("manager_behavior", result)
    atomic_json(
        Path("artifacts/evaluations/manager_behavior.json"),
        {
            k: v
            for k, v in result.items()
            if k not in ("predictions", "ownership", "recent_purchases", "recent_sales")
        },
    )
    return result


def rival_count_probabilities(store, entry):
    import hashlib

    spec = store.get("manager_behavior")
    for row in spec["predictions"]:
        if row["entry"] == entry:
            return row["probabilities"]
    history, _ = store.latest(f"entry_history_{entry}")
    selections = {}
    for event in sorted(r["event"] for r in history["current"])[-2:]:
        for source in (f"picks_{entry}_{event}", f"topk_picks_{entry}_{event}"):
            try:
                selections[str(event)] = store.latest(source)[0]
                break
            except ValueError:
                pass
    try:
        transfers = store.latest(f"transfers_{entry}")[0]
    except ValueError:
        transfers = []
    rows = behavior_rows(
        {"members": [{"entry": entry, "history": history, "picks": selections, "transfers": transfers}]},
        context_packets(store),
    )
    future = next(r for r in rows if r["label"] is None)
    if not spec["promoted_over_frequency"]:
        return spec["aggregate_probabilities"]
    probability = np.zeros(4)
    for name, weight in (("basic", 1 - spec["rich_weight"]), ("rich", spec["rich_weight"])):
        config = spec["models"][name]
        path = store.artifact_path(config["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != config["hash"]:
            raise ValueError("Behavior model integrity failed")
        model = lgb.Booster(model_file=str(path))
        values = np.asarray(model.predict(np.array([future["x"][: config["width"]]]), num_threads=1))[0]
        probability[np.array(config["classes"])] += weight * values
    return probability.tolist()


def context_packets(store):
    from .context_archive import load_context

    _, market = load_market(store)
    packets = {
        (event, row["id"]): row
        for (season, event, _), row in load_context(store).items()
        if season == market["season"]
    }
    next_event = next((e["id"] for e in market["events"] if e.get("is_next")), 38)
    for row in market["players"]:
        packets[next_event, row["id"]] = row["raw"]
    return packets


def decision_context(manager, event, packets):
    restored = any(
        c["name"] == "freehit" and c["event"] == event - 1
        for c in (manager.get("history") or {}).get("chips", [])
    )
    roster_event = event - 2 if restored else event - 1
    selections = manager["picks"].get(str(roster_event), {}).get("picks", [])
    previous = manager["picks"].get(str(event - 2), {}).get("picks", [])
    current = {r["element"] for r in selections}
    past = {r["element"] for r in previous}
    context = [packets[event, p] for p in current if (event, p) in packets]
    transfers = [r for r in manager.get("transfers", []) if r["event"] < event]
    recent = [r for r in transfers if r["event"] >= event - 3]
    bought = {r["element_in"]: r["event"] for r in transfers}
    flips = sum(r["element_out"] in bought and 0 < r["event"] - bought[r["element_out"]] <= 3 for r in recent)
    clubs = Counter(r["team"] for r in context)
    captain = next((p["element"] for p in selections if p.get("is_captain")), None)
    oldcaptain = next((p["element"] for p in previous if p.get("is_captain")), None)
    return [
        sum(r.get("status") in ("d", "i", "s", "u") for r in context),
        sum(
            (100 if r.get("chance_of_playing_next_round") is None else r["chance_of_playing_next_round"]) < 51
            for r in context
        ),
        sum(r["now_cost"] >= 90 for r in context),
        max(clubs.values(), default=0),
        sum(r["now_cost"] for r in context) / 1000,
        len(current - past) if past else 0,
        float(captain == oldcaptain and captain is not None),
        flips,
        len(recent),
        sum(r["element_in_cost"] - r["element_out_cost"] for r in recent) / 100,
        sum(r["element_in_cost"] for r in recent) / (100 * max(1, len(recent))),
        len(context) / 15,
        float(bool(previous)),
    ]
