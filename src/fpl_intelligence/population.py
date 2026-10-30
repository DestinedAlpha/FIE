"""Rank-stratified public population samples and shared-world rank scenarios."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import numpy as np
from threadpoolctl import threadpool_limits

from .ingestion import Official, load_market
from .rivals import rival_policy
from .simulation import score_plan
from .storage import atomic_json, digest, now


async def collect_population(store, per_stratum=16):
    if not 4 <= per_stratum <= 64:
        raise ValueError("Population sample per stratum must be 4–64")
    _, market = load_market(store)
    bootstrap = store.latest("bootstrap")[0]
    total = int(bootstrap["total_players"])
    boundaries = sorted(
        {
            0,
            min(1000, total),
            min(10000, total),
            min(100000, total),
            min(500000, total),
            min(1000000, total),
            min(3000000, total),
            total,
        }
    )
    event = max(e["id"] for e in market["events"] if e.get("data_checked"))
    rng = np.random.default_rng(181)
    targets = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        edges = np.linspace(lo + 1, hi + 1, per_stratum + 1)
        for a, b in zip(edges, edges[1:]):
            targets.append((int(rng.integers(int(a), max(int(a) + 1, int(b)))), lo, hi))
    official = Official(store)
    records = []
    misses = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        pages = {
            page: await official.fetch(
                client,
                f"leagues-classic/314/standings/?page_standings={page}",
                f"population_standings_{page}",
            )
            for page in sorted({(r - 1) // 50 + 1 for r, _, _ in targets})
        }
        members = []
        for target, lo, hi in targets:
            page = pages[(target - 1) // 50 + 1]
            if not page or page["league"]["name"].casefold() != "overall":
                misses.append({"rank": target, "reason": "Overall standings page unavailable"})
                continue
            rows = page["standings"]["results"]
            offset = (target - 1) % 50
            if offset >= len(rows):
                misses.append({"rank": target, "reason": "Rank outside actual returned standings"})
                continue
            members.append({**rows[offset], "stratum": [lo + 1, hi], "target_rank": target})

        async def one(member):
            entry = member["entry"]

            async def fetch(path, source):
                try:
                    body, meta = store.latest(source)
                    if meta["captured_at"] >= market["known_at"]:
                        return body
                except ValueError:
                    pass
                return await official.fetch(client, path, source)

            values = await asyncio.gather(
                fetch(f"entry/{entry}/history/", f"entry_history_{entry}"),
                fetch(f"entry/{entry}/transfers/", f"transfers_{entry}"),
                *(
                    fetch(f"entry/{entry}/event/{g}/picks/", f"picks_{entry}_{g}")
                    for g in range(max(1, event - 1), event + 1)
                ),
            )
            if any(v is None for v in values):
                misses.append({"entry": entry, "reason": "Incomplete history/transfers/picks"})
                return
            records.append(
                {
                    "entry": entry,
                    "name": member.get("player_name", ""),
                    "team": member["entry_name"],
                    "points": member["total"],
                    "rank": member["rank"],
                    "stratum": member["stratum"],
                    "target_rank": member["target_rank"],
                }
            )

        await asyncio.gather(*(one(m) for m in members))
    if len({r["entry"] for r in records}) != len(records):
        raise ValueError("Duplicate population entries require a fresh unique rank sample")
    result = {
        "at": now(),
        "market_id": market["id"],
        "season": market["season"],
        "released_event": event,
        "total_players": total,
        "per_stratum": per_stratum,
        "members": records,
        "misses": misses,
        "retrieved": len(records),
        "requested": len(targets),
        "selection": "Seeded uniform rank targets within equal-width substrata of the actual overall standings",
        "frame": "Registered current overall population, including inactive entries; no top-cohort claim of representativeness",
        "errors": official.errors,
    }
    result["id"] = store.put("population_sample", result)
    atomic_json(
        Path("artifacts/evaluations/population_coverage.json"),
        {k: v for k, v in result.items() if k not in ("members", "errors")},
    )
    return result


def evaluate_population(store, plan_id=None):
    sample = store.get("population_sample")
    plan = store.get("plan", plan_id or store.get("selected_plan")["plan_id"])
    forecast = store.get("forecast", plan["forecast_id"])
    state = store.get("state", plan["state_id"])
    if sample["market_id"] != plan["market_id"]:
        raise ValueError("Population sample and forecast market differ; refresh sample")
    own = state["points"] + score_plan(store, plan, forecast)
    league = {"released_event": sample["released_event"], "captured_at": sample["at"]}
    activity = fit_population_activity(store)
    for member in sample["members"]:
        member["population_count_prior"] = activity["predictions"][str(member["entry"])]
    results = {}
    weights: list[float] = []
    strata = []
    for member in sample["members"]:
        key = tuple(member["stratum"])
        n = sum(tuple(m["stratum"]) == key for m in sample["members"])
        weights.append((key[1] - key[0] + 1) / n)
        strata.append(key)
    sample_weights = np.array(weights)
    partial = abs(sample_weights.sum() - sample["total_players"]) > 1
    if partial:
        raise ValueError(
            "Population sample is missing an entire rank stratum; refresh before extrapolating rank"
        )
    import hashlib

    from . import manager_policy, rivals

    cache_key = digest(
        {
            "sample": sample["id"],
            "forecast": forecast["id"],
            "manager_policy": store.get("manager_policy")["id"],
            "behavior": store.get("manager_behavior")["id"],
            "activity": activity["predictions"],
            "end": plan["end_event"],
            "code": [
                hashlib.sha256(Path(str(module.__file__)).read_bytes()).hexdigest()
                for module in (manager_policy, rivals)
            ],
        }
    )
    cached = {}
    for item in store.list("population_worlds", 30):
        record = store.get("population_worlds", item["id"])
        if record["cache_key"] == cache_key:
            path = store.artifact_path(record["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["hash"]:
                raise ValueError("Population worlds checksum mismatch")
            with np.load(path) as loaded:
                cached = {k: loaded[k] for k in loaded.files}
            break
    for style in ("hold", "learned0"):
        outcomes = []
        if style in cached:
            outcomes = cached[style]
            print("Reused frozen population policy worlds", style, flush=True)
        for i, member in enumerate(sample["members"] if style not in cached else []):
            policy = rival_policy(store, member, league, forecast, plan["end_event"], style)
            outcomes.append(member["points"] + score_plan(store, policy, forecast))
            if (i + 1) % 16 == 0:
                print("Population policy", style, i + 1, "/", sample["retrieved"], flush=True)
        points = np.asarray(outcomes)
        cached[style] = points
        strict = (points > own[None, :]).astype(float)
        ties = (points == own[None, :]).astype(float)
        rank = 1 + sample_weights @ strict
        tie_rank = rank + sample_weights @ ties
        threshold = 0.01 * sample["total_players"]
        probability = float((tie_rank <= threshold).mean())
        rng = np.random.default_rng(181)
        bootstrap = []
        with threadpool_limits(limits=1):
            for _ in range(300):
                w = np.zeros(len(sample_weights))
                for key in set(strata):
                    indices = np.array([i for i, k in enumerate(strata) if k == key])
                    chosen = rng.choice(indices, len(indices), replace=True)
                    for i in chosen:
                        w[i] += sample_weights[i]
                bootstrap.append(float(((1 + w @ (strict + ties)) <= threshold).mean()))
        results[style] = {
            "top_one_percent_probability": probability,
            "entry_resampling_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "rank_quantiles": np.quantile(rank, [0.1, 0.5, 0.9]).tolist(),
            "rank_with_ties_quantiles": np.quantile(tie_rank, [0.1, 0.5, 0.9]).tolist(),
            "weighted_population": float(sample_weights.sum()),
        }
    cache_path = store.root / "scenarios" / ("population-" + cache_key + ".npz")
    if not cache_path.exists():
        np.savez_compressed(cache_path, **cached)
    worlds_id = store.put(
        "population_worlds",
        {
            "cache_key": cache_key,
            "path": str(cache_path),
            "hash": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        },
    )
    report = {
        "at": now(),
        "population_worlds_id": worlds_id,
        "sample_id": sample["id"],
        "plan_id": plan["id"],
        "forecast_id": forecast["id"],
        "members": sample["retrieved"],
        "worlds": forecast["worlds"],
        "scenarios": results,
        "target_event": plan["end_event"],
        "complete_strata": not partial,
        "limits": [
            "Weighted rank scenarios under declared sampled policies, not a validated probability of a season finish",
            "The choice model learned on current top-1,000 managers is an explicit transfer assumption outside that cohort",
            "Entry bootstrap reflects sampled-entry variation; model, nonresponse, future-price and policy error are additional",
            "Overall points ties use a conservative rank range; official transfer tie-breaks require future realized actions",
        ],
    }
    report["id"] = store.put("population_forecast", report)
    return report


def fit_population_activity(store):
    """Calibrate rank-tier activity priors on earlier public decisions."""

    sample = store.get("population_sample")
    histories = {m["entry"]: store.latest(f"entry_history_{m['entry']}")[0] for m in sample["members"]}
    maximum = sample["released_event"]
    members = sample["members"]

    def probability(member, event, alpha):
        tier = member["stratum"]
        counts = np.ones(4)
        own = np.zeros(4)
        for m in members:
            if m["stratum"] != tier:
                continue
            for r in histories[m["entry"]]["current"]:
                if 1 < r["event"] < event and not any(
                    c["event"] == r["event"] and c["name"] in ("freehit", "wildcard")
                    for c in histories[m["entry"]]["chips"]
                ):
                    counts[min(3, int(r["event_transfers"]))] += 1
        counts /= counts.sum()
        for r in histories[member["entry"]]["current"]:
            if 1 < r["event"] < event:
                own[min(3, int(r["event_transfers"]))] += 1
        return (own + alpha * counts) / (own.sum() + alpha)

    def score(events, alpha):
        losses = []
        base = []
        for m in members:
            for event in events:
                row = next((r for r in histories[m["entry"]]["current"] if r["event"] == event), None)
                if row is None:
                    continue
                y = min(3, int(row["event_transfers"]))
                p = probability(m, event, alpha)
                losses.append(float(-np.log(p[y])))
                b = probability(m, event, 1e9)
                base.append(float(-np.log(b[y])))
        return float(np.mean(losses)), float(np.mean(base)), len(losses)

    alpha = min((1, 5, 10, 20), key=lambda a: score(range(maximum - 3, maximum - 1), a)[0])
    loss, baseline, n = score(range(maximum - 1, maximum + 1), alpha)
    predictions = {str(m["entry"]): probability(m, maximum + 1, alpha).tolist() for m in members}
    result = {
        "at": now(),
        "sample_id": sample["id"],
        "alpha": alpha,
        "log_loss": loss,
        "tier_baseline_log_loss": baseline,
        "test_decisions": n,
        "predictions": predictions,
        "protocol": "Shrink individual transfer history to its rank-stratum prior; choose strength on two earlier deadlines, test the following two",
        "limits": [
            "Current-rank stratification is retrospective; performance is conditional on this sample",
            "Inactive behavior can change; future policies remain scenarios",
        ],
    }
    result["id"] = store.put("population_activity", result)
    return result
