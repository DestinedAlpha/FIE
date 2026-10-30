"""Pre-deadline price/news transition learning and exact affordability bridges."""

from __future__ import annotations

import asyncio
import hashlib
import json
import lzma
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import lightgbm as lgb
import numpy as np
from scipy.special import softmax
from sklearn.metrics import log_loss

from .context_archive import FIELDS
from .ingestion import load_market
from .scenarios import sale_bounds
from .storage import atomic_json, digest, now

FEATURES = (
    "price",
    "ownership",
    "incoming",
    "outgoing",
    "net_flow",
    "position",
    "chance",
    "chance_observed",
    "injury",
    "suspension",
    "news",
    "week",
    "window_hours",
    "hours_to_deadline",
)


def market_features(p, event, duration=48, lead=48):
    incoming = float(p.get("transfers_in_event") or 0)
    outgoing = float(p.get("transfers_out_event") or 0)
    chance = p.get("chance_of_playing_next_round")
    return [
        float(p["now_cost"]) / 10,
        float(p.get("selected_by_percent") or 0) / 100,
        np.log1p(incoming),
        np.log1p(outgoing),
        (incoming - outgoing) / (incoming + outgoing + 1000),
        float(p["element_type"]),
        float(chance) / 100 if chance is not None else 1.0,
        float(chance is not None),
        float(p.get("status") in ("i", "d")),
        float(p.get("status") == "s"),
        float(bool(p.get("news"))),
        event / 38,
        duration / 48,
        lead / 48,
    ]


async def collect_market_windows(store):
    """Actual Git snapshots about 48h before each archived pre-deadline snapshot."""
    archive = store.get("context_archive")
    semaphore = asyncio.Semaphore(4)
    results = []
    errors = []
    async with httpx.AsyncClient(timeout=40, follow_redirects=False) as client:

        async def one(entry):
            source = f"market_window_{entry['season'].replace('/', '-')}_{entry['event']}"
            try:
                cached, meta = store.latest(source)
                if cached["end_hash"] == entry["hash"]:
                    results.append(
                        {
                            "source": source,
                            "hash": meta["hash"],
                            "season": entry["season"],
                            "event": entry["event"],
                        }
                    )
                    return
            except ValueError:
                pass
            async with semaphore:
                end, endmeta = store.latest(entry["source"], content_hash=entry["hash"])
                endtime = datetime.fromisoformat(end["source_captured_at"].replace("Z", "+00:00"))
                anchor = (endtime - timedelta(hours=48)).isoformat()
                cmd = [
                    "gh",
                    "api",
                    "--method",
                    "GET",
                    "repos/Randdalf/fplcache/commits",
                    "-f",
                    "until=" + anchor,
                    "-f",
                    "per_page=1",
                ]
                commits = json.loads(await asyncio.to_thread(subprocess.check_output, cmd, text=True))
                if not commits:
                    raise ValueError("No earlier archive commit")
                commit = commits[0]
                sha = commit["sha"]
                committed = commit["commit"]["committer"]["date"]
                detail = json.loads(
                    await asyncio.to_thread(
                        subprocess.check_output,
                        ["gh", "api", f"repos/Randdalf/fplcache/commits/{sha}"],
                        text=True,
                    )
                )
                paths = [
                    f["filename"]
                    for f in detail.get("files", [])
                    if f["filename"].startswith("cache/") and f["filename"].endswith(".json.xz")
                ]
                if len(paths) != 1:
                    raise ValueError("Earlier commit has no unique bootstrap snapshot")
                path = paths[0]
                parts = path.split("/")
                clock = parts[-1].split(".")[0].zfill(4)
                capture = datetime.fromisoformat(
                    f"{int(parts[1]):04d}-{int(parts[2]):02d}-{int(parts[3]):02d}T{clock[:2]}:{clock[2:]}:00+00:00"
                )
                if not capture <= datetime.fromisoformat(committed.replace("Z", "+00:00")) < endtime:
                    raise ValueError("Price-window clocks do not prove precedence")
                duration = (endtime - capture).total_seconds() / 3600
                if not 40 <= duration <= 72:
                    raise ValueError("Price-window duration outside declared support")
                url = f"https://raw.githubusercontent.com/Randdalf/fplcache/{sha}/{path}"
                response = await client.get(url)
                response.raise_for_status()
                if len(response.content) > 2_000_000:
                    raise ValueError("Price snapshot exceeds byte budget")
                decoder = lzma.LZMADecompressor(memlimit=128_000_000)
                raw = decoder.decompress(response.content, max_length=16_000_000)
                if not decoder.eof:
                    raise ValueError("Price snapshot exceeds decompression budget")
                bootstrap = json.loads(raw)
                elements = [{k: p.get(k) for k in FIELDS} for p in bootstrap["elements"]]
                body = {
                    "season": entry["season"],
                    "event": entry["event"],
                    "start": capture.isoformat(),
                    "end": endtime.isoformat(),
                    "deadline": entry["deadline"],
                    "commit": sha,
                    "committed_at": committed,
                    "url": url,
                    "compressed_hash": hashlib.sha256(response.content).hexdigest(),
                    "elements": elements,
                    "end_elements": end["elements"],
                    "end_hash": entry["hash"],
                    "duration_hours": duration,
                }
                meta = store.snapshot(source, url, body, capture.isoformat())
                results.append(
                    {
                        "source": source,
                        "hash": meta["hash"],
                        "season": entry["season"],
                        "event": entry["event"],
                    }
                )

        async def guarded(entry):
            try:
                await one(entry)
            except (
                ValueError,
                KeyError,
                httpx.HTTPError,
                subprocess.CalledProcessError,
                lzma.LZMAError,
            ) as exc:
                errors.append({"season": entry["season"], "event": entry["event"], "error": str(exc)[:180]})

        await asyncio.gather(*(guarded(e) for e in archive["entries"]))
    report = {
        "at": now(),
        "windows": sorted(results, key=lambda r: (r["season"], r["event"])),
        "retrieved": len(results),
        "requested": len(archive["entries"]),
        "errors": errors,
        "scope": "Observed roughly 48-hour pre-deadline price/news transitions, with both Git and source clocks preceding the target deadline",
    }
    report["id"] = store.put("market_windows", report)
    return report


def temperature_fit(prob, y):
    choices = (0.7, 1.0, 1.3, 1.7, 2.0)
    return min(
        choices,
        key=lambda t: log_loss(y, softmax(np.log(np.clip(prob, 1e-8, 1)) / t, axis=1), labels=[0, 1, 2]),
    )


def train_market(store):
    rows = []
    for entry in store.get("market_windows")["windows"]:
        body, _ = store.latest(entry["source"], content_hash=entry["hash"])
        ends = {int(p["code"]): p for p in body["end_elements"]}
        lead = (
            datetime.fromisoformat(body["deadline"].replace("Z", "+00:00"))
            - datetime.fromisoformat(body["start"])
        ).total_seconds() / 3600
        for p in body["elements"]:
            q = ends.get(int(p["code"]))
            if q is None:
                continue
            delta = int(q["now_cost"]) - int(p["now_cost"])
            unavailable = int(
                q.get("chance_of_playing_next_round") == 0 and p.get("chance_of_playing_next_round") != 0
            )
            rows.append(
                {
                    "group": f"{body['season']}-GW{body['event']:02d}",
                    "x": market_features(p, body["event"], body["duration_hours"], lead),
                    "y": int(np.sign(delta)) + 1,
                    "delta": delta,
                    "unavailable": unavailable,
                }
            )
    train = [r for r in rows if r["group"] < "2024/25-GW31"]
    cal = [r for r in rows if "2024/25-GW31" <= r["group"] < "2025/26-GW01"]
    test = [r for r in rows if r["group"] >= "2025/26-GW01"]
    if not all((train, cal, test)):
        raise ValueError("Price model requires three chronological partitions")
    model = lgb.LGBMClassifier(
        n_estimators=160,
        num_leaves=16,
        min_child_samples=120,
        reg_lambda=20,
        n_jobs=1,
        verbosity=-1,
        random_state=131,
    )
    model.fit([r["x"] for r in train], [r["y"] for r in train])
    t = temperature_fit(model.predict_proba([r["x"] for r in cal]), [r["y"] for r in cal])
    probability = softmax(np.log(np.clip(model.predict_proba([r["x"] for r in test]), 1e-8, 1)) / t, axis=1)
    freq = (np.bincount([r["y"] for r in train], minlength=3) + 1) / (len(train) + 3)
    y = np.array([r["y"] for r in test])
    nll = -np.log(probability[np.arange(len(y)), y])
    base = -np.log(freq[y])
    bygroup = defaultdict(list)
    for r, a, b in zip(test, nll, base):
        bygroup[r["group"]].append(float(a - b))
    differences = np.array([np.mean(v) for _, v in sorted(bygroup.items())])
    rng = np.random.default_rng(131)
    n = len(differences)
    starts = rng.integers(0, n, (2000, (n + 3) // 4))
    idx = ((starts[:, :, None] + np.arange(4)) % n).reshape(2000, -1)[:, :n]
    ci = np.quantile(differences[idx].mean(1), [0.025, 0.975]).tolist()
    news = lgb.LGBMClassifier(
        n_estimators=120,
        num_leaves=10,
        min_child_samples=150,
        reg_lambda=30,
        n_jobs=1,
        verbosity=-1,
        random_state=132,
    )
    news.fit([r["x"] for r in train], [r["unavailable"] for r in train])
    npred = np.asarray(news.predict_proba([r["x"] for r in test]))[:, 1]
    ny = np.array([r["unavailable"] for r in test])
    nprior = (sum(r["unavailable"] for r in train) + 1) / (len(train) + 2)
    news_brier = float(np.mean((npred - ny) ** 2))
    news_base = float(np.mean((nprior - ny) ** 2))
    directory = (
        store.root
        / "models"
        / ("market-" + digest({"windows": store.get("market_windows")["id"], "version": 1})[:20])
    )
    directory.mkdir(parents=True, exist_ok=True)
    specs = {}
    for name, m in (("price", model), ("news", news)):
        path = directory / (name + ".txt")
        m.booster_.save_model(str(path))
        specs[name] = {"path": str(path), "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {
        "at": now(),
        "features": FEATURES,
        "models": specs,
        "temperature": t,
        "classes": ["fall", "stable", "rise"],
        "train_rows": len(train),
        "calibration_rows": len(cal),
        "test_rows": len(test),
        "test_deadlines": n,
        "price_log_loss": float(nll.mean()),
        "frequency_log_loss": float(base.mean()),
        "block_log_loss_difference_95ci": ci,
        "promoted": bool(ci[1] < 0),
        "news_brier": news_brier,
        "news_frequency_brier": news_base,
        "news_promoted": news_brier < news_base,
        "news_prior": nprior,
        "frequency": freq.tolist(),
        "magnitudes": {str(k): [r["delta"] for r in train if r["y"] == k] for k in (0, 1, 2)},
        "window_hours": [40, 72],
        "protocol": "Train before 2024/25 GW31; calibrate final eight prior-season deadlines; test all later deadlines. Inputs and targets both precede each football deadline.",
        "source_id": store.get("market_windows")["id"],
        "limits": [
            "Conditional transition model, not access to the unpublished FPL price-change algorithm",
            "Approximately 48-hour archives; intraday horizons outside this support are scenario sensitivities",
            "Reported-unavailable transitions concern new official news, not certainty of actual zero minutes",
        ],
    }
    report["id"] = store.put("market_model", report)
    atomic_json(
        Path("artifacts/evaluations/market_model.json"),
        {k: v for k, v in report.items() if k != "magnitudes"},
    )
    return report


def market_predictions(store, players, event):
    spec = store.get("market_model")
    models = {}
    for name, item in spec["models"].items():
        path = store.artifact_path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["hash"]:
            raise ValueError("Market model checksum mismatch")
        models[name] = lgb.Booster(model_file=str(path))
    ids = sorted(players)
    x = np.array([market_features(players[p].raw, event) for p in ids])
    probs = softmax(
        np.log(np.clip(models["price"].predict(x, num_threads=1), 1e-8, 1)) / spec["temperature"], axis=1
    )
    if not spec["promoted"]:
        probs[:] = spec["frequency"]
    unavailable = models["news"].predict(x, num_threads=1)
    if not spec["news_promoted"]:
        unavailable[:] = spec["news_prior"]
    return {
        p: {"price_probabilities": probs[i].tolist(), "new_unavailable_probability": float(unavailable[i])}
        for i, p in enumerate(ids)
    }, spec


def price_bridge(store, plan_id=None, worlds=4000):
    players, market = load_market(store)
    state = store.get("state")
    plan = store.get("plan", plan_id or store.get("selected_plan")["plan_id"])
    if plan["market_id"] != market["id"] or plan["state_id"] != state["id"]:
        raise ValueError("Refresh stale plan before price analysis")
    pred, spec = market_predictions(store, players, state["event"])
    first = plan["ledger"][0]
    lots = {r["player_id"]: r for r in state["lots"]}
    from .domain import Lot

    rng = np.random.default_rng(131)
    remaining = np.full(worlds, state["bank"], dtype=float)
    optimistic = remaining.copy()
    rows = []
    for move, sign in ((first["sell"], 1), (first["buy"], -1)):
        for r in move:
            p = r["player_id"]
            classes = rng.choice(3, worlds, p=pred[p]["price_probabilities"])
            changes = np.zeros(worlds, dtype=int)
            for k in (0, 2):
                mask = classes == k
                pool = spec["magnitudes"][str(k)] or ([-1] if k == 0 else [1])
                changes[mask] = rng.choice(pool, mask.sum())
            future = np.maximum(0, players[p].price + changes)
            if sign > 0:
                lot = Lot.model_validate(lots[p])
                bounds = np.array([sale_bounds(lot, players[p].price, int(v)) for v in future])
                remaining += bounds[:, 0]
                optimistic += bounds[:, 1]
            else:
                remaining -= future
                optimistic -= future
            rows.append({"player_id": p, "direction": "sell" if sign > 0 else "buy", **pred[p]})
    deadline = next(e["deadline_time"] for e in market["events"] if e["id"] == state["event"])
    hours = (
        datetime.fromisoformat(deadline.replace("Z", "+00:00")) - datetime.fromisoformat(now())
    ).total_seconds() / 3600
    report = {
        "at": now(),
        "state_id": state["id"],
        "market_id": market["id"],
        "plan_id": plan["id"],
        "market_model_id": spec["id"],
        "window_hours": 48,
        "hours_to_deadline": hours,
        "within_trained_time_support": 40 <= hours <= 72,
        "affordable_probability_bounds": [float(np.mean(remaining >= 0)), float(np.mean(optimistic >= 0))],
        "bank_units_quantiles": np.quantile(remaining, [0.1, 0.5, 0.9]).tolist(),
        "moves": rows,
        "worlds": worlds,
        "assumptions": [
            "48-hour conditional price transitions; player price directions sampled independently",
            "Unknown acquisition lots use exact lawful sale bounds, not invented purchase history",
            "A longer/shorter wait is an explicit timing sensitivity, not separately calibrated intraday probability",
        ],
    }
    report["id"] = store.put("price_bridge", report)
    return report
