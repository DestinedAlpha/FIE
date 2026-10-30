"""Retrained source ablations and decision effects on identical deadline splits."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .decision import score_metrics
from .fusion import calibrate, corrected, fit_tree, load_data, tree_predict
from .fusion_data import tree_matrix
from .horizon import block_interval
from .storage import atomic_json, now


def evaluate_context(store):
    arrays, meta, features = load_data(store)
    x = tree_matrix(
        arrays["tab"],
        arrays["news"],
        arrays["market"],
        arrays["workload"],
        arrays["text"],
        arrays["sequence"],
    )
    history = np.column_stack([arrays["tab"], arrays["sequence"][:, :, :16].reshape(len(meta), -1)])
    official = x.copy()
    official[
        :,
        len(arrays["tab"][0]) + len(arrays["news"][0]) : len(arrays["tab"][0])
        + len(arrays["news"][0])
        + len(arrays["market"][0])
        + len(arrays["workload"][0]),
    ] = 0
    matrices = {"history_only": history, "official_news": official, "all_context": x}
    groups = np.array([r["deadline_group"] for r in meta])
    unique = sorted(set(groups))
    testing = [g for g in unique if g >= "2025/26-GW01"]
    records = []
    folds = []
    cache = store.root / "context-checkpoints" / features["id"][:20]
    cache.mkdir(parents=True, exist_ok=True)
    for first in range(0, len(testing), 8):
        block = testing[first : first + 8]
        checkpoint = cache / (str(first) + ".json")
        if checkpoint.exists():
            import json

            saved = json.loads(checkpoint.read_text())
            records.extend(saved["records"])
            folds.append(saved["fold"])
            print("Restored context ablation", block[-1], flush=True)
            continue
        begin = len(records)
        before = [g for g in unique if g < block[0]]
        cut = before[-4]
        train = np.flatnonzero(groups < cut)
        cal = np.flatnonzero((groups >= cut) & (groups < block[0]))
        test = np.flatnonzero(np.isin(groups, block))
        results = {}
        for kind, matrix in matrices.items():
            fitted = fit_tree(arrays, matrix, train, cal)
            curves, offset = calibrate(*tree_predict(fitted, matrix[cal]), arrays, cal)
            p, c, s = corrected(*tree_predict(fitted, matrix[test]), curves, offset)
            results[kind] = ((p * (c + 2 * arrays["dc_prior"][test])).sum(1), 1 - p[:, 0], p[:, 2:].sum(1), s)
        for j, i in enumerate(test):
            row = {
                "deadline_group": str(groups[i]),
                "actual": float(arrays["target"][i]),
                "played": float(arrays["minutes"][i] > 0),
                "long": float(arrays["minutes"][i] >= 60),
                "player_id": meta[i]["player_id"],
                "event": meta[i]["event"],
                "season": meta[i]["season"],
                "actual_minutes": float(arrays["minutes"][i]),
            }
            for kind, (mean, anyplay, long, start) in results.items():
                row.update(
                    {kind: float(mean[j]), kind + "_p_any": float(anyplay[j]), kind + "_p_60": float(long[j])}
                )
            records.append(row)
        folds.append({"test": block, "train_before": cut, "calibration_before": block[0]})
        atomic_json(checkpoint, {"records": records[begin:], "fold": folds[-1]})
        print("Retrained context ablation", block[-1], flush=True)
    metrics = {}
    captains: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    byweek = defaultdict(list)
    for r in records:
        byweek[r["deadline_group"]].append(r)
    for kind in matrices:
        y = np.array([r["actual"] for r in records])
        pred = np.array([r[kind] for r in records])
        metrics[kind] = {
            "points": score_metrics(y, pred),
            "appearance_brier": float(np.mean([(r["played"] - r[kind + "_p_any"]) ** 2 for r in records])),
            "sixty_brier": float(np.mean([(r["long"] - r[kind + "_p_60"]) ** 2 for r in records])),
            "paired_rmse_vs_history_95ci": block_interval(records, kind, "history_only"),
        }
        for group, rows in byweek.items():
            players: dict[int, dict[str, float]] = defaultdict(
                lambda: {"pred": 0.0, "actual": 0.0, "minutes": 0.0}
            )
            for r in rows:
                p = players[r["player_id"]]
                p["pred"] += r[kind]
                p["actual"] += r["actual"]
                p["minutes"] += r["actual_minutes"]
            choice = sorted(players, key=lambda p: -players[p]["pred"])[:2]
            captains[kind]["bonus"] += next(
                (players[p]["actual"] for p in choice if players[p]["minutes"] > 0), 0
            )
    result = {
        "at": now(),
        "feature_id": features["id"],
        "protocol": "Independently refit history-only, official-news, and full-context trees on identical expanding blocks with earlier calibration; no inference-mask substitution",
        "rows": len(records),
        "folds": folds,
        "metrics": metrics,
        "captain_bonus": {k: v["bonus"] for k, v in captains.items()},
        "source_use": "Official archived news/set pieces/availability, lagged qualified managers/market and other-competition workload enter measured feature branches",
        "limits": [
            "Source availability differs by season; source-on/off results do not establish causation",
            "Current-rank top-1,000 behavior is separately tested; historical expert signals use only prior-qualified managers",
            "Specialist article claims remain independently supported evidence/scenarios; missing past articles are not fabricated",
        ],
    }
    result["id"] = store.put("context_ablation", result)
    atomic_json(Path("artifacts/evaluations/context_ablation.json"), result)
    return result
