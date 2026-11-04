"""Content-addressed, batched rival policy outcomes on a frozen football sample."""

from __future__ import annotations

import hashlib
import inspect

import numpy as np

from .storage import digest, now


def rival_outcomes(store, rivals, league, forecast, end_event, styles):
    from .manager_policy import action_preferences, rollout_action
    from .rivals import rival_policy
    from .simulation import score_plans

    lineage = {
        "forecast": forecast["id"],
        "league": league.get("id"),
        "rivals": rivals,
        "end": end_event,
        "styles": styles,
        "manager_policy": store.get("manager_policy")["id"] if store.list("manager_policy", 1) else None,
        "fusion_model": store.get("fusion_model")["id"] if store.list("fusion_model", 1) else None,
        "behavior": store.get("manager_behavior")["id"] if store.list("manager_behavior", 1) else None,
        "functions": [inspect.getsource(f) for f in (rival_policy, rollout_action, action_preferences)],
    }
    key = digest(lineage)
    for item in store.list("rival_worlds", 100):
        record = store.get("rival_worlds", item["id"])
        if record["cache_key"] != key:
            continue
        path = store.artifact_path(record["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["hash"]:
            raise ValueError("Rival world checksum mismatch")
        with np.load(path, allow_pickle=False) as loaded:
            return {(int(k.split(":")[0]), k.split(":")[1]): loaded[k] for k in loaded.files}, record["id"]
    policies = []
    keys = []
    for member in rivals:
        for style in styles:
            policies.append(rival_policy(store, member, league, forecast, end_event, style, score=False))
            keys.append((member["entry"], style, member["points"]))
    arrays = score_plans(store, policies, forecast)
    results = {(entry, style): points + score for (entry, style, points), score in zip(keys, arrays)}
    path = store.root / "scenarios" / ("rivals-" + key + ".npz")
    np.savez_compressed(path, **{f"{entry}:{style}": values for (entry, style), values in results.items()})
    record = {
        "at": now(),
        "cache_key": key,
        "lineage": lineage,
        "path": str(path),
        "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    record["id"] = store.put("rival_worlds", record)
    return results, record["id"]
