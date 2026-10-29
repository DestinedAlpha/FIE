"""Shared multi-source tensors, with a deadline boundary on every historical row."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from .context_archive import load_context
from .decision import NAMES, canonical_rows, player_key
from .elite import elite_features
from .features import number
from .storage import Store, atomic_json, now
from .workload import time_value, workload_features

LENGTH = 12
SEQUENCE_FIELDS = (
    ("minutes", 90),
    ("starts", 1),
    ("expected_goals", 1),
    ("expected_assists", 1),
    ("goals_scored", 1),
    ("assists", 1),
    ("total_points", 10),
    ("saves", 5),
    ("clearances_blocks_interceptions", 15),
    ("tackles", 8),
    ("recoveries", 15),
    ("bps", 40),
    ("bonus", 3),
    ("yellow_cards", 1),
    ("red_cards", 1),
    ("goals_conceded", 4),
)
TOKEN_COUNT = 4096
TEXT_LENGTH = 40


def tokens(text):
    words = re.findall(r"[a-z]+|\d+", text.casefold())[:TEXT_LENGTH]
    result = np.zeros(TEXT_LENGTH, dtype=np.int64)
    for i, word in enumerate(words):
        result[i] = 2 + int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "little") % (
            TOKEN_COUNT - 2
        )
    return result


def sequence(past):
    values = np.zeros((LENGTH, len(SEQUENCE_FIELDS) * 2), dtype=np.float32)
    mask = np.zeros(LENGTH, dtype=bool)
    for i, row in enumerate(past[-LENGTH:], LENGTH - min(LENGTH, len(past))):
        mask[i] = True
        for j, (name, scale) in enumerate(SEQUENCE_FIELDS):
            observed = row.get(name) not in (None, "")
            values[i, j] = np.clip(number(row, name) / scale, -5, 10) if observed else 0
            values[i, j + len(SEQUENCE_FIELDS)] = observed
    return values, mask


def source_features(context, expert, workload, code, deadline):
    available = context is not None
    context = context or {}
    status = context.get("status", "")
    chance = context.get("chance_of_playing_next_round")
    news = context.get("news") or ""
    penalty = context.get("penalties_order")
    text = tokens(news)
    flags = [float(status == s) for s in ("a", "d", "i", "s", "u")]
    flags += [
        float(chance is not None),
        float(chance) / 100 if chance is not None else 0,
        float(penalty == 1),
        float(penalty == 2),
        float(context.get("direct_freekicks_order") == 1),
        float(context.get("corners_and_indirect_freekicks_order") == 1),
        float(bool(news)),
        float(available),
    ]
    market = [
        number(context, "selected_by_percent") / 100,
        np.log1p(max(0, number(context, "transfers_in_event"))) / 15,
        np.log1p(max(0, number(context, "transfers_out_event"))) / 15,
        float(expert.get("ownership", 0)),
        float(expert.get("intended_xi", 0)),
        float(expert.get("captain", 0)),
        min(1, expert.get("sample", 0) / 100),
        float(bool(expert)),
        number(context, "now_cost") / 100,
    ]
    load = workload_features(workload, code, deadline) if workload else [0.0] * 7
    quality = [1.0, float(available), float(available or expert), float(load[-1] > 0)]
    return (
        text,
        np.array(flags, dtype=np.float32),
        np.array(market, dtype=np.float32),
        np.array(load, dtype=np.float32),
        np.array(quality, dtype=np.float32),
    )


def tree_matrix(tab, news, market, workload, text, seq):
    # Same source information for the boosted-tree challenger, including text
    # hash counts and the complete lagged numerical sequence.
    bag = np.zeros((len(text), 64), dtype=np.float32)
    for i, row in enumerate(text):
        for word in row[row > 0]:
            bag[i, int(word) % 64] += 1 / TEXT_LENGTH
    return np.concatenate(
        [tab, news, market, workload, bag, seq[:, :, : len(SEQUENCE_FIELDS)].reshape(len(tab), -1)], axis=1
    )


def build_fusion_data(store: Store):
    base = store.get("decision_features")
    frame = pl.read_parquet(store.artifact_path(base["path"]))
    originals, _meta, _quality = canonical_rows(store)
    mapping = {(r["season"], int(r["element"]), int(r["fixture"])): r for r in originals}
    context = load_context(store)
    experts, elite = elite_features(store)
    expert_samples = {}
    for (season, event, _), values in experts.items():
        expert_samples[season, event] = values["sample"]
    try:
        work = store.get("workload")["records"]
    except ValueError:
        work = []
    work_by_code = defaultdict(list)
    for row in work:
        work_by_code[row["player_code"]].append(row)
    history: dict[Any, list[dict]] = defaultdict(list)
    groups = defaultdict(list)
    for row in frame.to_dicts():
        groups[row["deadline_group"]].append(row)
    arrays: dict[str, list] = {
        key: []
        for key in (
            "tab",
            "sequence",
            "sequence_mask",
            "text",
            "news",
            "market",
            "workload",
            "quality",
            "target",
            "base_target",
            "minutes",
            "start",
            "start_mask",
            "dc",
            "dc_prior",
            "aux",
        )
    }
    records = []
    used_context = set()
    missing = 0
    archive = store.get("context_archive")
    deadlines = {(r["season"], r["event"]): r["deadline"] for r in archive["entries"]}
    for group, rows in sorted(groups.items()):
        for row in rows:
            original = mapping[row["season"], row["player_id"], row["fixture_id"]]
            code = original.get("code")
            key = player_key(original)
            packet = context.get((row["season"], row["event"], code))
            if packet is None:
                missing += 1
                continue
            used_context.add((row["season"], row["event"]))
            seq, mask = sequence(history[key])
            sample = expert_samples.get((row["season"], row["event"]))
            expert = experts.get(
                (row["season"], row["event"], row["player_id"]), {"sample": sample} if sample else {}
            )
            txt, news, market, load, quality = source_features(
                packet, expert, work_by_code.get(code, []), code, deadlines[row["season"], row["event"]]
            )
            quality[0] = bool(mask.any())
            arrays["tab"].append([row[k] for k in NAMES])
            arrays["sequence"].append(seq)
            arrays["sequence_mask"].append(mask)
            for name, value in [
                ("text", txt),
                ("news", news),
                ("market", market),
                ("workload", load),
                ("quality", quality),
            ]:
                arrays[name].append(value)
            for name, value in [
                ("target", row["target"]),
                ("base_target", row["base_target"]),
                ("minutes", row["minutes_target"]),
                ("start", row["start_target"] or 0),
                ("start_mask", row["start_target"] is not None),
                ("dc", row["dc_target"] if row["dc_target"] is not None else -1),
                ("dc_prior", row["dc_probability"]),
            ]:
                arrays[name].append(value)
            aux = []
            for name in (
                "goals_scored",
                "assists",
                "saves",
                "bonus",
                "clean_sheets",
                "yellow_cards",
                "red_cards",
            ):
                aux.append(number(original, name) if original.get(name) not in (None, "") else -1)
            arrays["aux"].append(aux)
            records.append(
                {
                    k: row[k]
                    for k in (
                        "season",
                        "event",
                        "deadline_group",
                        "player_id",
                        "fixture_id",
                        "position",
                        "team",
                        "baseline",
                        "prior_start",
                    )
                    if k in row
                }
                | {
                    "prior_start": row["p_start"],
                    "price_lag": int(packet["now_cost"]),
                    "price_observed": True,
                    "code": code,
                    "deadline": deadlines[row["season"], row["event"]],
                }
            )
        # Both matches of a double share identical pre-deadline histories.
        for row in rows:
            original = mapping[row["season"], row["player_id"], row["fixture_id"]]
            key = player_key(original)
            history[key].append(original)
            history[key] = history[key][-LENGTH:]
    directory = store.root / "features"
    directory.mkdir(exist_ok=True)
    converted = {
        name: np.asarray(
            value,
            dtype=(
                np.int64
                if name == "text"
                else bool
                if name in ("sequence_mask", "start_mask")
                else np.float32
            ),
        )
        for name, value in arrays.items()
    }
    converted["sequence"] = converted["sequence"].astype(np.float16)
    path = directory / "fusion-inputs.npz"
    np.savez_compressed(path, **cast(Any, converted))
    metadata = directory / "fusion-metadata.parquet"
    pl.DataFrame(records, infer_schema_length=None).write_parquet(metadata)
    input_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_hash = hashlib.sha256(metadata.read_bytes()).hexdigest()
    frozen_inputs = directory / ("fusion-" + input_hash[:24] + ".npz")
    frozen_metadata = directory / ("fusion-" + metadata_hash[:24] + ".parquet")
    path.replace(frozen_inputs)
    metadata.replace(frozen_metadata)
    result = {
        "at": now(),
        "version": "fusion-inputs-v1",
        "rows": len(records),
        "excluded_not_registered_before_deadline": missing,
        "numerical_features": NAMES,
        "sequence_length": LENGTH,
        "sequence_fields": SEQUENCE_FIELDS,
        "token_buckets": TOKEN_COUNT,
        "text_length": TEXT_LENGTH,
        "sources": {
            "news_deadlines": len(used_context),
            "verified_prior_managers": elite["retrieved_managers"],
            "manager_gameweeks": elite["retrieved_manager_gameweeks"],
        },
        "inputs_path": str(frozen_inputs),
        "metadata_path": str(frozen_metadata),
        "hash": input_hash,
        "metadata_hash": metadata_hash,
        "lineage": {
            "context_archive": archive["id"],
            "elite_cohort": elite["id"],
            "workload": store.get("workload")["id"],
        },
        "coverage": {
            "news": float(converted["quality"][:, 1].mean()),
            "expert_lag": float((converted["market"][:, 7] > 0).mean()),
            "workload": float(converted["quality"][:, 3].mean()),
        },
        "rules": "Archived news has pre-deadline capture proof; expert choices lag one deadline; workload uses retrospective match facts and a four-hour completion cutoff, not original publication proof",
        "missing_policy": "Explicit masks; unavailable workload and expert history are never treated as observed zero",
    }
    result["id"] = store.put("fusion_features", result)
    atomic_json(Path("artifacts/evaluations/fusion_data_coverage.json"), result)
    return converted, records, result


def live_inputs(store, rows, market, players):
    """Construct inference tensors using the same transforms as historical fitting."""
    originals, _, _ = canonical_rows(store)
    history = defaultdict(list)
    for original in originals:
        history[player_key(original)].append(original)
    experts, _ = elite_features(store)
    samples = {(season, event): values["sample"] for (season, event, _), values in experts.items()}
    try:
        workload = store.get("workload")["records"]
    except ValueError:
        workload = []
    work_by_code = defaultdict(list)
    for record in workload:
        # An upstream future fixture erroneously marked finished cannot become a
        # known observation merely because we forecast a distant gameweek.
        if time_value(record["known_after"]) < time_value(now()):
            work_by_code[record["player_code"]].append(record)
    deadlines = {e["id"]: e["deadline_time"] for e in market["events"]}
    next_event = next((e["id"] for e in market["events"] if e.get("is_next")), 38)
    latest_expert = max(
        (event for season, event, _ in experts if season == market["season"] and event <= next_event),
        default=0,
    )
    arrays: dict[str, list] = {
        key: []
        for key in ("tab", "sequence", "sequence_mask", "text", "news", "market", "workload", "quality")
    }
    for row in rows:
        player = players[row["player_id"]]
        seq, mask = sequence(history[("code", player.code)])
        context = dict(player.raw)
        # Current published choices may be carried forward as dated context;
        # unreleased future choices are never invented.
        sample = samples.get((market["season"], latest_expert))
        expert = experts.get(
            (market["season"], latest_expert, player.id), {"sample": sample} if sample else {}
        )
        txt, news, market_values, load, quality = source_features(
            context, expert, work_by_code.get(player.code, []), player.code, deadlines[row["event"]]
        )
        quality[0] = mask.any()
        arrays["tab"].append(row["x"])
        arrays["sequence"].append(seq)
        arrays["sequence_mask"].append(mask)
        for key, value in [
            ("text", txt),
            ("news", news),
            ("market", market_values),
            ("workload", load),
            ("quality", quality),
        ]:
            arrays[key].append(value)
    return {
        key: np.asarray(
            value, dtype=np.int64 if key == "text" else bool if key == "sequence_mask" else np.float32
        )
        for key, value in arrays.items()
    }
