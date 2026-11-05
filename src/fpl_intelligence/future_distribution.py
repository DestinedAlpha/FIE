"""Marginal distribution acceptance using the actual future-mixture constructor."""

from __future__ import annotations

import hashlib
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .consensus import paired_interval, replay_packets
from .decision import canonical_rows, future_inputs, load_bundle, predict_models, sample_point_outcomes
from .future_expert import combine, frozen_spec
from .future_policy_replay import prepare
from .storage import Store, atomic_json, digest, now


def crps(draws, actual):
    ordered = np.sort(np.asarray(draws, dtype=float))
    n = len(ordered)
    return float(np.abs(ordered - actual).mean() - np.dot(ordered, 2 * np.arange(1, n + 1) - n - 1) / n**2)


def run_distribution_gate(parent, baseline_id=None):
    baseline, refinement = parent.get("pipeline_replay", baseline_id), parent.get("horizon_refinement")
    originals, _, _ = canonical_rows(parent)
    spec = frozen_spec(parent, refinement, "2025/26-GW01")
    protocol = {
        "version": "future-marginal-distributions-v1",
        "baseline_id": baseline["id"],
        "observations_hash": digest([r for r in originals if r["season"] <= "2025/26"]),
        "refinement_id": refinement["id"],
        "season": "2025/26",
        "events": list(range(20, 39)),
        "scope": "Actual reference weights, recovery/discipline inputs and future mixture; marginal single-fixture weeks only",
        "sampling": "At most 160 predeadline relevant and 160 other players per source deadline; deterministic seed 137",
        "seeds": [11, 29, 73],
        "worlds_per_seed": 512,
        "gate": "Upper paired CRPS-delta intervals below zero for all and relevant samples, nonworse appearance/60-minute Brier, relevant 80% coverage at least 75%",
        "limits": "Previously studied historical reconstruction; marginal acceptance does not establish joint rival or double-gameweek calibration",
    }
    protocol_id = parent.put("future_distribution_protocol", protocol)
    if parent.list("future_distribution_validation", 1):
        prior = parent.get("future_distribution_validation")
        if prior["protocol_id"] == protocol_id:
            return prior
    packets = replay_packets(parent)
    truth: dict = defaultdict(float)
    minutes: dict = defaultdict(int)
    completed: set[int] = set()
    for row in originals:
        if row["season"] == "2025/26":
            key = int(row["fixture"]), int(row["element"])
            truth[key] += float(row.get("total_points") or 0)
            minutes[key] += int(row.get("minutes") or 0)
            completed.add(key[0])
    archive_path = parent.root / "features" / ("future-distribution-" + digest(originals)[:20] + ".parquet")
    if not archive_path.exists():
        pl.DataFrame(originals, infer_schema_length=None, strict=False).write_parquet(archive_path)
    archive = {"path": archive_path, "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
    scratch = Path("artifacts/runtime/future-distribution")
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    for old in baseline["weeks"]:
        if old["policy"] != "primary_multiweek" or old["season"] != "2025/26":
            continue
        event = old["event"]
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            store = Store(Path(temporary) / "data")
            _, players, market, model = prepare(
                parent, store, old, packets["2025/26", event], originals, archive, spec
            )
            rows, scheduled = future_inputs(store, market, players, event, 38)
            counts: dict = defaultdict(set)
            for row in rows:
                counts[row["player_id"], row["event"]].add(row["fixture_id"])
            candidates = [
                r
                for r in rows
                if 0 < r["event"] - event <= 18
                and r["fixture_id"] in completed
                and len(counts[r["player_id"], r["event"]]) == 1
            ]
            generator = np.random.default_rng(137 + event)
            selected: list[dict] = []
            for relevant in (False, True):
                group = [
                    r
                    for r in candidates
                    if (players[r["player_id"]].price >= 60 or players[r["player_id"]].ownership >= 5)
                    == relevant
                ]
                if group:
                    selected.extend(
                        group[i] for i in generator.choice(len(group), min(160, len(group)), replace=False)
                    )
            if not selected:
                continue
            bundle = load_bundle(store, model)
            _, rp, rc = predict_models(
                bundle, np.asarray([r["x"] for r in selected]), [r["dc_probability"] for r in selected]
            )
            hp, hc, _, included = combine(
                store,
                model,
                selected,
                scheduled,
                market,
                players,
                rp,
                rc,
                np.zeros(len(selected)),
                event,
                packets["2025/26", event]["deadline"],
            )
            assert included.all()
            rp[:, 1:] *= np.asarray([r["availability"] for r in selected])[:, None]
            rp[:, 0] = 1 - rp[:, 1:].sum(1)
            for i, row in enumerate(selected):
                player = players[row["player_id"]]
                key = row["fixture_id"], player.id
                outcome = {
                    "event": event,
                    "relevant": player.price >= 60 or player.ownership >= 5,
                    "weight": 1 + 4 * min(player.ownership / 20, 1),
                }
                for name, p, c in (("reference", rp, rc), ("candidate", hp, hc)):
                    scores, covered = [], []
                    for seed in protocol["seeds"]:
                        rng = np.random.default_rng(np.random.SeedSequence([seed, event, *key]))
                        size = protocol["worlds_per_seed"]
                        draws, _, _ = sample_point_outcomes(
                            rng,
                            p[i],
                            c[i],
                            row["dc_probability"],
                            model["residuals"],
                            player.position,
                            rng.random(size),
                            rng.random(size),
                        )
                        scores.append(crps(draws, truth[key]))
                        low, high = np.quantile(draws, [0.1, 0.9])
                        covered.append(float(low <= truth[key] <= high))
                    outcome[name] = {
                        "crps": float(np.mean(scores)),
                        "coverage80": float(np.mean(covered)),
                        "appearance_brier": float((1 - p[i, 0] - (minutes[key] > 0)) ** 2),
                        "sixty_brier": float((p[i, 2:].sum() - (minutes[key] >= 60)) ** 2),
                    }
                results.append(outcome)
        print("Future marginal distributions", event, len(selected), flush=True)
    metrics: dict[str, Any] = {}
    for label, subset in (("all", results), ("relevant", [r for r in results if r["relevant"]])):
        metrics[label] = {
            name: {
                metric: float(
                    np.average([r[name][metric] for r in subset], weights=[r["weight"] for r in subset])
                )
                for metric in ("crps", "coverage80", "appearance_brier", "sixty_brier")
            }
            for name in ("reference", "candidate")
        }
        differences = []
        for event in sorted({r["event"] for r in subset}):
            rows_at = [r for r in subset if r["event"] == event]
            differences.append(
                float(
                    np.average(
                        [r["candidate"]["crps"] - r["reference"]["crps"] for r in rows_at],
                        weights=[r["weight"] for r in rows_at],
                    )
                )
            )
        metrics[label]["crps_delta_95ci"] = paired_interval(differences)
    passed = (
        all(
            m["crps_delta_95ci"][1] < 0
            and all(m["candidate"][key] <= m["reference"][key] for key in ("appearance_brier", "sixty_brier"))
            for m in metrics.values()
        )
        and metrics["relevant"]["candidate"]["coverage80"] >= 0.75
    )
    report = {
        "at": now(),
        "protocol_id": protocol_id,
        "protocol": protocol,
        "metrics": metrics,
        "rows": len(results),
        "deadlines": len({r["event"] for r in results}),
        "passed": bool(passed),
        "promoted": False,
    }
    report["id"] = parent.put("future_distribution_validation", report)
    atomic_json(Path("artifacts/evaluations/future-distribution-validation.json"), report)
    return report
