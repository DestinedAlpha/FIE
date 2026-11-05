"""Matched full-policy replay with the immediate world samples held identical."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from .consensus import paired_interval, replay_packets
from .decision import canonical_rows, learned_forecast
from .decision_training import reprice
from .discipline import build_discipline
from .domain import TeamState
from .future_expert import frozen_spec
from .objectives import choose_recommendation
from .optimization import optimize
from .pipeline_replay import (
    attach_availability,
    attach_dated_sources,
    attach_market,
    attach_observations,
    link_file,
)
from .pipeline_verification import frozen_availability
from .rules import Rules, score_gameweek, transition
from .scenarios import action_from_row
from .storage import Store, atomic_json, digest, now
from .worlds import load_worlds


def prepare(parent, store, row, packet, originals, archive, future=None):
    frozen = json.loads(parent.artifact_path(row["decision_file"]).read_text())
    attach_observations(store, archive, packet)
    players, market = attach_market(store, packet, originals)
    model = {k: v for k, v in frozen["model"].items() if k != "id"}
    for name, spec in frozen["model_artifacts"].items():
        link_file(
            parent.artifact_path(spec["path"]), store.root / "models" / "primary-reference" / (name + ".txt")
        )
    availability = frozen_availability(parent, row)
    attach_availability(store, parent, availability)
    attach_dated_sources(parent, store, packet["deadline"])
    build_discipline(store)
    if future is not None:
        for spec in future["models"].values():
            original = parent.artifact_path(spec["path"])
            link_file(original, store.root / original.relative_to(parent.root))
        identity = store.put("future_horizon_model", future)
        model["future_horizon_model_id"] = identity
    model["id"] = store.put("decision_model", model)
    return frozen, players, market, model


def run_future_policy(parent, *, season="2025/26", events=None, worlds=4000, seconds=60):
    events = events or list(range(20, 39))
    baseline = parent.get("pipeline_replay")
    if worlds != baseline["protocol"]["worlds"] or seconds != baseline["protocol"]["seconds"]:
        raise ValueError("A matched replay must use the retained reference's worlds and solver budget")
    refinement = parent.get("horizon_refinement")
    spec = frozen_spec(parent, refinement, season + "-GW01")
    existing = {(r["season"], r["event"]): r for r in baseline["weeks"] if r["policy"] == "primary_multiweek"}
    diagnostic = season == parent.get("market")["season"]
    if not diagnostic and events != list(range(20, 39)):
        raise ValueError("Policy acceptance requires the complete GW20–38 window")
    originals, _, _ = canonical_rows(parent)
    packets = replay_packets(parent)
    signature = digest(
        {
            "baseline": baseline["id"],
            "future": spec,
            "season": season,
            "events": events,
            "worlds": worlds,
            "seconds": seconds,
            "code": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
            },
        }
    )
    directory = parent.root / "models" / ("future-policy-" + signature[:20])
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = (
        parent.root / "features" / ("future-policy-observations-" + digest(originals)[:20] + ".parquet")
    )
    if not archive_path.exists():
        pl.DataFrame(originals, infer_schema_length=None, strict=False).write_parquet(archive_path)
    archive = {"path": archive_path, "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
    guards = {
        name: parent.get(name)["id"]
        for name in ("state", "market", "decision_model", "forecast", "selected_plan")
    }
    protocol = {
        "signature": signature,
        "season": season,
        "events": events,
        "at": now(),
        "scope": "Complete chip-window expected-points policy; learned expert only beyond the current gameweek",
        "weight": spec["weight"],
        "training_before": spec["fit_before"],
        "worlds": worlds,
        "seconds": seconds,
        "baseline_id": baseline["id"],
        "refinement_id": refinement["id"],
        "gate": "Identical current-week worlds and no illegal transitions; positive net return gain plus lower future weighted loss against the refitted deployed reference",
        "current_season": "Diagnostic only after earlier-season policy acceptance; never selects configuration",
    }
    atomic_json(directory / "protocol.json", protocol)
    outcomes = []
    state = None
    # Completed decisions are never retuned when extending a diagnostic.
    if diagnostic and parent.list("future_policy_diagnostic", 1):
        previous = parent.get("future_policy_diagnostic")
        previous_events = [r["event"] for r in previous["weeks"]]
        if (
            previous["season"] == season
            and events[: len(previous_events)] == previous_events
            and previous["protocol"]["refinement_id"] == refinement["id"]
            and previous["protocol"]["worlds"] == worlds
            and previous["protocol"]["seconds"] == seconds
        ):
            from .future_policy_verification import verify_future_policy

            verify_future_policy(parent, previous)
            for saved in previous["weeks"]:
                body = json.loads(parent.artifact_path(saved["decision_file"]).read_text())
                if body["future_model"] != spec or body["input_id"] != packets[season, saved["event"]]["id"]:
                    raise ValueError("Cannot extend diagnostic across different frozen inputs or models")
                atomic_json(directory / f"gw{saved['event']}.json", saved)
    scratch = Path("artifacts/runtime/future-policy")
    scratch.mkdir(parents=True, exist_ok=True)
    for event in events:
        packet = packets[season, event]
        original_row = existing[season, event]
        path = directory / f"gw{event}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            state = TeamState.model_validate(saved["next_state"])
            outcomes.append(saved)
            continue
        with tempfile.TemporaryDirectory(prefix="deadline-", dir=scratch) as temporary:
            store = Store(Path(temporary) / "data")
            frozen, players, market, model = prepare(parent, store, original_row, packet, originals, archive)
            rules = Rules.from_bootstrap(packet["bootstrap"])
            if state is None:
                state = TeamState.model_validate(frozen["states_before"]["primary_multiweek"])
            state = reprice(state, players, rules)
            state.known_at = packet["deadline"]
            state_id = store.put("state", state.model_dump())
            end = 19 if event <= 19 else 38
            reference = learned_forecast(store, event, min(38, end + 4), worlds, 11)
            frozen, players, market, model = prepare(
                parent, store, original_row, packet, originals, archive, spec
            )
            forecast = learned_forecast(store, event, min(38, end + 4), worlds, 11)
            with load_worlds(store, reference) as ref, load_worlds(store, forecast) as actual:
                with np.load(
                    parent.artifact_path(original_row["decision_file"]).parent / "observed-worlds.npz"
                ) as old:
                    if not (
                        np.array_equal(ref[f"points_{event}"], old["points"])
                        and np.array_equal(ref[f"minutes_{event}"], old["minutes"])
                    ):
                        raise ValueError("Reference regeneration differs from retained baseline worlds")
                if not (
                    np.array_equal(ref[f"points_{event}"], actual[f"points_{event}"])
                    and np.array_equal(ref[f"minutes_{event}"], actual[f"minutes_{event}"])
                ):
                    raise ValueError("Future-only expert changed the current-week distribution")
                observed_file = directory / f"gw{event}-worlds.npz"
                np.savez_compressed(
                    observed_file,
                    points=actual[f"points_{event}"],
                    minutes=actual[f"minutes_{event}"],
                )
            print("Future-only world identity passed", season, event, flush=True)
            plan = optimize(store, state_id, forecast["id"], end, seconds, use_decision_critic=False)
            plan, selection = choose_recommendation(store, [plan], forecast)
            action = action_from_row(plan["ledger"][0])
            following, ledger = transition(state, action, players, rules)
            # Seal the selected policy before reading target outcomes.
            decision = {
                "season": season,
                "event": event,
                "input_id": packet["id"],
                "forecast_cutoff": forecast["cutoff"],
                "model": model,
                "reference_decision_file": original_row["decision_file"],
                "reference_decision_sha256": original_row["decision_sha256"],
                "future_model": spec,
                "observed_worlds": {
                    "path": str(observed_file),
                    "sha256": hashlib.sha256(observed_file.read_bytes()).hexdigest(),
                },
                "state_before": state.model_dump(),
                "action": action.model_dump(),
                "ledger": ledger,
                "planned_ledger": plan["ledger"],
                "solver": plan["solver"],
                "selection": selection,
                "immediate_worlds_identical": True,
                "forecast": [
                    {k: r[k] for k in ("player_id", "fixture_id", "event", "model_mean_before_mc")}
                    for r in forecast["fixtures"]
                ],
                "reference_forecast": [
                    {k: r[k] for k in ("player_id", "fixture_id", "event", "model_mean_before_mc")}
                    for r in reference["fixtures"]
                ],
            }
            decision_file = directory / f"gw{event}-decision.json"
            atomic_json(decision_file, decision)
            points: dict = defaultdict(float)
            minutes: dict = defaultdict(int)
            truth: dict = defaultdict(float)
            completed_fixtures = set()
            for row in originals:
                if row["season"] == season:
                    completed_fixtures.add(int(row["fixture"]))
                    truth[int(row["fixture"]), int(row["element"])] += float(row.get("total_points") or 0)
                    if int(row["round"]) == event:
                        points[int(row["element"])] += float(row.get("total_points") or 0)
                        minutes[int(row["element"])] += int(row.get("minutes") or 0)
            assert action.lineup is not None
            score = (
                score_gameweek(points, minutes, action.lineup, players, rules, action.chip) - ledger["hit"]
            )
            ref_by_key = {(r["fixture_id"], r["player_id"]): r for r in reference["fixtures"]}
            losses = []
            for record in forecast["fixtures"]:
                if record["event"] <= event or record["fixture_id"] not in completed_fixtures:
                    continue
                key = record["fixture_id"], record["player_id"]
                target = truth[key]
                weight = 1 + 4 * min(players[record["player_id"]].ownership / 20, 1)
                losses.append(
                    (
                        weight,
                        (record["model_mean_before_mc"] - target) ** 2,
                        (ref_by_key[key]["model_mean_before_mc"] - target) ** 2,
                    )
                )
            values = np.asarray(losses)
            loss_delta = (
                float(np.average(values[:, 1] - values[:, 2], weights=values[:, 0])) if len(values) else None
            )
            following.points += int(score)
            saved = {
                "season": season,
                "event": event,
                "net_points": score,
                "baseline_points": original_row["net_points"],
                "hit": ledger["hit"],
                "chip": action.chip,
                "next_state": following.model_dump(),
                "decision_file": str(decision_file),
                "decision_sha256": hashlib.sha256(decision_file.read_bytes()).hexdigest(),
                "future_weighted_loss_delta": loss_delta,
                "immediate_worlds_identical": True,
            }
            atomic_json(path, saved)
            outcomes.append(saved)
            state = following
            parent.put(
                "future_policy_progress",
                {
                    "at": now(),
                    "season": season,
                    "event": event,
                    "completed": len(outcomes),
                    "total": len(events),
                },
            )
            print(
                "Completed future policy",
                season,
                event,
                "candidate",
                score,
                "baseline",
                original_row["net_points"],
                flush=True,
            )
    delta = [r["net_points"] - r["baseline_points"] for r in outcomes]
    loss_changes = [
        r["future_weighted_loss_delta"] for r in outcomes if r["future_weighted_loss_delta"] is not None
    ]
    report = {
        "at": now(),
        "protocol": protocol,
        "season": season,
        "weeks": outcomes,
        "candidate_points": sum(r["net_points"] for r in outcomes),
        "baseline_points": sum(r["baseline_points"] for r in outcomes),
        "net_return_delta_95ci": paired_interval(delta),
        "future_weighted_loss_delta_95ci": paired_interval(loss_changes),
        "live_pointers_unchanged": all(parent.get(k)["id"] == v for k, v in guards.items()),
        "policy_gate_passed": not diagnostic and sum(delta) > 0 and paired_interval(loss_changes)[1] < 0,
        "diagnostic_only": diagnostic,
        "promoted": False,
        "directory": str(directory),
    }
    from .future_policy_verification import verify_future_policy

    report["verification"] = verify_future_policy(parent, report)
    kind = "future_policy_diagnostic" if diagnostic else "future_policy_replay"
    report["id"] = parent.put(kind, report)
    atomic_json(Path("artifacts/evaluations/" + kind.replace("_", "-") + ".json"), report)
    return report
