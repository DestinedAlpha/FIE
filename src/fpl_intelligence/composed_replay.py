"""Deadline-refitted replay of the served model composition, without live writes.

This is a later statistical reconstruction and a bounded-search diagnostic. It
does not certify an unlimited production solve or constitute an untouched holdout.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import polars as pl

from .consensus import paired_interval, replay_packets
from .decision import canonical_rows, learned_forecast
from .decision_training import reprice
from .domain import Action, TeamState
from .future_expert import frozen_spec, load_expert
from .future_policy_replay import prepare
from .objectives import SELECTION_VERSION, choose_recommendation
from .optimization import optimize
from .pipeline_replay import link_file
from .pipeline_verification import frozen_availability
from .replay import deadline_rows, replay_configuration_hash
from .rules import Rules, score_gameweek, transition
from .scenarios import action_from_row
from .storage import Store, atomic_json, digest, now
from .strategy import solve_action

POLICIES = ("composed_multiweek", "future_reference_multiweek", "composed_one_week_no_chips")
LIVE = ("state", "market", "decision_model", "forecast", "selected_plan")


def checked(store, spec):
    path = store.artifact_path(spec["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["hash"]:
        raise ValueError("Composed replay artifact checksum mismatch")
    return path


def attach_role(parent, stage, weekly, model):
    """Install only the retained strictly-earlier weights, not today's weights."""
    evidence = weekly["training"]["role_minutes"]
    group = f"{weekly['season']}-GW{weekly['event']:02d}"
    if not evidence["fit_max_group"] < evidence["calibration_max_group"] < group:
        raise ValueError("Role training reaches the evaluation deadline")
    spec = parent.get("role_minutes_model", evidence["configuration_id"])
    spec = {k: v for k, v in spec.items() if k != "id"}
    retained = weekly["training"]["models"]["role_minutes"]
    original = checked(parent, retained)
    target = stage.root / "models" / "deadline-role" / "minutes.txt"
    link_file(original, target)
    spec.update(
        path=str(target),
        hash=retained["hash"],
        curves=evidence["curves"],
        context_archive_id=evidence["context_id"],
        training_evidence=evidence,
        completed_labels_through=evidence["calibration_max_group"],
    )
    features = parent.get("fusion_features", evidence["feature_id"])
    for kind in ("shot_archive", "temporal_model", "elite_cohort", "workload"):
        identity = features["lineage"].get(kind)
        if identity:
            body = parent.get(kind, identity)
            stage.put(kind, {k: v for k, v in body.items() if k != "id"})
    identity = stage.put("fusion_features", {k: v for k, v in features.items() if k != "id"})
    if identity != spec["feature_id"]:
        raise ValueError("Retained role feature identity changed")
    role_id = stage.put("role_minutes_model", spec)
    body = {k: v for k, v in model.items() if k != "id"}
    body["role_minutes_model_id"] = role_id
    body["id"] = stage.put("decision_model", body)
    return body, spec


def verify_composed(parent, report):
    """Independently re-score sealed actions and verify every account transition."""
    packets = replay_packets(parent)
    originals, _, _ = canonical_rows(parent)
    points: dict = defaultdict(lambda: defaultdict(float))
    minutes: dict = defaultdict(lambda: defaultdict(int))
    for row in originals:
        key = row["season"], int(row["round"])
        points[key][int(row["element"])] += float(row.get("total_points") or 0)
        minutes[key][int(row["element"])] += int(row.get("minutes") or 0)
    states: dict = {}
    events: dict = {}
    initial = None
    for row in report["weeks"]:
        key = row["season"], row["event"]
        packet = packets[key]
        body = json.loads(
            checked(parent, {"path": row["decision_file"], "hash": row["decision_sha256"]}).read_text()
        )
        if body["input_id"] != packet["id"] or body["cutoff"] != packet["deadline"]:
            raise ValueError("Composed replay deadline input changed")
        reference = json.loads(checked(parent, body["reference_decision"]).read_text())
        group = f"{key[0]}-GW{key[1]:02d}"
        for evidence in (reference["model"]["fit_evidence"], body["role_evidence"]):
            if not evidence["fit_max_group"] < evidence["calibration_max_group"] < group:
                raise ValueError("Composed model training reaches its target")
        future = body["future_model"]
        if (
            not future["training_evidence"]["fit_max_label"]
            < future["training_evidence"]["cal_max_label"]
            < future["fit_before"]
            <= group
        ):
            raise ValueError("Composed future expert reaches its target")
        load_expert(parent, future)
        for spec in reference["model_artifacts"].values():
            checked(parent, spec)
        frozen_availability(parent, {"decision_file": body["reference_decision"]["path"]})
        checked(parent, body["role_artifact"])
        players, _ = deadline_rows(
            [], packet["bootstrap"], packet["fixture_proof"]["fixtures"], *key, packet["deadline"]
        )
        rules = Rules.from_bootstrap(packet["bootstrap"])
        if set(body["actions"]) != set(POLICIES):
            raise ValueError("Composed replay omitted a control")
        for policy in POLICIES:
            sealed, result = body["actions"][policy], row["outcomes"][policy]
            before = TeamState.model_validate(sealed["state_before"])
            if policy in states:
                expected = reprice(states[policy], players, rules)
                expected.known_at = packet["deadline"]
                if before != expected or events[policy] != key[1] - 1:
                    raise ValueError("Composed replay account continuity failed")
            else:
                initial = initial or before
                if before != initial or before != TeamState.model_validate(
                    reference["states_before"]["primary_multiweek"]
                ):
                    raise ValueError("Composed replay initial accounts differ")
            action = Action.model_validate(sealed["action"])
            following, ledger = transition(before, action, players, rules)
            if ledger != sealed["ledger"] or ledger["hit"] != result["hit"] or action.chip != result["chip"]:
                raise ValueError("Composed replay ledger mismatch")
            if policy == POLICIES[2] and action.chip:
                raise ValueError("The chip-free control spent a chip")
            assert action.lineup is not None
            score = (
                score_gameweek(points[key], minutes[key], action.lineup, players, rules, action.chip)
                - ledger["hit"]
            )
            if score != result["net_points"]:
                raise ValueError("Composed replay actual score mismatch")
            following.points += int(score)
            if following.model_dump() != result["next_state"]:
                raise ValueError("Composed replay reported an incorrect next state")
            states[policy], events[policy] = following, key[1]
    if [r["event"] for r in report["weeks"]] != report["protocol"]["events"]:
        raise ValueError("Composed replay is incomplete")
    return {
        "sealed_deadlines": len(report["weeks"]),
        "independently_scored_actions": len(report["weeks"]) * len(POLICIES),
        "training_cutoffs_and_model_checksums": "passed",
        "account_continuity": "passed",
    }


def run_composed(parent, *, end_event=9, worlds=4000, seconds=60):
    if not 100 <= worlds <= 20000 or seconds <= 0:
        raise ValueError("Invalid research budget")
    season = parent.get("market")["season"]
    events = list(range(1, end_event + 1))
    market = parent.get("market")
    if not events or any(
        not next(e for e in market["events"] if e["id"] == event).get("data_checked") for event in events
    ):
        raise ValueError("Replay accepts finalized gameweeks only")
    active = parent.get("decision_model")
    role = parent.get("role_minutes_model", active["role_minutes_model_id"])
    weekly = parent.get("role_weekly_replay")
    if weekly["configuration_hash"] != replay_configuration_hash(role):
        raise ValueError("Retained role replay has a different model configuration")
    weeks = {r["event"]: r for r in weekly["weeks"] if r["season"] == season}
    baseline = parent.get("pipeline_replay")
    reference = {
        r["event"]: r
        for r in baseline["weeks"]
        if r["season"] == season and r["policy"] == "primary_multiweek"
    }
    future = frozen_spec(parent, parent.get("horizon_refinement"), season + "-GW01")
    guards = {k: parent.get(k)["id"] for k in LIVE}
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    signature = digest(
        {
            "model": active["id"],
            "weekly": weekly["id"],
            "baseline": baseline["id"],
            "future": future,
            "events": events,
            "worlds": worlds,
            "seconds": seconds,
            "code": code,
        }
    )
    directory = parent.root / "models" / ("composed-replay-" + signature[:20])
    protocol = {
        "signature": signature,
        "season": season,
        "events": events,
        "worlds": worlds,
        "seconds_per_mip": seconds,
        "selection_version": SELECTION_VERSION,
        "production_model_id": active["id"],
        "role_configuration_hash": weekly["configuration_hash"],
        "initial_account": "Identical frozen ownership-based £100m GW1 seed",
        "policies": list(POLICIES),
        "scope": "Current model families refitted before each deadline, full chip-window planning with current selection guards; identical random seed and per-MIP main-search allowance. Selection controls use the production minimum 60 seconds. One-week control uses the same composed forecast but no chips.",
        "limitations": [
            "Later reconstruction; not prospective",
            "Current season is a diagnostic, not an untouched promotion holdout",
            "Bounded research search does not certify unlimited live policy",
            "One-week control conserves chips and is not a matched chip strategy",
            "No parameter or model selection from this diagnostic",
        ],
    }
    atomic_json(directory / "protocol.json", protocol)
    originals, _, _ = canonical_rows(parent)
    archive_path = parent.root / "features" / ("composed-observations-" + digest(originals)[:20] + ".parquet")
    if not archive_path.exists():
        pl.DataFrame(originals, infer_schema_length=None, strict=False).write_parquet(archive_path)
    archive = {"path": archive_path, "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
    packets = replay_packets(parent)
    states: dict = {}
    outcomes: list[dict] = []
    for event in events:
        packet, retained = packets[season, event], reference[event]
        checkpoint = directory / f"gw{event}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            # Never trust a checkpoint before verifying its seal and continuity.
            partial = {
                "protocol": {**protocol, "events": list(range(1, event + 1))},
                "weeks": outcomes + [saved],
            }
            verify_composed(parent, partial)
            outcomes.append(saved)
            states = {p: TeamState.model_validate(saved["outcomes"][p]["next_state"]) for p in POLICIES}
            continue
        with tempfile.TemporaryDirectory(prefix="fie-composed-") as temporary:
            stage = Store(Path(temporary) / "data")
            checked(parent, {"path": retained["decision_file"], "hash": retained["decision_sha256"]})
            frozen, players, _, model = prepare(parent, stage, retained, packet, originals, archive, future)
            if next(e["id"] for e in stage.get("market")["events"] if e.get("is_next")) != event:
                raise ValueError("Historical market would skip the served role model")
            for name, spec in frozen["model_artifacts"].items():
                checked(parent, spec)
                if spec["hash"] != weeks[event]["training"]["models"][name]["hash"]:
                    raise ValueError("Base and role replay weights belong to different fits")
            if weeks[event]["input_id"] != packet["id"]:
                raise ValueError("Role replay uses a different deadline packet")
            end = 19 if event <= 19 else 38
            reference_forecast = learned_forecast(stage, event, min(38, end + 4), worlds, 11)
            composed = None
            rules = Rules.from_bootstrap(packet["bootstrap"])
            actions, following_states = {}, {}
            # Finish the reference while its model is current. The production
            # optimizer correctly rejects forecasts after a model replacement.
            for policy in (POLICIES[1], POLICIES[0], POLICIES[2]):
                if policy == POLICIES[0]:
                    model, _ = attach_role(parent, stage, weeks[event], model)
                    composed = learned_forecast(stage, event, min(38, end + 4), worlds, 11)
                state = states.get(policy) or TeamState.model_validate(
                    frozen["states_before"]["primary_multiweek"]
                )
                state = reprice(state, players, rules)
                state.known_at = packet["deadline"]
                state_id = stage.put("state", state.model_dump())
                forecast = reference_forecast if policy == POLICIES[1] else composed
                assert forecast is not None
                if policy == POLICIES[2]:
                    means = {r["player_id"]: r["mean"] for r in forecast["players"] if r["event"] == event}
                    d = solve_action(state, players, rules, means, seconds=seconds, relative_gap=0)
                    action, solver, selection = d.action, d.solver, None
                else:
                    plan = optimize(stage, state_id, forecast["id"], end, seconds, use_decision_critic=False)
                    plan, selection = choose_recommendation(stage, [plan], forecast)
                    action, solver = action_from_row(plan["ledger"][0]), plan["solver"]
                following, ledger = transition(state, action, players, rules)
                actions[policy] = {
                    "state_before": state.model_dump(),
                    "action": action.model_dump(),
                    "ledger": ledger,
                    "solver": solver,
                    "selection": selection,
                }
                following_states[policy] = following
                print(
                    "Solved composed diagnostic",
                    event,
                    policy,
                    "hit",
                    ledger["hit"],
                    "chip",
                    action.chip,
                    flush=True,
                )
            decision = {
                "season": season,
                "event": event,
                "input_id": packet["id"],
                "cutoff": packet["deadline"],
                "reference_decision": {
                    "path": retained["decision_file"],
                    "hash": retained["decision_sha256"],
                },
                "role_artifact": weeks[event]["training"]["models"]["role_minutes"],
                "role_evidence": weeks[event]["training"]["role_minutes"],
                "future_model": future,
                "actions": actions,
            }
            decision_file = directory / f"gw{event}-decision.json"
            atomic_json(decision_file, decision)  # Seal all policies before scoring.
            points: dict = defaultdict(float)
            minutes: dict = defaultdict(int)
            for row in originals:
                if row["season"] == season and int(row["round"]) == event:
                    points[int(row["element"])] += float(row.get("total_points") or 0)
                    minutes[int(row["element"])] += int(row.get("minutes") or 0)
            saved = {
                "season": season,
                "event": event,
                "decision_file": str(decision_file),
                "decision_sha256": hashlib.sha256(decision_file.read_bytes()).hexdigest(),
                "outcomes": {},
            }
            for policy, sealed in actions.items():
                action = Action.model_validate(sealed["action"])
                assert action.lineup is not None
                score = (
                    score_gameweek(points, minutes, action.lineup, players, rules, action.chip)
                    - sealed["ledger"]["hit"]
                )
                following = following_states[policy]
                following.points += int(score)
                saved["outcomes"][policy] = {
                    "net_points": score,
                    "hit": sealed["ledger"]["hit"],
                    "chip": action.chip,
                    "next_state": following.model_dump(),
                }
                states[policy] = following
            atomic_json(checkpoint, saved)
            outcomes.append(saved)
            print(
                "Scored composed diagnostic",
                event,
                {p: r["net_points"] for p, r in saved["outcomes"].items()},
                flush=True,
            )
    delta = [
        r["outcomes"][POLICIES[0]]["net_points"] - r["outcomes"][POLICIES[1]]["net_points"] for r in outcomes
    ]
    report = {
        "at": now(),
        "protocol": protocol,
        "weeks": outcomes,
        "totals": {p: sum(r["outcomes"][p]["net_points"] for r in outcomes) for p in POLICIES},
        "composed_minus_future_reference_95ci": paired_interval(delta),
        "diagnostic_only": True,
        "promoted": False,
        "live_pointers_unchanged": all(parent.get(k)["id"] == v for k, v in guards.items()),
    }
    report["verification"] = verify_composed(parent, report)
    if not report["live_pointers_unchanged"]:
        raise ValueError("Live state changed during the isolated research replay")
    report["id"] = parent.put("composed_policy_replay", report)
    atomic_json(Path("artifacts/evaluations/composed_policy_replay.json"), report)
    return report
