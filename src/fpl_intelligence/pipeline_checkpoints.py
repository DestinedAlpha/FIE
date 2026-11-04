"""Recover completed historical work from immutable portable decision bundles."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from .decision_training import reprice
from .domain import Action, TeamState
from .pipeline_verification import frozen_availability
from .replay import deadline_rows
from .rules import Rules, score_gameweek, transition


def recover_deadline(store, path, packet, protocol_id, originals, previous):
    body = json.loads(path.read_text())
    if (
        body["protocol_id"] != protocol_id
        or body["input_id"] != packet["id"]
        or body["information_cutoff"] != packet["deadline"]
    ):
        raise ValueError("Retained replay decision has different inputs or implementation")
    players, _ = deadline_rows(
        [],
        packet["bootstrap"],
        packet["fixture_proof"]["fixtures"],
        packet["season"],
        packet["event"],
        packet["deadline"],
    )
    rules = Rules.from_bootstrap(packet["bootstrap"])
    training = body["model"]["fit_evidence"]
    group = f"{packet['season']}-GW{packet['event']:02d}"
    if not training["fit_max_group"] < training["calibration_max_group"] < group:
        raise ValueError("Retained model reaches its target deadline")
    for spec in body["model_artifacts"].values():
        if hashlib.sha256(store.artifact_path(spec["path"]).read_bytes()).hexdigest() != spec["hash"]:
            raise ValueError("Retained replay model checksum mismatch")
    frozen_availability(store, {"decision_file": str(path)})
    points: dict[int, float] = defaultdict(float)
    minutes: dict[int, int] = defaultdict(int)
    for row in originals:
        if row["season"] == packet["season"] and int(row["round"]) == packet["event"]:
            points[int(row["element"])] += float(row.get("total_points") or 0)
            minutes[int(row["element"])] += int(row.get("minutes") or 0)
    if not points:
        raise ValueError("No actual observations for the retained deadline")
    forecast = body["actions"][0]["forecast_summary"]
    results, following_states = [], {}
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    for frozen in body["actions"]:
        name = frozen["policy"]
        before = TeamState.model_validate(body["states_before"][name])
        if previous:
            expected = reprice(previous[name], players, rules)
            expected.known_at = packet["deadline"]
            if expected != before:
                raise ValueError("Retained account does not continue the preceding deadline")
        action = Action.model_validate(frozen["action"])
        following, ledger = transition(before, action, players, rules)
        if following.model_dump() != frozen["next_state"] or ledger != frozen["ledger"]:
            raise ValueError("Retained action no longer matches its legal transition")
        assert action.lineup is not None
        score = score_gameweek(points, minutes, action.lineup, players, rules, action.chip) - ledger["hit"]
        following.points += int(score)
        following_states[name] = following
        results.append(
            {
                "season": packet["season"],
                "event": packet["event"],
                "policy": name,
                "net_points": score,
                "action": frozen["action"],
                "ledger": ledger,
                "solver": frozen["solver"],
                "decision_file": str(path),
                "decision_sha256": checksum,
                "input_id": packet["id"],
                "model_id": body["model"]["id"],
                "forecast_id": forecast["id"],
                "forecast_cutoff": forecast["cutoff"],
            }
        )
    return {
        "weeks": results,
        "next_states": {k: v.model_dump() for k, v in following_states.items()},
        "frozen_models": str(path.parent),
    }
