"""Independent checks of future-expert seals, world identity and actual returns."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np

from .consensus import replay_packets
from .decision import canonical_rows
from .decision_training import reprice
from .domain import Action, TeamState
from .future_expert import load_expert
from .pipeline_verification import frozen_availability
from .replay import deadline_rows
from .rules import Rules, score_gameweek, transition


def checked_json(store, path, checksum):
    resolved = store.artifact_path(path)
    if hashlib.sha256(resolved.read_bytes()).hexdigest() != checksum:
        raise ValueError("Future policy decision checksum mismatch")
    return json.loads(resolved.read_text())


def verify_future_policy(store, report):
    originals, _, _ = canonical_rows(store)
    packets = replay_packets(store)
    truth: dict = defaultdict(lambda: defaultdict(float))
    minutes: dict = defaultdict(lambda: defaultdict(int))
    for row in originals:
        key, player = (row["season"], int(row["round"])), int(row["element"])
        truth[key][player] += float(row.get("total_points") or 0)
        minutes[key][player] += int(row.get("minutes") or 0)
    state, previous = None, None
    for row in report["weeks"]:
        key = row["season"], row["event"]
        packet = packets[key]
        body = checked_json(store, row["decision_file"], row["decision_sha256"])
        baseline = checked_json(store, body["reference_decision_file"], body["reference_decision_sha256"])
        if body["input_id"] != packet["id"] or body["forecast_cutoff"] != packet["deadline"]:
            raise ValueError("Future policy uses a different deadline input")
        group = f"{key[0]}-GW{key[1]:02d}"
        training = body["future_model"]["training_evidence"]
        if (
            not training["fit_max_label"]
            < training["cal_max_label"]
            < body["future_model"]["fit_before"]
            <= group
        ):
            raise ValueError("Future expert training reaches its target")
        core = baseline["model"]["fit_evidence"]
        if not core["fit_max_group"] < core["calibration_max_group"] < group:
            raise ValueError("Reference training reaches its target")
        load_expert(store, body["future_model"])
        for item in baseline["model_artifacts"].values():
            if hashlib.sha256(store.artifact_path(item["path"]).read_bytes()).hexdigest() != item["hash"]:
                raise ValueError("Reference model checksum mismatch")
        frozen_availability(store, {"decision_file": body["reference_decision_file"]})
        worlds = body["observed_worlds"]
        path = store.artifact_path(worlds["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != worlds["sha256"]:
            raise ValueError("Future policy world checksum mismatch")
        old = store.artifact_path(body["reference_decision_file"]).parent / "observed-worlds.npz"
        with np.load(path) as candidate, np.load(old) as reference:
            if not all(np.array_equal(candidate[k], reference[k]) for k in ("points", "minutes")):
                raise ValueError("Future-only policy changed current-week worlds")
        players, _ = deadline_rows(
            [], packet["bootstrap"], packet["fixture_proof"]["fixtures"], *key, packet["deadline"]
        )
        rules = Rules.from_bootstrap(packet["bootstrap"])
        before = TeamState.model_validate(body["state_before"])
        if state is None:
            if before != TeamState.model_validate(baseline["states_before"]["primary_multiweek"]):
                raise ValueError("Candidate and baseline start from different accounts")
        else:
            expected = reprice(state, players, rules)
            expected.known_at = packet["deadline"]
            if previous != (key[0], key[1] - 1) or before != expected:
                raise ValueError("Future policy account continuity failed")
        action = Action.model_validate(body["action"])
        state, ledger = transition(before, action, players, rules)
        if ledger != body["ledger"] or ledger["hit"] != row["hit"] or action.chip != row["chip"]:
            raise ValueError("Future policy ledger differs from independent transition")
        assert action.lineup is not None
        score = (
            score_gameweek(truth[key], minutes[key], action.lineup, players, rules, action.chip)
            - ledger["hit"]
        )
        if score != row["net_points"]:
            raise ValueError("Future policy actual score mismatch")
        state.points += int(score)
        if state.model_dump() != row["next_state"]:
            raise ValueError("Future policy next state mismatch")
        previous = key
    return {
        "sealed_deadlines": len(report["weeks"]),
        "training_cutoffs_and_model_checksums": "passed",
        "current_week_world_identity": "passed",
        "continuous_accounts_and_actual_scores": "passed",
    }
