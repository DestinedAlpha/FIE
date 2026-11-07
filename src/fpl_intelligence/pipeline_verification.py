"""Check sealed decisions and continuity independently of replay execution."""

from __future__ import annotations

import hashlib
import json

from .consensus import replay_packets
from .decision_training import reprice
from .domain import Action, TeamState
from .replay import deadline_rows
from .rules import Rules, transition


def frozen_availability(store, row):
    """Relocate the retained availability bundle without altering its sealed proof."""
    path = store.artifact_path(row["decision_file"])
    body = json.loads(path.read_text())
    models = {}
    for name, spec in body["availability"]["models"].items():
        original = store.artifact_path(spec["path"])
        suffix = original.relative_to(store.root / "models").parts
        retained = path.parent.joinpath(*suffix)
        if hashlib.sha256(retained.read_bytes()).hexdigest() != spec["hash"]:
            raise ValueError("Retained availability model checksum mismatch")
        models[name] = {**spec, "path": str(retained)}
    return {**body["availability"], "models": models}


def audit_continuity(store, report):
    packets = replay_packets(store)
    states: dict[tuple[str, str], TeamState] = {}
    last_events: dict[tuple[str, str], int] = {}
    sealed = {}
    checked = 0
    for row in sorted(report["weeks"], key=lambda r: (r["season"], r["event"], r["policy"])):
        key = row["season"], row["event"]
        if key not in sealed:
            path = store.artifact_path(row["decision_file"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["decision_sha256"]:
                raise ValueError("Frozen decision checksum mismatch")
            body = json.loads(path.read_text())
            if len({json.dumps(s, sort_keys=True) for s in body["states_before"].values()}) != 1 and not any(
                k[0] == key[0] for k in states
            ):
                raise ValueError("Initial policy states differ")
            frozen_availability(store, row)
            packet = packets[key]
            players, _ = deadline_rows(
                [], packet["bootstrap"], packet["fixture_proof"]["fixtures"], *key, packet["deadline"]
            )
            sealed[key] = body, players, Rules.from_bootstrap(packet["bootstrap"])
        body, players, rules = sealed[key]
        frozen = next(a for a in body["actions"] if a["policy"] == row["policy"])
        if (
            row["action"] != frozen["action"]
            or row["ledger"] != frozen["ledger"]
            or row["solver"] != frozen["solver"]
        ):
            raise ValueError("Reported action differs from the sealed pre-scoring decision")
        before = TeamState.model_validate(body["states_before"][row["policy"]])
        account = row["season"], row["policy"]
        if account in states:
            if row["event"] != last_events[account] + 1:
                raise ValueError("Continuous replay skipped a deadline")
            expected = reprice(states[account], players, rules)
            expected.known_at = body["information_cutoff"]
            if expected != before:
                raise ValueError("Continuous account reset or changed between deadlines")
        following, _ = transition(before, Action.model_validate(row["action"]), players, rules)
        if following.model_dump() != frozen["next_state"]:
            raise ValueError("Sealed next account disagrees with the rules engine")
        following.points += int(row["net_points"])
        states[account], last_events[account] = following, row["event"]
        checked += 1
    return {
        "continuous_actions": checked,
        "sealed_actions": "matched",
        "account_continuity": "passed",
        "availability_checksums": "passed",
    }
