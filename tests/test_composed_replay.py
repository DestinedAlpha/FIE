import hashlib
import json
from copy import deepcopy

import pytest

from fpl_intelligence.composed_replay import POLICIES, verify_composed
from fpl_intelligence.domain import Action
from fpl_intelligence.rules import best_lineup, score_gameweek, transition
from fpl_intelligence.storage import Store, atomic_json


def sealed_replay(tmp_path, monkeypatch, market, rules, state):
    store = Store(tmp_path / "data")
    directory = store.root / "models" / "sealed"
    spec = directory / "role.txt"
    spec.parent.mkdir(parents=True)
    spec.write_text("retained weight proof")
    artifact = {"path": str(spec), "hash": hashlib.sha256(spec.read_bytes()).hexdigest()}
    evidence = {"fit_max_group": "2025/26-GW30", "calibration_max_group": "2025/26-GW38"}
    original = directory / "reference.json"
    atomic_json(
        original,
        {
            "model": {"fit_evidence": evidence},
            "model_artifacts": {"minutes": artifact},
            "states_before": {"primary_multiweek": state.model_dump()},
        },
    )
    packet = {"id": "pinned", "deadline": state.known_at, "bootstrap": {}, "fixture_proof": {"fixtures": []}}
    monkeypatch.setattr(
        "fpl_intelligence.composed_replay.replay_packets", lambda *_: {(state.season, state.event): packet}
    )
    monkeypatch.setattr("fpl_intelligence.composed_replay.deadline_rows", lambda *_: (market, []))
    monkeypatch.setattr("fpl_intelligence.composed_replay.load_expert", lambda *_: {})
    monkeypatch.setattr("fpl_intelligence.composed_replay.frozen_availability", lambda *_: {})
    monkeypatch.setattr("fpl_intelligence.composed_replay.Rules.from_bootstrap", lambda *_: rules)
    originals = [
        {"season": state.season, "round": state.event, "element": p, "total_points": p % 8, "minutes": 90}
        for p in market
    ]
    monkeypatch.setattr("fpl_intelligence.composed_replay.canonical_rows", lambda *_: (originals, {}, {}))
    points = {r["element"]: r["total_points"] for r in originals}
    minutes = {p: 90 for p in market}
    action = Action(lineup=best_lineup([x.player_id for x in state.lots], points, market, rules))
    following, ledger = transition(state, action, market, rules)
    score = score_gameweek(points, minutes, action.lineup, market, rules)
    following.points += int(score)
    body = {
        "input_id": "pinned",
        "cutoff": state.known_at,
        "reference_decision": {
            "path": str(original),
            "hash": hashlib.sha256(original.read_bytes()).hexdigest(),
        },
        "role_evidence": evidence,
        "role_artifact": artifact,
        "future_model": {
            "fit_before": "2026/27-GW01",
            "training_evidence": {"fit_max_label": "2025/26-GW30", "cal_max_label": "2025/26-GW38"},
        },
        "actions": {
            p: {"state_before": state.model_dump(), "action": action.model_dump(), "ledger": ledger}
            for p in POLICIES
        },
    }
    decision = directory / "decision.json"
    atomic_json(decision, body)
    report = {
        "protocol": {"events": [state.event]},
        "weeks": [
            {
                "season": state.season,
                "event": state.event,
                "decision_file": str(decision),
                "decision_sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
                "outcomes": {
                    p: {
                        "net_points": score,
                        "hit": ledger["hit"],
                        "chip": None,
                        "next_state": following.model_dump(),
                    }
                    for p in POLICIES
                },
            }
        ],
    }
    return store, report, decision, spec


def test_sealed_scores_and_financial_state_are_independently_verified(
    tmp_path, monkeypatch, market, rules, state
):
    store, report, _, _ = sealed_replay(tmp_path, monkeypatch, market, rules, state)
    assert verify_composed(store, report)["independently_scored_actions"] == 3
    for field in ("net_points", "next_state", "hit"):
        invalid = deepcopy(report)
        result = invalid["weeks"][0]["outcomes"][POLICIES[0]]
        if field == "next_state":
            result[field]["bank"] += 1
        else:
            result[field] += 4
        with pytest.raises(ValueError):
            verify_composed(store, invalid)


def test_changed_decision_or_weights_fail_verification(tmp_path, monkeypatch, market, rules, state):
    store, report, decision, weight = sealed_replay(tmp_path, monkeypatch, market, rules, state)
    weight.write_text("different weights")
    with pytest.raises(ValueError, match="checksum"):
        verify_composed(store, report)
    decision.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify_composed(store, report)


def test_valid_checksum_cannot_hide_target_week_training(tmp_path, monkeypatch, market, rules, state):
    store, report, decision, _ = sealed_replay(tmp_path, monkeypatch, market, rules, state)
    body = json.loads(decision.read_text())
    body["role_evidence"]["calibration_max_group"] = f"{state.season}-GW{state.event:02d}"
    atomic_json(decision, body)
    report["weeks"][0]["decision_sha256"] = hashlib.sha256(decision.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="training reaches"):
        verify_composed(store, report)
