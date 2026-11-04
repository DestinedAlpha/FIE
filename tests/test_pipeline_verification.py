import copy
import hashlib
import json

import pytest

from fpl_intelligence.domain import Action
from fpl_intelligence.pipeline_verification import audit_continuity
from fpl_intelligence.rules import best_lineup, transition
from fpl_intelligence.storage import Store


@pytest.fixture
def continuous_proof(tmp_path, market, rules, state, monkeypatch):
    store = Store(tmp_path / "data")
    packets, rows = {}, []
    accounts = {name: state.model_copy(deep=True) for name in ("primary", "control")}
    for event in (9, 10):
        deadline = f"2026-10-{event + 20}T11:00:00+00:00"
        packet = {
            "bootstrap": {},
            "fixture_proof": {"fixtures": []},
            "deadline": deadline,
            "id": f"input-{event}",
            "season": state.season,
            "event": event,
        }
        packets[state.season, event] = packet
        frozen = {
            "states_before": {},
            "availability": {"models": {}},
            "actions": [],
            "information_cutoff": deadline,
            "protocol_id": "test-protocol",
            "input_id": packet["id"],
            "model": {
                "id": "test-model",
                "fit_evidence": {"fit_max_group": "2026/27-GW07", "calibration_max_group": "2026/27-GW08"},
            },
            "model_artifacts": {},
        }
        for name, account in accounts.items():
            account.known_at = deadline
            action = Action(lineup=best_lineup(list(range(1, 16)), {p: 1 for p in market}, market, rules))
            following, ledger = transition(account, action, market, rules)
            frozen["states_before"][name] = account.model_dump()
            frozen["actions"].append(
                {
                    "policy": name,
                    "action": action.model_dump(),
                    "ledger": ledger,
                    "next_state": following.model_dump(),
                    "solver": {"gap": 0},
                    "forecast_summary": {"id": f"forecast-{event}", "cutoff": deadline},
                }
            )
            following.points += 12
            accounts[name] = following
        path = store.root / "models" / f"gw{event}" / "frozen-decision.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(frozen))
        for action in frozen["actions"]:
            rows.append(
                {
                    "season": state.season,
                    "event": event,
                    "policy": action["policy"],
                    "action": action["action"],
                    "ledger": action["ledger"],
                    "solver": action["solver"],
                    "decision_file": str(path),
                    "decision_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "net_points": 12,
                }
            )
    monkeypatch.setattr("fpl_intelligence.pipeline_verification.replay_packets", lambda *_: packets)
    monkeypatch.setattr("fpl_intelligence.pipeline_verification.deadline_rows", lambda *_: (market, []))
    monkeypatch.setattr(
        "fpl_intelligence.pipeline_verification.reprice", lambda account, *_: account.model_copy(deep=True)
    )
    monkeypatch.setattr("fpl_intelligence.rules.Rules.from_bootstrap", lambda *_: rules)
    monkeypatch.setattr("fpl_intelligence.pipeline_checkpoints.deadline_rows", lambda *_: (market, []))
    monkeypatch.setattr(
        "fpl_intelligence.pipeline_checkpoints.reprice", lambda account, *_: account.model_copy(deep=True)
    )
    return store, {"weeks": rows}


def test_independent_verification_checks_sealed_actions_and_continuous_accounts(continuous_proof):
    store, report = continuous_proof
    assert audit_continuity(store, report)["continuous_actions"] == 4
    changed = copy.deepcopy(report)
    action = changed["weeks"][0]["action"]["lineup"]
    action["captain"], action["vice"] = action["vice"], action["captain"]
    with pytest.raises(ValueError, match="sealed pre-scoring"):
        audit_continuity(store, changed)


def test_continuity_verifier_rejects_individually_legal_account_resets(continuous_proof):
    store, report = continuous_proof
    for row in report["weeks"]:
        if row["event"] == 9:
            row["net_points"] = 999
    with pytest.raises(ValueError, match="reset or changed"):
        audit_continuity(store, report)


def test_portable_proofs_recover_accounts_without_refitting_or_reading_future_outcomes(continuous_proof):
    from fpl_intelligence.domain import TeamState
    from fpl_intelligence.pipeline_checkpoints import recover_deadline
    from fpl_intelligence.pipeline_verification import replay_packets

    store, report = continuous_proof
    packets = replay_packets(store)
    observations = [
        {"season": "2026/27", "round": e, "element": p, "total_points": 1, "minutes": 90}
        for e in (9, 10)
        for p in range(1, 46)
    ]
    states = {}
    for event in (9, 10):
        row = next(r for r in report["weeks"] if r["event"] == event)
        path = store.artifact_path(row["decision_file"])
        result = recover_deadline(
            store, path, packets["2026/27", event], "test-protocol", observations, states
        )
        assert [r["net_points"] for r in result["weeks"]] == [12, 12]
        states = {k: TeamState.model_validate(v) for k, v in result["next_states"].items()}
    assert all(s.points == 24 and s.event == 11 for s in states.values())
    with pytest.raises(ValueError, match="different inputs or implementation"):
        recover_deadline(store, path, packets["2026/27", 10], "another-protocol", observations, states)


def test_new_evaluation_exports_work_without_private_account(tmp_path):
    from fastapi.testclient import TestClient

    from fpl_intelligence.api import create_app

    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as client:
        assert client.get("/api/prospective/accounts").json() == {
            "shadow_registry": None,
            "shadow_cohort": None,
            "shadow_evaluation": None,
        }
        assert client.get("/api/replay/pipeline").status_code == 422
