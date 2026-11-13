import hashlib
import importlib.util
import json
import sqlite3
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import highspy
import pytest

from fpl_intelligence.context import extraction_eligible, score_authentic
from fpl_intelligence.domain import OptimizeRequest
from fpl_intelligence.optimization import MIP
from fpl_intelligence.readiness import operational_readiness
from fpl_intelligence.search_control import configure_solver, search_session
from fpl_intelligence.storage import Store


def test_authentic_extraction_requires_literal_grounding_and_preserves_training_uncertainty():
    cases = [
        {
            "player_id": 10,
            "name": "A",
            "text": "A is back in training.",
            "required": ["training"],
            "allowed": {"training": ["reported"]},
        },
        {"player_id": 11, "name": "B", "text": "B scored yesterday.", "required": [], "allowed": {}},
    ]
    claim = SimpleNamespace(
        player_id=10, claim_type="training", status="reported", evidence_span="A is back in training."
    )
    rows, unknown = score_authentic(cases, [claim])
    assert all(r["passed"] for r in rows) and unknown == 0
    claim.status = "confirmed"
    assert not score_authentic(cases, [claim])[0][0]["passed"]
    claim.status, claim.evidence_span = "reported", "A will start."
    assert not score_authentic(cases, [claim])[0][0]["literal_support"]
    claim.player_id = 99999
    assert score_authentic(cases, [claim])[1] == 1
    false = SimpleNamespace(
        player_id=11, claim_type="available", status="confirmed", evidence_span="B scored yesterday."
    )
    assert not score_authentic(cases, [false])[0][1]["passed"]


def installer():
    spec = importlib.util.spec_from_file_location("audited_installer", Path("scripts/install_update.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def installation(tmp_path):
    source, target = tmp_path / "update", tmp_path / "existing"
    source.mkdir()
    target.mkdir()
    (target / "data").mkdir()
    (target / "pyproject.toml").write_text('name = "fpl-intelligence"')
    (target / ".env").write_text("TEST_SECRET=preserve")
    (target / "app.py").write_text("old")
    with sqlite3.connect(target / "data/state.sqlite") as db:
        db.execute("create table account(bank integer)")
        db.execute("insert into account values (3)")
    files = {"app.py": "new", "added.py": "new module"}
    for name, body in files.items():
        (source / name).write_text(body)
    (source / "app-files.json").write_text(json.dumps(list(files)))
    (source / "update-checksums.json").write_text(
        json.dumps(
            {"files": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files}}
        )
    )
    return source, target


def test_tampered_update_cannot_modify_existing_installation(tmp_path):
    source, target = installation(tmp_path)
    (source / "app.py").write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        installer().install(source, target)
    assert (target / "app.py").read_text() == "old"
    assert not (target / "added.py").exists()


@pytest.mark.parametrize(
    "name", [".ENV", "DATA/state.sqlite", "C:/outside.py", "../outside.py", "dir\\outside.py"]
)
def test_update_private_paths_are_rejected_on_all_platforms(tmp_path, name):
    source, target = installation(tmp_path)
    (source / "app-files.json").write_text(json.dumps([name]))
    with pytest.raises(ValueError, match="private or unsafe"):
        installer().install(source, target)
    assert (target / "app.py").read_text() == "old"


def test_failed_research_merge_restores_application_and_database(tmp_path, monkeypatch):
    source, target = installation(tmp_path)
    module = installer()

    def fail(*_):
        with sqlite3.connect(target / "data/state.sqlite") as db:
            db.execute("update account set bank=999")
        raise RuntimeError("interrupted merge")

    monkeypatch.setattr(module, "merge_research", fail)
    with pytest.raises(RuntimeError, match="interrupted merge"):
        module.install(source, target)
    assert (target / "app.py").read_text() == "old"
    assert not (target / "added.py").exists()
    assert (target / ".env").read_text() == "TEST_SECRET=preserve"
    with sqlite3.connect(target / "data/state.sqlite") as db:
        assert db.execute("select bank from account").fetchone()[0] == 3
    journal = next(target.glob("artifacts/runtime/before-update-*/transaction.json"))
    assert json.loads(journal.read_text())["status"] == "rolled_back"


def test_unlimited_solve_has_no_clock_or_early_gap_and_does_not_leak_to_research():
    solver = highspy.Highs()
    with search_session(unlimited=True):
        assert configure_solver(solver, 0.001, 0.1)
        assert solver.getOptionValue("time_limit")[1] == float("inf")
        assert solver.getOptionValue("mip_rel_gap")[1] == 0
        assert solver.getOptionValue("mip_abs_gap")[1] == 0
    other = highspy.Highs()
    assert not configure_solver(other, 3.5, 0.01)
    assert other.getOptionValue("time_limit")[1] == 3.5
    model = MIP()
    x = model.var("x", cost=3)
    model.row({x: 1}, upper=1)
    _, result = model.solve(None)
    assert result["status"] == "Optimal" and result["gap"] == 0
    assert result["time_limit"] is None


def test_cancelled_search_stops_before_solving():
    cancel = Event()
    cancel.set()
    with pytest.raises(CancelledError):
        with search_session(unlimited=True, cancel=cancel):
            pytest.fail("Cancelled work must not start")


def test_cancel_reaches_running_highs_callbacks():
    import numpy as np

    cancel = Event()
    model = MIP()
    rng = np.random.default_rng(19)
    variables = [model.var(i, cost=float(v)) for i, v in enumerate(rng.uniform(1, 20, 80))]
    for _ in range(30):
        weights = rng.uniform(1, 10, 80)
        model.row(dict(zip(variables, weights)), upper=float(weights.sum() * 0.4))
    with pytest.raises(CancelledError):
        with search_session(unlimited=True, cancel=cancel, progress=lambda _: cancel.set()):
            model.solve(None)
    assert cancel.is_set()


def test_known_failed_extractor_cannot_reach_live_calls(tmp_path, monkeypatch):
    import asyncio

    from fpl_intelligence.research import EVIDENCE_VERSION, extract_with_fallback
    from fpl_intelligence.storage import now

    store = Store(tmp_path)
    store.put(
        "provider_capability",
        dict(provider="gemini", model="test", verified_until="2099-01-01T00:00:00+00:00"),
    )
    store.put(
        "extraction_benchmark",
        {
            "at": now(),
            "guard_version": EVIDENCE_VERSION,
            "results": [
                dict(
                    provider="gemini",
                    model="test",
                    success=True,
                    f1=1,
                    status_accuracy=0.8,
                    supported_spans=True,
                    injection_rejected=True,
                )
            ],
        },
    )

    async def forbidden(*_):
        pytest.fail("Known status-classification failure must be blocked before inference")

    monkeypatch.setattr("fpl_intelligence.research.extract", forbidden)
    with pytest.raises(ValueError, match="extraction quality"):
        asyncio.run(extract_with_fallback(store, {"text": "news"}, []))


def test_public_schema_has_responsive_default_and_retains_explicit_unlimited():
    body = dict(state_id="s", forecast_id="f", end_event=19)
    assert OptimizeRequest(**body).time_limit == 180
    assert OptimizeRequest(**body, time_limit=30).time_limit == 30
    assert OptimizeRequest(**body, time_limit=None).time_limit is None
    assert OptimizeRequest(**body, time_limit=1000).time_limit == 1000
    assert json.loads(Path("schemas/optimize.schema.json").read_text()) == OptimizeRequest.model_json_schema()


def test_extraction_cannot_pass_with_wrong_availability_categories():
    good = dict(success=True, f1=1, status_accuracy=1, supported_spans=True, injection_rejected=True)
    assert extraction_eligible(good)
    assert not extraction_eligible({**good, "status_accuracy": 5 / 6})
    assert not extraction_eligible({**good, "supported_spans": False})


def test_partial_start_labels_and_conflicting_duplicates_are_not_silently_accepted(monkeypatch):
    from fpl_intelligence.decision import canonical_rows

    rows = [
        dict(
            season="2025/26",
            element=i,
            fixture=1,
            round=1,
            team=team,
            was_home=home,
            team_h_score=1,
            team_a_score=0,
            minutes=90,
            total_points=2,
            starts=starts,
            kickoff_time="2025-08-01T15:00:00Z",
        )
        for i, team, home, starts in [(1, "A", True, None), (2, "B", False, 1)]
    ]
    monkeypatch.setattr(
        "fpl_intelligence.decision.load_market", lambda _: ({}, {"teams": [], "fixtures": [], "events": []})
    )
    monkeypatch.setattr("fpl_intelligence.observations.observation_view", lambda _: [dict(r) for r in rows])
    result, _, _ = canonical_rows(None)
    assert result[0]["start_observed"] is False
    assert result[1]["start_observed"] is True
    rows.append({**rows[1], "minutes": 80})
    with pytest.raises(ValueError, match="Conflicting historical player-fixture"):
        canonical_rows(None)
    rows[-1] = dict(rows[1])
    result, _, _ = canonical_rows(None)
    assert len(result) == 2


def test_database_liveness_is_not_account_readiness(tmp_path):
    store = Store(tmp_path)
    result = operational_readiness(store)
    assert not result["ready"] and "Missing forecast" in result["reasons"]


def test_completed_search_cannot_publish_after_market_or_model_changes(tmp_path):
    from fpl_intelligence.readiness import selection_is_current

    store = Store(tmp_path)
    state = store.put("state", {"event": 10})
    market = store.put("market", {"events": []})
    model = store.put("decision_model", {"version": "accepted"})
    forecast = store.put("forecast", {"market_id": market, "decision_model_id": model})
    plan = dict(state_id=state, market_id=market, forecast_id=forecast)
    assert selection_is_current(store, plan)
    store.put("decision_model", {"version": "new accepted"})
    assert not selection_is_current(store, plan)
    store.put("market", {"events": [], "new_prices": True})
    assert not selection_is_current(store, plan)


def test_completed_search_cannot_publish_after_its_deadline(tmp_path):
    from fpl_intelligence.readiness import selection_is_current

    store = Store(tmp_path)
    state = store.put("state", {"event": 10})
    market = store.put(
        "market",
        {
            "events": [
                {"id": 10, "deadline_time": "2000-01-01T00:00:00Z"},
                {"id": 11, "deadline_time": "2099-01-01T00:00:00Z"},
            ]
        },
    )
    model = store.put("decision_model", {"version": "accepted"})
    forecast = store.put("forecast", {"market_id": market, "decision_model_id": model})
    assert not selection_is_current(store, dict(state_id=state, market_id=market, forecast_id=forecast))


def test_installer_recovers_unfinished_transaction_on_retry(tmp_path):
    source, target = installation(tmp_path)
    backup = target / "artifacts/runtime/before-update-interrupted"
    (backup / "app").mkdir(parents=True)
    (backup / "app/app.py").write_text("original")
    (target / "app.py").write_text("partial-copy")
    journal = {"files": ["app.py"], "existed": ["app.py"], "status": "applying"}
    (backup / "transaction.json").write_text(json.dumps(journal))
    installer().install(source, target)
    assert json.loads((backup / "transaction.json").read_text())["status"] == "rolled_back"
    assert (target / "app.py").read_text() == "new"


def test_existing_account_cannot_be_overwritten_without_matching_version(
    tmp_path, market, rules, state, monkeypatch
):
    from fastapi.testclient import TestClient

    from fpl_intelligence.api import create_app

    app = create_app(tmp_path, job_runner=lambda s, i, c, work: work())
    store = app.state.store
    monkeypatch.setattr("fpl_intelligence.api.load_market", lambda *_: (market, {"source_hash": "x"}))
    monkeypatch.setattr(store, "latest", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr("fpl_intelligence.rules.Rules.from_bootstrap", lambda *_: rules)
    old = store.put("state", state.model_dump())
    client = TestClient(app, base_url="http://localhost")
    payload = {"state": state.model_dump()}
    assert client.post("/api/state", json=payload).status_code == 409
    payload["expected_state_id"] = old
    assert client.post("/api/state", json=payload).status_code == 200


def test_running_job_can_be_cancelled_without_blocking_serial_queue(tmp_path, monkeypatch):
    import time

    from fastapi.testclient import TestClient

    from fpl_intelligence.api import create_app
    from fpl_intelligence.search_control import check_cancelled

    entered = Event()

    def work(_, **kwargs):
        entered.set()
        until = time.monotonic() + 5
        while time.monotonic() < until:
            check_cancelled()
            time.sleep(0.01)
        pytest.fail("Cancellation was not propagated")

    monkeypatch.setattr("fpl_intelligence.workflow.refresh_cycle", work)
    client = TestClient(
        create_app(tmp_path, job_runner=lambda s, i, c, work: work()), base_url="http://localhost"
    )
    response = client.post("/api/refresh-cycle")
    assert response.status_code == 200
    identity = response.json()["job_id"]
    assert entered.wait(2)
    assert client.post(f"/api/jobs/{identity}/cancel").status_code == 200
    for _ in range(100):
        status = client.get(f"/api/jobs/{identity}").json()["status"]
        if status == "cancelled":
            break
        time.sleep(0.01)
    assert status == "cancelled"


def test_scheduler_recovers_after_unexpected_exception(tmp_path, monkeypatch):
    import fpl_intelligence.scheduling as module

    store = Store(tmp_path)
    store.put("refresh_schedule", {"enabled": True, "interval_minutes": 30})
    store.put("market", {"events": []})
    store.snapshot("bootstrap", "https://example.test/", {})
    with store.db() as db:
        db.execute("update snapshots set captured_at='2020-01-01T00:00:00+00:00'")
    original = store.list
    attempts = []

    def sometimes(kind, *args):
        if kind == "refresh_schedule":
            attempts.append(kind)
            if len(attempts) == 1:
                raise RuntimeError("temporary failure")
        return original(kind, *args)

    class Stop:
        calls = 0

        def wait(self, _):
            self.calls += 1
            return self.calls > 2

    monkeypatch.setattr(store, "list", sometimes)
    monkeypatch.setattr(
        module,
        "threading",
        SimpleNamespace(Event=Stop, Thread=lambda **kw: SimpleNamespace(start=kw["target"])),
    )
    launched = []
    module.start_scheduler(store, lambda *args: launched.append(args) or {"job_id": "ok"})
    assert len(attempts) == 2 and len(launched) == 1
    assert "temporary failure" in store.get("scheduler_issue")["reason"]
