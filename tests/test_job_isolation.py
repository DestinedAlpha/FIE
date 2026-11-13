"""Actual process/HTTP tests; the blocked-process fixture never polls cancellation."""

import json
import subprocess
import sys
import time

from fastapi.testclient import TestClient

from fpl_intelligence.api import create_app
from fpl_intelligence.storage import Store


def wait_for(client, identity, expected, seconds=8):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        job = client.get(f"/api/jobs/{identity}").json()
        if job["status"] in expected:
            return job
        time.sleep(0.04)
    raise AssertionError(job)


def test_noncooperative_process_is_killed_and_queue_is_released(tmp_path, monkeypatch):
    import fpl_intelligence.job_execution as module

    original = subprocess.Popen
    processes = []

    def popen(command, **kwargs):
        # An actual sleeping native-style worker ignores every cancellation hook.
        if not processes:
            command = [sys.executable, "-c", "import time; time.sleep(60)"]
        process = original(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    store = Store(tmp_path)
    protected = {
        k: store.put(k, {"retained": k})
        for k in ("state", "market", "decision_model", "forecast", "selected_plan")
    }
    client = TestClient(create_app(tmp_path), base_url="http://127.0.0.1")
    first = client.post("/api/forecast", json={"worlds": 137}).json()["job_id"]
    limit = time.monotonic() + 5
    while not store.list("job_worker", 1) and time.monotonic() < limit:
        time.sleep(0.02)
    assert store.list("job_worker", 1)
    # A queued dependent request must not start after the user's cancellation.
    queued = client.post("/api/train", json={}).json()["job_id"]
    start = time.monotonic()
    assert client.post(f"/api/jobs/{first}/cancel").status_code == 200
    wait_for(client, first, {"cancelled"})
    assert time.monotonic() - start < 5
    assert processes[0].poll() is not None
    assert client.get(f"/api/jobs/{queued}").json()["status"] == "cancelled"
    assert {k: store.get(k)["id"] for k in protected} == protected
    # The next real worker reaches its requested route, rather than remaining
    # behind the killed process. Missing data fails explicitly in this fixture.
    second = client.post("/api/forecast", json={"worlds": 139}).json()["job_id"]
    result = wait_for(client, second, {"failed"}, seconds=30)
    assert "worker exited" not in result["error"].lower()
    request = json.loads(result["request"])
    assert request["_execution"]["path"] == "/api/forecast"
    assert request["_execution"]["payload"] == {"worlds": 139}
    assert client.post(f"/api/jobs/{first}/cancel").json()["status"] == "cancelled"


def test_quick_refresh_never_trains_forecasts_or_optimizes(tmp_path, monkeypatch):
    from fpl_intelligence.ingestion import Official
    from fpl_intelligence.workflow import refresh_data

    store = Store(tmp_path)
    store.put("state", {"entry_id": None})
    store.put("market", {"registered": True})
    seen = []

    async def ingest(self, **kwargs):
        seen.append(kwargs)
        return {"id": "official-data"}

    def forbidden(*args, **kwargs):
        raise AssertionError("A data/team refresh must not start numerical work")

    monkeypatch.setattr(Official, "ingest", ingest)
    monkeypatch.setattr(
        "fpl_intelligence.account_sync.reconcile_public", lambda *args: {"account_changed": False}
    )
    monkeypatch.setattr("fpl_intelligence.training.train_current", forbidden)
    monkeypatch.setattr("fpl_intelligence.simulation.forecast", forbidden)
    monkeypatch.setattr("fpl_intelligence.optimization.optimize", forbidden)
    result = refresh_data(store)
    assert seen == [{"entry": None, "histories": False, "auxiliary": False}]
    assert result["plan_optimized"] is False and result["forecast_updated"] is False


def test_completed_result_does_not_wait_for_blocked_interpreter_shutdown(tmp_path, monkeypatch):
    from threading import Event

    import fpl_intelligence.job_execution as module
    from fpl_intelligence.storage import atomic_json

    store = Store(tmp_path)
    original = subprocess.Popen
    processes = []

    def popen(command, **kwargs):
        # The work has finished; interpreter/native cleanup is stuck afterward.
        atomic_json(store.root / "jobs" / "completed" / "result.json", {"result": {"finished": True}})
        process = original([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    started = time.monotonic()
    assert module.run_isolated(store, "completed", Event()) == {"finished": True}
    assert time.monotonic() - started < 6
    assert processes[0].poll() is not None
    assert not (store.root / "jobs" / "completed").exists()


def test_durable_cancellation_reaches_the_owning_supervisor(tmp_path, monkeypatch):
    import fpl_intelligence.job_execution as module

    original = subprocess.Popen
    processes = []

    def popen(command, **kwargs):
        process = original([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    owner = TestClient(create_app(tmp_path), base_url="http://127.0.0.1")
    identity = owner.post("/api/train", json={}).json()["job_id"]
    store = Store(tmp_path)
    expires = time.monotonic() + 5
    while not store.list("job_worker", 1) and time.monotonic() < expires:
        time.sleep(0.02)
    assert store.list("job_worker", 1)
    # Another controller has no in-memory cancellation Event for this job.
    other = TestClient(create_app(tmp_path), base_url="http://127.0.0.1")
    assert other.get("/api/jobs/" + identity).json()["status"] == "running"
    other.post("/api/jobs/" + identity + "/cancel")
    wait_for(owner, identity, {"cancelled"})
    assert processes[0].poll() is not None


def test_only_one_live_api_owner_can_use_an_installation(tmp_path):
    import pytest

    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as first:
        identity = first.get("/api/runtime").json()["installation_id"]
        with pytest.raises(ValueError, match="already running"):
            with TestClient(create_app(tmp_path), base_url="http://127.0.0.1"):
                pass
        assert first.get("/api/runtime").json()["installation_id"] == identity


def test_search_budget_is_shared_but_exact_search_keeps_bounded_seeds():
    from fpl_intelligence.search_control import (
        auxiliary_search,
        effective_time_limit,
        planning_budget,
        search_session,
    )

    with search_session(True):
        assert effective_time_limit(1) is None
        with auxiliary_search():
            assert effective_time_limit(1) == 1
        assert effective_time_limit(1) is None
        with planning_budget(0.1):
            first = effective_time_limit(100)
            time.sleep(0.02)
            assert 0 < effective_time_limit(100) < first <= 0.100001
            with planning_budget(10):
                assert effective_time_limit(None) < first
        assert effective_time_limit(1) is None
