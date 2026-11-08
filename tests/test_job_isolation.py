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
    result = wait_for(client, second, {"failed"})
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
