import time
from threading import Event

from fastapi.testclient import TestClient

from fpl_intelligence.api import create_app
from fpl_intelligence.search_control import check_cancelled, effective_time_limit
from fpl_intelligence.storage import Store


def test_startup_refresh_is_visible_cancellable_and_does_not_block_http(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.put("state", {"event": 11})
    store.put("market", {})
    store.put("decision_model", {"promoted": True})
    selected = store.put("selected_plan", {"plan_id": "retained"})
    entered = Event()
    monkeypatch.setattr("fpl_intelligence.maintenance.maintain_models", lambda *_: None)
    monkeypatch.setattr("fpl_intelligence.readiness.operational_readiness", lambda *_: {"ready": False})

    def no_market(*_):
        raise ValueError("No normalized market in this isolated job test")

    monkeypatch.setattr("fpl_intelligence.api.load_market", no_market)

    def refresh(s):
        assert effective_time_limit(60) is None
        entered.set()
        expires = time.monotonic() + 5
        while not entered.wait(0.01) or time.monotonic() < expires:
            check_cancelled()
            entered.wait(0.01)
            time.sleep(0.01)

    monkeypatch.setattr("fpl_intelligence.workflow.refresh_cycle", refresh)
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as client:
        assert entered.wait(5)
        response = client.get("/api/dashboard")
        assert response.status_code == 200
        job = response.json()["active_jobs"][0]
        assert job["kind"] == "startup-refresh"
        assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = client.get(f"/api/jobs/{job['id']}").json()["status"]
            if status == "cancelled":
                break
            time.sleep(0.01)
        assert status == "cancelled"
    assert store.get("selected_plan")["id"] == selected
