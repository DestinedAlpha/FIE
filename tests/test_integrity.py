from contextlib import closing
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fpl_intelligence.api import create_app
from fpl_intelligence.features import profile
from fpl_intelligence.ingestion import alias, resolve_name
from fpl_intelligence.optimization import MIP
from fpl_intelligence.research import FreeCapability, reserve_quota, validate_url
from fpl_intelligence.storage import Store


def test_raw_snapshots_are_hashed_and_temporal(tmp_path):
    store = Store(tmp_path)
    first = store.snapshot("test", "https://example.test/", {"v": 1})
    second = store.snapshot("test", "https://example.test/", {"v": 2})
    assert first["hash"] != second["hash"]
    assert store.latest("test", first["captured_at"])[0] == {"v": 1}
    assert store.latest("test")[0] == {"v": 2}
    store.backup(tmp_path / "backup.sqlite")
    import sqlite3

    with closing(sqlite3.connect(tmp_path / "backup.sqlite")) as db:
        assert db.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 2
    (tmp_path / "backup.sqlite").unlink()  # Backups must release Windows file handles.


def test_numeric_mapping_keys_roundtrip_hash(tmp_path):
    store = Store(tmp_path)
    store.snapshot("numeric", "https://example.test/", {1: "a", 2: "b", 10: "c"})
    assert store.latest("numeric")[0] == {"1": "a", "2": "b", "10": "c"}


@pytest.mark.parametrize(
    "url",
    [
        "http://premierleague.com/x",
        "https://127.0.0.1/",
        "https://169.254.169.254/",
        "file:///etc/passwd",
        "https://premierleague.com@evil.test/",
        "https://evil.premierleague.com/",
        "https://premierleague.com:8443/",
        "https://premierleague.com.evil.test/",
    ],
)
def test_source_fetch_rejects_untrusted_urls(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_free_guard_fail_closed_and_quota(tmp_path):
    store = Store(tmp_path)
    until = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    cap = FreeCapability(provider="gemini", model="test", verified_until=until, daily_calls=1)
    with pytest.raises(ValueError, match="blocked"):
        reserve_quota(store, cap, 100)
    cap = cap.model_copy(
        update={"free_text_verified": True, "unbilled_account_attested": True, "terms_eligible": True}
    )
    reserve_quota(store, cap, 100)
    with pytest.raises(ValueError, match="quota exhausted"):
        reserve_quota(store, cap, 100)


def test_missing_defensive_actions_are_not_observed_zero():
    p = profile(2, [{"minutes": 90, "starts": 1, "expected_goals": "0.1", "expected_assists": "0.1"}])
    assert p["action_coverage"] == 0
    assert p["actions90"] > 0


def test_accent_alias_and_ambiguous_identity(market):
    assert alias("Joško Groß") == "joskogross"
    market[1] = market[1].model_copy(update={"name": "Alex Palmer", "web_name": "Palmer"})
    market[8] = market[8].model_copy(update={"name": "Cole Palmer", "web_name": "Palmer"})
    assert resolve_name("Alex Palmer", 1, market) == 1


def test_mip_tiny_knapsack_oracle():
    from itertools import product

    values = [4, 7, 8, 9]
    costs = [2, 4, 5, 6]
    mip = MIP()
    variables = [mip.var(i, cost=v) for i, v in enumerate(values)]
    mip.row({i: c for i, c in zip(variables, costs)}, upper=9)
    solution, info = mip.solve(3)
    oracle = max(
        sum(v * x for v, x in zip(values, choice))
        for choice in product((0, 1), repeat=4)
        if sum(c * x for c, x in zip(costs, choice)) <= 9
    )
    assert round(info["objective"]) == oracle
    assert np.dot(solution, costs) <= 9


def test_api_empty_onboarding_and_invalid_jobs(tmp_path):
    from importlib.metadata import version

    with TestClient(
        create_app(tmp_path, job_runner=lambda s, i, c, work: work()), base_url="http://localhost"
    ) as client:
        assert client.get("/openapi.json").json()["info"]["version"] == version("fpl-intelligence")
        health = client.get("/api/health").json()
        assert health["status"] == "ok" and health["integrity"] in ("ok", "not_checked")
        assert client.get("/api/dashboard").json()["market"] is None
        assert client.get("/api/jobs/missing").status_code == 404
        assert client.post("/api/forecast", json={"end_event": 99}).status_code == 422
        assert client.post("/api/state", json={"state": {"lots": []}}).status_code == 422
        assert client.post("/api/refresh", headers={"Origin": "https://evil.test"}).status_code == 403
        assert client.get("/api/health", headers={"Host": "evil.test"}).status_code == 400
        local = Store(tmp_path)
        scenario_id = local.put("scenario_plan", {"scenario": {"name": "Hypothesis"}, "ledger": []})
        assert client.get(f"/api/plans/{scenario_id}").json()["scenario"]["name"] == "Hypothesis"
        assert client.get(f"/api/export/{scenario_id}/json").status_code == 200
        assert local.list("plan", 1) == []  # Reading a hypothesis never selects a production plan.


@pytest.mark.parametrize(
    "reference",
    [
        r"C:\Users\user\FIE\data\models\base\minutes.txt",
        r"\\server\old\models\base\minutes.txt",
        "/old/FIE/data/models/base/minutes.txt",
    ],
)
def test_relocated_weights_accept_both_platform_path_formats(tmp_path, reference):
    store = Store(tmp_path)
    weights = tmp_path / "models/base/minutes.txt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"retained weights")
    assert store.artifact_path(reference).read_bytes() == b"retained weights"


def test_windows_metadata_cannot_traverse_the_relocated_model_directory(tmp_path):
    with pytest.raises(ValueError, match="Unrecognized"):
        Store(tmp_path).artifact_path(r"C:\old\models\..\secrets.txt")


def test_health_probe_uses_cached_integrity_without_rescanning_the_database(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from fpl_intelligence.storage import now

    app = create_app(tmp_path)
    store = app.state.store
    with store.db() as db:
        result = db.execute("PRAGMA quick_check").fetchone()[0]
    store.put("database_integrity", {"at": now(), "result": result})
    original = store.db
    statements = []

    @contextmanager
    def traced():
        with original() as db:
            db.set_trace_callback(statements.append)
            yield db

    monkeypatch.setattr(store, "db", traced)
    with TestClient(app, base_url="http://localhost") as client:
        result = client.get("/api/health").json()
        assert result["status"] == "ok" and result["integrity"] == "ok"
        assert result["integrity_checked_at"]
        statements.clear()
        assert client.get("/api/health").json()["status"] == "ok"
        assert not any("quick_check" in statement.lower() for statement in statements)
        store.put("database_integrity", {"at": now(), "result": "corrupt test fixture"})
        assert client.get("/api/health").json()["status"] == "degraded"


def test_latest_artifact_lookup_preserves_ties_without_sorting_large_bodies(tmp_path):
    store = Store(tmp_path)
    first = store.put("large_artifact", {"value": 1})
    second = store.put("large_artifact", {"value": 2})
    with store.db() as db:
        db.execute("UPDATE objects SET created_at='2026-01-01T00:00:00Z' WHERE kind='large_artifact'")
        plan = db.execute(
            "EXPLAIN QUERY PLAN SELECT body,id FROM objects WHERE kind=? ORDER BY created_at DESC,rowid DESC LIMIT 1",
            ("large_artifact",),
        ).fetchall()
    assert store.get("large_artifact")["id"] == second
    assert [r["id"] for r in store.list("large_artifact")] == [second, first]
    assert not any("TEMP B-TREE" in str(tuple(r)) for r in plan)
