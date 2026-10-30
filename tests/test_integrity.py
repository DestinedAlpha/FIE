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
    assert alias("Joško Groß") == "joskogro"
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
    with TestClient(create_app(tmp_path), base_url="http://localhost") as client:
        assert client.get("/api/health").json()["integrity"] == "ok"
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
