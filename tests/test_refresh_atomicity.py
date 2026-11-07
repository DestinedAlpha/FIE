import asyncio
import copy

import httpx
import pytest

from fpl_intelligence.ingestion import Official, normalize
from fpl_intelligence.storage import Store


def core():
    bootstrap = {
        "events": [{"id": 1, "deadline_time": "2026-08-14T17:00:00Z"}],
        "teams": [{"id": 1}, {"id": 2}],
        "element_types": [{"id": 2}],
        "elements": [
            {
                "id": 1,
                "code": 9,
                "first_name": "Test",
                "second_name": "Player",
                "web_name": "Player",
                "team": 1,
                "element_type": 2,
                "now_cost": 50,
                "selected_by_percent": "1.0",
                "status": "a",
            }
        ],
    }
    return bootstrap, [{"id": 1, "team_h": 1, "team_a": 2, "event": 1}]


def test_failed_partner_does_not_publish_new_core(tmp_path, monkeypatch):
    store = Store(tmp_path)
    bootstrap, fixtures = core()
    normalize(store, bootstrap, fixtures, publish_core=True)
    market = store.get("market")["id"]
    changed = copy.deepcopy(bootstrap)
    changed["elements"][0]["now_cost"] = 60
    adapter = Official(store)

    async def get(client, url, source):
        if source == "bootstrap":
            return changed
        raise httpx.ConnectError("fixtures unavailable")

    monkeypatch.setattr(adapter.transport, "get", get)
    with pytest.raises(ValueError, match="Core refresh failed"):
        asyncio.run(adapter.ingest(histories=False))
    assert store.get("market")["id"] == market
    assert store.latest("bootstrap")[0] == bootstrap
    assert store.latest("fixtures")[0] == fixtures


@pytest.mark.parametrize(
    "invalid", ["unknown_team", "duplicate_fixture", "invalid_ownership", "unknown_event"]
)
def test_invalid_core_cannot_replace_valid_market(tmp_path, invalid):
    store = Store(tmp_path)
    bootstrap, fixtures = core()
    normalize(store, bootstrap, fixtures, publish_core=True)
    market = store.get("market")["id"]
    old = copy.deepcopy(bootstrap)
    if invalid == "unknown_team":
        bootstrap["elements"][0]["team"] = 99
    elif invalid == "duplicate_fixture":
        fixtures.append(fixtures[0].copy())
    elif invalid == "unknown_event":
        fixtures[0]["event"] = 99
    else:
        bootstrap["elements"][0]["selected_by_percent"] = "nan"
    with pytest.raises(ValueError):
        normalize(store, bootstrap, fixtures, publish_core=True)
    assert store.get("market")["id"] == market
    assert store.latest("bootstrap")[0] == old


def test_core_registry_publication_rolls_back_together(tmp_path, monkeypatch):
    store = Store(tmp_path)
    bootstrap, fixtures = core()
    normalize(store, bootstrap, fixtures, publish_core=True)
    market = store.get("market")["id"]
    changed = copy.deepcopy(bootstrap)
    changed["elements"][0]["now_cost"] = 60

    def fail(*args, **kwargs):
        raise OSError("simulated failed market write")

    monkeypatch.setattr(store, "put", fail)
    with pytest.raises(OSError):
        normalize(store, changed, fixtures, publish_core=True)
    assert store.get("market")["id"] == market
    assert store.latest("bootstrap")[0] == bootstrap
    with store.db() as db:
        assert db.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 2
