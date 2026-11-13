import copy
from types import SimpleNamespace

from fpl_intelligence.readiness import decision_market_signature, forecast_market_current
from fpl_intelligence.storage import Store


def test_live_flow_reuse_keeps_prices_news_rules_and_replay_guards(tmp_path, monkeypatch):
    s = Store(tmp_path)
    monkeypatch.setattr("fpl_intelligence.readiness.now", lambda: "2026-11-13T10:30:00+00:00")
    monkeypatch.setattr(
        "fpl_intelligence.rules.Rules.from_bootstrap", lambda b: SimpleNamespace(hash=b["rules_hash"])
    )
    source = s.snapshot("bootstrap", "https://example.test", {"rules_hash": "r"})
    market = {
        "season": "2026/27",
        "players": [
            {
                "id": 1,
                "price": 50,
                "ownership": 10,
                "raw": {"now_cost": 50, "transfers_in_event": 10, "news": "", "minutes": 900},
            }
        ],
        "teams": [],
        "events": [],
        "fixtures": [],
        "source_hash": source["hash"],
    }
    old = s.put("market", market)
    f = {"market_id": old, "information_cutoff": "2026-11-13T10:00:00+00:00", "rules_hash": "r"}
    fresh = copy.deepcopy(market)
    fresh["players"][0]["raw"]["transfers_in_event"] = 11
    fresh["players"][0]["raw"]["price_change_hourly_rate"] = 90
    fresh["players"][0]["raw"]["price_change_locked_until"] = "2026-11-13T10:40:00Z"
    fresh["players"][0]["ownership"] = 10.1
    fresh["id"] = s.put("market", fresh)
    assert forecast_market_current(s, f, fresh)
    assert decision_market_signature(fresh) == decision_market_signature(market)
    for field, value in [("now_cost", 51), ("news", "Injured"), ("minutes", 990)]:
        changed = copy.deepcopy(fresh)
        changed["players"][0]["raw"][field] = value
        assert not forecast_market_current(s, f, changed)
    assert not forecast_market_current(s, {**f, "rules_hash": "changed"}, fresh)
    assert not forecast_market_current(s, {**f, "information_cutoff": "2026-11-13T09:00:00+00:00"}, fresh)
    s.put("replay_context", {"information_cutoff": f["information_cutoff"]})
    assert not forecast_market_current(s, f, fresh)
    assert forecast_market_current(s, f, {"id": old})
