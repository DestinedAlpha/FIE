import json

import pytest

from fpl_intelligence.independent_sources import compare_fivestat, parse_fivestat
from fpl_intelligence.storage import Store


def packet(market):
    player = market[1]
    context = {
        "teams": [
            {"id": player.team, "name": "Arsenal", "short_name": "ARS"},
            {"id": 20, "name": "Liverpool", "short_name": "LIV"},
        ],
        "fixtures": [{"id": 77, "event": 11, "team_h": player.team, "team_a": 20}],
    }
    payload = [
        {
            "fpl_id": player.id,
            "name": player.web_name,
            "team": "Arsenal",
            "position": "GK",
            "fixtures": [
                {
                    "gw": 11,
                    "label": "LIV (H)",
                    "proj_xg": 0.0,
                    "proj_xa": 0.0,
                    "proj_cs": 0.3,
                    "expected_minutes": 90,
                    "proj_fpl": 4.5,
                }
            ],
        }
    ]
    return context, payload


def parse(payload, market, context):
    return parse_fivestat("<script>const capData = " + json.dumps(payload) + ";</script>", market, context)


def test_public_projection_identity_fixture_units_and_missing_coverage(market):
    context, payload = packet(market)
    result = parse(payload, market, context)
    assert len(result["records"]) == 1 and result["records"][0]["fixture_id"] == 77
    assert result["records"][0]["minutes"] == 90 and result["records"][0]["clean_sheet"] == 0.3
    payload[0]["name"] = "Different Player"
    with pytest.raises(ValueError, match="No FiveStat projections match"):
        parse(payload, market, context)


@pytest.mark.parametrize("change", ["duplicate", "bad_minutes", "bad_fixture", "nonfinite"])
def test_invalid_provider_projection_never_becomes_a_verified_record(market, change):
    context, payload = packet(market)
    row = payload[0]["fixtures"][0]
    if change == "duplicate":
        payload.append(payload[0])
    elif change == "bad_minutes":
        row["expected_minutes"] = 100
    elif change == "bad_fixture":
        row["label"] = "EVE (H)"
    else:
        row["proj_xg"] = float("nan")
    with pytest.raises(ValueError):
        parse(payload, market, context)


def test_capture_after_deadline_is_never_backdated_for_independent_comparison(tmp_path, market):
    store = Store(tmp_path)
    context, payload = packet(market)
    result = parse(payload, market, context)
    mid = store.put("market", {**context, "events": [{"id": 11, "deadline_time": "2026-11-07T11:00:00Z"}]})
    source = {**result, "season": "2026/27", "captured_at": "2026-11-07T11:01:00Z"}
    store.put("independent_projections", source)
    forecast = {
        "season": "2026/27",
        "information_cutoff": "2026-11-07T11:02:00Z",
        "market_id": mid,
        "fixtures": [{"player_id": 1, "event": 11, "fixture_id": 77, "mean": 4, "minutes": 88}],
    }
    assert not compare_fivestat(store, forecast)["records"]
    forecast["information_cutoff"] = "2026-11-07T10:59:00Z"
    assert compare_fivestat(store, forecast)["status"] == "Capture outside forecast information window"
