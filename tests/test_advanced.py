import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fpl_intelligence.domain import Lot
from fpl_intelligence.features import profile
from fpl_intelligence.open_models import verify_models
from fpl_intelligence.research import Claim, ExtractedClaim, FreeCapability, extract, request_claims
from fpl_intelligence.scenarios import ScenarioSpec, sale_bounds
from fpl_intelligence.settings import redact
from fpl_intelligence.simulation import sample_lineup
from fpl_intelligence.storage import Store, atomic_json, digest


@given(st.integers(35, 160), st.integers(35, 180), st.integers(30, 200))
def test_unknown_acquisition_price_range_contains_actual_purchase(purchase, current, future):
    from fpl_intelligence.rules import selling_price

    actual = selling_price(purchase, current)
    lot = Lot(player_id=1, sell_price=actual, quote_market_price=current)
    low, high = sale_bounds(lot, current, future)
    assert low <= selling_price(purchase, future) <= high
    assert 0 <= low <= high <= future


def test_scenario_rejects_information_from_future():
    with pytest.raises(ValueError, match="precede"):
        ScenarioSpec(
            name="Unavailable",
            reveal_event=12,
            players=[{"player_id": 1, "from_event": 10, "availability": 0}],
        )
    with pytest.raises(ValueError, match="Unbounded"):
        ScenarioSpec(
            name="Invalid",
            reveal_event=10,
            players=[{"player_id": 1, "from_event": 10, "parameters": {"minutes_cap": 1000}}],
        )


def test_future_injury_override_never_decays_into_availability(market):
    members = [market[p] for p in range(1, 14)]
    profiles = {p.id: profile(p.position, []) for p in members}
    profiles[13]["availability_override"] = 0
    start, end, quality = sample_lineup(
        np.random.default_rng(31), members, profiles, 300, 20, np.zeros((300, len(members)))
    )
    assert np.all(end[:, 12] == 0)
    profiles[13]["availability_override"] = 1
    profiles[13]["minutes_cap"] = 45
    profiles[13]["p_start"] = 0.999
    start, end, _ = sample_lineup(
        np.random.default_rng(31), members, profiles, 300, 20, np.zeros((300, len(members)))
    )
    assert np.all(end[:, 12] - start[:, 12] <= 45)


def test_secret_safe_diagnostics(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_this_is_a_private_example_credential")
    result = redact("Failure: gsk_this_is_a_private_example_credential")
    assert result == "Failure: [redacted]"


def test_open_numerical_artifact_tampering_is_rejected(tmp_path):
    root = tmp_path / "vendor"
    (root / "models").mkdir(parents=True)
    path = root / "models" / "numeric.json"
    path.write_text("{}")
    atomic_json(
        root / "models" / "artifact_manifest.json",
        {"files": {"numeric.json": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}},
    )
    verify_models(root)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="checksum"):
        verify_models(root)


def test_stale_source_rejected_before_provider_dispatch(tmp_path):
    known = datetime.now(timezone.utc)
    source = {
        "text": "Public football text",
        "known_at": known.isoformat(),
        "published_at": (known - timedelta(days=12)).isoformat(),
    }
    cap = FreeCapability(
        provider="gemini", model="test", verified_until=(known + timedelta(days=1)).isoformat()
    )
    with pytest.raises(ValueError, match="stale"):
        asyncio.run(request_claims(Store(tmp_path), source, cap, []))


def test_provider_claim_type_contract_is_enumerated():
    with pytest.raises(ValueError):
        ExtractedClaim(
            player_id=1,
            claim_type="availability",
            statement="Available now",
            evidence_span="Available now",
            status="confirmed",
            origin="source",
        )


def test_extraction_batch_is_atomic(tmp_path, market, monkeypatch):
    import fpl_intelligence.research as module

    monkeypatch.setattr("fpl_intelligence.ingestion.load_market", lambda store: (market, {}))
    store = Store(tmp_path)
    known = datetime.now(timezone.utc)
    source = {
        "url": "https://www.arsenal.com/news",
        "text": "Player 1 is unavailable.",
        "known_at": known.isoformat(),
    }
    valid = Claim(
        player_id=1,
        claim_type="unavailable",
        statement="Player 1 is unavailable",
        evidence_span="Player 1 is unavailable.",
        source_url=source["url"],
        source_hash=digest(source["text"]),
        known_at=known.isoformat(),
        expires_at=(known + timedelta(days=1)).isoformat(),
        status="reported",
        origin="test",
    )

    async def fake(*args):
        return [valid, valid.model_copy(update={"evidence_span": "Invented claim"})]

    monkeypatch.setattr(module, "request_claims", fake)
    cap = FreeCapability(
        provider="gemini", model="test", verified_until=(known + timedelta(days=1)).isoformat()
    )
    with pytest.raises(ValueError, match="verbatim"):
        asyncio.run(extract(store, source, cap, [{"id": 1}]))
    assert store.list("claim") == []


def test_confirmed_account_not_replaced_by_old_seed(tmp_path, state):
    from fpl_intelligence.ingestion import reconcile_seed

    store = Store(tmp_path)
    store.put("state", state.model_dump())
    with pytest.raises(ValueError, match="confirmed account"):
        reconcile_seed(store, {})


def test_adaptive_scenario_keeps_original_decisions_before_observation(
    tmp_path, market, state, rules, monkeypatch
):
    import fpl_intelligence.scenarios as module
    from fpl_intelligence.domain import Action
    from fpl_intelligence.rules import best_lineup, transition
    from fpl_intelligence.scenarios import action_from_row, rollout

    store = Store(tmp_path)
    state_id = store.put("state", state.model_dump())
    rows = []
    current = state
    for event in range(9, 12):
        lineup = best_lineup(list(range(1, 16)), dict.fromkeys(market, 3.0), market, rules)
        current, row = transition(current, Action(lineup=lineup), market, rules)
        rows.append(row)
    frozen = {"id": "market", "source_hash": "bootstrap"}
    monkeypatch.setattr(module, "load_market", lambda *args: (market, frozen))
    monkeypatch.setattr(store, "latest", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(module.Rules, "from_bootstrap", lambda *args: rules)
    monkeypatch.setattr(
        module, "score_plan", lambda store, plan, forecast: plan.update(simulation={"mean": 0})
    )
    forecast = {
        "id": "forecast",
        "players": [
            {"event": e, "player_id": p, "mean": 3 if p < 16 else 2} for e in range(9, 12) for p in market
        ],
    }
    base = {"state_id": state_id, "market_id": "market", "start_event": 9, "end_event": 11, "ledger": rows}
    result = rollout(store, base, forecast, ScenarioSpec(name="Observed change", reveal_event=11))
    assert [action_from_row(r) for r in result["ledger"][:2]] == [action_from_row(r) for r in rows[:2]]
    assert all(r["bank_after"] >= 0 for r in result["ledger"])
    assert result["validation"]["shared_prefix"]


def test_partial_scenario_probabilities_rejected_before_simulation(tmp_path):
    from fpl_intelligence.scenarios import evaluate_scenarios

    store = Store(tmp_path)
    market_id = store.put("market", {})
    forecast_id = store.put("forecast", {})
    plan_id = store.put("plan", {"market_id": market_id, "forecast_id": forecast_id})
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_scenarios(
            store,
            plan_id,
            [
                ScenarioSpec(name="First branch", reveal_event=10, probability=0.5),
                ScenarioSpec(name="Second branch", reveal_event=10),
            ],
        )
