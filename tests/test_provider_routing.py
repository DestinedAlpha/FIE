import asyncio
from datetime import datetime, timedelta, timezone

from fpl_intelligence.research import EVIDENCE_VERSION, FreeCapability, extract_with_fallback
from fpl_intelligence.storage import Store, now


def test_authentic_quote_quality_informs_routing_without_altering_saved_preference(tmp_path, monkeypatch):
    store = Store(tmp_path)
    until = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    for provider in ("gemini", "groq"):
        cap = FreeCapability(
            provider=provider,
            model=provider + "-model",
            verified_until=until,
            daily_calls=20,
            free_text_verified=True,
            unbilled_account_attested=True,
            terms_eligible=True,
        )
        store.put("provider_capability", cap.model_dump())
    preferred = store.put("provider_selection", {"provider": "gemini", "model": "gemini-model"})
    store.put(
        "authentic_extraction_benchmark",
        {
            "kind": "authentic_labeled_extraction_benchmark",
            "guard_version": EVIDENCE_VERSION,
            "at": now(),
            "results": [
                {"provider": "gemini", "model": "gemini-model", "passed_cases": 11, "total_cases": 16},
                {"provider": "groq", "model": "groq-model", "passed_cases": 16, "total_cases": 16},
            ],
        },
    )
    calls = []

    async def extract(s, source, cap, entities):
        calls.append(cap.provider)
        return ["grounded-claim"]

    monkeypatch.setattr("fpl_intelligence.research.extract", extract)
    report = asyncio.run(extract_with_fallback(store, {"text": "A verified quote."}, []))
    assert calls == ["groq"] and report["provider"] == "groq"
    assert store.get("provider_selection")["id"] == preferred


def test_obsolete_quality_and_expired_preference_cannot_block_a_current_free_route(tmp_path, monkeypatch):
    store = Store(tmp_path)
    until = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    for provider, model, expiry in [
        ("gemini", "retired", "2020-01-01T00:00:00Z"),
        ("groq", "current", until),
    ]:
        cap = FreeCapability(
            provider=provider,
            model=model,
            verified_until=expiry,
            daily_calls=20,
            free_text_verified=True,
            unbilled_account_attested=True,
            terms_eligible=True,
        )
        store.put("provider_capability", cap.model_dump())
    store.put("provider_selection", {"provider": "gemini", "model": "retired"})
    store.put(
        "extraction_benchmark",
        {
            "guard_version": "obsolete",
            "at": now(),
            "results": [{"provider": "groq", "model": "current", "success": True, "status_accuracy": 0}],
        },
    )
    calls = []

    async def extract(s, source, cap, entities):
        calls.append(cap.model)
        return []

    monkeypatch.setattr("fpl_intelligence.research.extract", extract)
    asyncio.run(extract_with_fallback(store, {"text": "current source"}, []))
    assert calls == ["current"]
