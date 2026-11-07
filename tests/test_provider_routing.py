import asyncio
from datetime import datetime, timedelta, timezone

from fpl_intelligence.research import FreeCapability, extract_with_fallback
from fpl_intelligence.storage import Store, now


def test_authentic_quote_quality_informs_routing_without_altering_saved_preference(tmp_path, monkeypatch):
    store = Store(tmp_path)
    until = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    for provider in ("gemini", "groq"):
        cap = FreeCapability(
            provider=provider, model=provider + "-model", verified_until=until, daily_calls=20
        )
        store.put("provider_capability", cap.model_dump())
    preferred = store.put("provider_selection", {"provider": "gemini", "model": "gemini-model"})
    store.put(
        "authentic_extraction_benchmark",
        {
            "kind": "authentic_labeled_extraction_benchmark",
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
