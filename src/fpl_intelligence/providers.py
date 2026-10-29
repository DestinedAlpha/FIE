"""Independent provider setup: a failed optional key never aborts numerical startup."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .research import FreeCapability
from .settings import load_environment
from .storage import Store, now


def configure(store: Store, unbilled: bool, client=None) -> dict:
    if not unbilled:
        raise ValueError("Free, unbilled accounts must be explicitly confirmed")
    load_environment()
    results: list[dict[str, Any]] = []
    capabilities = []
    owned_client = client is None
    client = client or httpx.Client(timeout=20)
    try:
        for provider in ("gemini", "groq"):
            key = os.getenv("GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY", "").strip()
            if not key:
                results.append({"provider": provider, "status": "not_configured"})
                continue
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models"
                if provider == "gemini"
                else "https://api.groq.com/openai/v1/models"
            )
            headers = {"x-goog-api-key": key} if provider == "gemini" else {"Authorization": "Bearer " + key}
            try:
                response = client.get(url, headers=headers)
                if response.status_code != 200:
                    results.append(
                        {
                            "provider": provider,
                            "status": "unavailable",
                            "http_status": response.status_code,
                            "message": "Check the key and account access; the numerical planner remains available",
                        }
                    )
                    continue
                payload = response.json()
                catalog = {
                    m.get("id", m.get("name", "")).removeprefix("models/")
                    for m in payload.get("models", payload.get("data", []))
                }
                candidates = ("gemini-3.5-flash-lite",) if provider == "gemini" else ("openai/gpt-oss-120b",)
                matched = [m for m in candidates if m in catalog]
                if not matched:
                    results.append(
                        {
                            "provider": provider,
                            "status": "unavailable",
                            "message": "No supported free text model in the authenticated catalog",
                        }
                    )
                    continue
                for model in matched:
                    capability = FreeCapability(
                        provider=provider,
                        model=model,
                        free_text_verified=True,
                        unbilled_account_attested=True,
                        terms_eligible=True,
                        verified_until=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                        daily_calls=20,
                        daily_tokens=160000,
                        output_tokens=4000,
                    )
                    store.put("provider_capability", capability.model_dump())
                    capabilities.append({"provider": provider, "model": model})
                results.append({"provider": provider, "status": "authenticated", "models": matched})
            except (httpx.HTTPError, ValueError, TypeError, KeyError):
                results.append(
                    {
                        "provider": provider,
                        "status": "unavailable",
                        "message": "Authentication endpoint could not be verified; retry provider setup later",
                    }
                )
    finally:
        if owned_client:
            client.close()
    if capabilities:
        chosen = next((c for c in capabilities if c["provider"] == "gemini"), capabilities[0])
        store.put(
            "provider_selection",
            {
                **chosen,
                "at": now(),
                "basis": "authenticated eligible free text provider; benchmark remains separately reported",
            },
        )
    report = {
        "at": now(),
        "providers": results,
        "configured": capabilities,
        "numerical_planner_available": True,
        "paid_provider_calls_allowed": False,
        "key_values_included": False,
    }
    report["id"] = store.put("provider_setup", report)
    return report
