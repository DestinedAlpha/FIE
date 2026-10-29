"""Private configuration loading and centralized secret-safe diagnostics."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv


def load_environment(path: Path | None = None) -> None:
    candidate = path or Path.cwd() / ".env"
    if candidate.is_file():
        load_dotenv(candidate, override=False)


def redact(value: object) -> str:
    text = str(value)
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        secret = os.getenv(name)
        if secret:
            text = text.replace(secret, "[redacted]")
    return re.sub(r"(?:gsk_|AIza|AQ\.)[A-Za-z0-9_.-]{15,}", "[redacted]", text)


def provider_configuration() -> dict:
    load_environment()
    return {
        "gemini_key_present": bool(os.getenv("GEMINI_API_KEY")),
        "groq_key_present": bool(os.getenv("GROQ_API_KEY")),
        "paid_provider_calls_allowed": False,
    }
