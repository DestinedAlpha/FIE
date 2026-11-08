"""Measured, bounded public-source requests with Retry-After and circuit recovery."""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from .search_control import check_cancelled
from .storage import now


def retry_seconds(value: str | None, attempt: int) -> float:
    if value:
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError):
                pass
    return 2**attempt + random.Random(attempt + 601).uniform(0, 0.25)


class PublicTransport:
    def __init__(self, store, concurrency=4, spacing=0.0):
        self.store = store
        self.semaphore = asyncio.Semaphore(concurrency)
        self.spacing = spacing
        self.lock = asyncio.Lock()
        self.last = 0.0
        self.failures: dict[str, int] = {}
        self.open_until: dict[str, float] = {}

    async def get(
        self,
        client,
        url,
        source,
        *,
        json_body=True,
        binary=False,
        return_response=False,
        allow_redirects=False,
        max_bytes=20_000_000,
    ):
        check_cancelled()
        host = urlsplit(url).hostname or ""
        if self.open_until.get(host, 0) > time.monotonic():
            raise ValueError(f"Source circuit open for {host}; last known snapshot retained")
        async with self.semaphore:
            check_cancelled()
            if self.open_until.get(host, 0) > time.monotonic():
                raise ValueError(f"Source circuit open for {host}; queued request deferred")
            for attempt in range(3):
                check_cancelled()
                async with self.lock:
                    pause = self.spacing - (time.monotonic() - self.last)
                    if pause > 0:
                        await asyncio.sleep(pause)
                    self.last = time.monotonic()
                started = time.monotonic()
                status, count, error = None, 0, None
                response = None
                latency = None
                try:
                    async with client.stream("GET", url) as response:
                        status = response.status_code
                        if status in (429, 500, 502, 503, 504):
                            response.raise_for_status()
                        if not (allow_redirects and response.is_redirect):
                            response.raise_for_status()
                        chunks = []
                        async for chunk in response.aiter_bytes():
                            count += len(chunk)
                            if count > max_bytes:
                                raise ValueError("Source response exceeds its byte budget")
                            chunks.append(chunk)
                        content = b"".join(chunks)
                    body: Any
                    if return_response:
                        # aiter_bytes already decoded the wire encoding. Keeping
                        # that header would decode plain HTML a second time.
                        headers = {
                            k: v
                            for k, v in response.headers.items()
                            if k.lower() not in {"content-encoding", "content-length"}
                        }
                        body = httpx.Response(
                            status_code=status,
                            headers=headers,
                            content=content,
                            request=response.request,
                        )
                    elif binary:
                        body = content
                    elif json_body:
                        import json

                        if content.lstrip().startswith(b"<"):
                            raise ValueError(
                                "Source returned HTML instead of JSON (including HTTP 200 error pages)"
                            )
                        body = json.loads(content)
                    else:
                        body = content.decode("utf-8-sig")
                    self.failures[host] = 0
                    return body
                except (httpx.HTTPError, ValueError) as exc:
                    latency = time.monotonic() - started
                    error = type(exc).__name__
                    self.failures[host] = self.failures.get(host, 0) + 1
                    if self.failures[host] >= 5:
                        self.open_until[host] = time.monotonic() + 300
                    transient = status is None or status in (429, 500, 502, 503, 504)
                    if not transient or attempt == 2 or self.open_until.get(host, 0) > time.monotonic():
                        raise
                    pause = retry_seconds(response.headers.get("Retry-After") if response else None, attempt)
                    if pause > 30:
                        self.open_until[host] = time.monotonic() + pause
                        raise ValueError(f"{host} requested a longer retry interval; deferred") from exc
                    await asyncio.sleep(pause)
                finally:
                    self.store.put(
                        "source_request",
                        {
                            "at": now(),
                            "source": source,
                            "host": host,
                            "attempt": attempt + 1,
                            "status": status,
                            "bytes": count,
                            "latency_seconds": latency if latency is not None else time.monotonic() - started,
                            "error_type": error,
                        },
                    )
        raise ValueError("Request exhausted its bounded attempts")


def request_metrics(store):
    from datetime import timedelta

    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    with store.db() as db:
        return [
            dict(r)
            for r in db.execute(
                "SELECT json_extract(body,'$.host') AS host,count(*) AS calls,"
                "sum(json_extract(body,'$.bytes')) AS bytes,"
                "avg(json_extract(body,'$.latency_seconds')) AS mean_latency_seconds,"
                "sum(CASE WHEN json_extract(body,'$.error_type') IS NOT NULL THEN 1 ELSE 0 END) AS errors "
                "FROM objects WHERE kind='source_request' AND created_at>=? GROUP BY host",
                (since,),
            )
        ]
