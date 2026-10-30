"""Local opt-in refresh cadence; shares the serial API job queue."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from .storage import now


def start_scheduler(store, launch):
    stop = threading.Event()

    def run():
        while not stop.wait(30):
            try:
                if not store.list("refresh_schedule", 1):
                    continue
                settings = store.get("refresh_schedule")
                if not settings.get("enabled"):
                    continue
                with store.db() as db:
                    if db.execute(
                        "SELECT 1 FROM jobs WHERE status IN ('queued','running') LIMIT 1"
                    ).fetchone():
                        continue
                _, metadata = store.latest("bootstrap")
                market = store.get("market")
                at = datetime.now(timezone.utc)
                event = next((e for e in market["events"] if e.get("is_next")), None)
                interval = settings.get("interval_minutes", 240)
                if event:
                    remaining = (
                        datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00")) - at
                    ).total_seconds() / 3600
                    if 0 < remaining <= 1:
                        interval = min(interval, 5)
                    elif 0 < remaining <= 6:
                        interval = min(interval, 30)
                age = (
                    at - datetime.fromisoformat(metadata["captured_at"].replace("Z", "+00:00"))
                ).total_seconds() / 60
                if age < interval:
                    continue
                if store.list("scheduled_refresh", 1):
                    previous = store.get("scheduled_refresh")
                    since = (at - datetime.fromisoformat(previous["at"])).total_seconds() / 60
                    if since < min(interval, 15):
                        continue
                from .workflow import refresh_cycle

                job = launch("scheduled-refresh", {}, lambda: refresh_cycle(store))
                store.put(
                    "scheduled_refresh", {"at": now(), "job_id": job["job_id"], "interval_minutes": interval}
                )
            except ValueError as error:
                store.put("scheduler_issue", {"at": now(), "reason": str(error)[:240]})

    thread = threading.Thread(target=run, name="fpl-refresh-cadence", daemon=True)
    thread.start()
    return stop, thread
