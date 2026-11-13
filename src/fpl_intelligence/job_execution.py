"""Supervise disposable calculation processes, including non-cooperative native code."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from concurrent.futures import CancelledError
from contextvars import ContextVar
from pathlib import Path
from typing import Any

request_context: ContextVar[dict | None] = ContextVar("fpl_job_request", default=None)


class FileCancellation:
    def __init__(self, path):
        self.path = path

    def is_set(self):
        return self.path.exists()


def record_progress(store, identity, info):
    import math

    from .storage import now

    values = {"job_id": identity, "at": now()}
    phase = getattr(info, "phase", None)
    if phase:
        store.put("job_phase", values | {"phase": phase})
    plan_id = getattr(info, "plan_id", None)
    if plan_id:
        store.put("job_phase", values | {"phase": phase, "plan_id": plan_id})
    for key in ("mip_primal_bound", "mip_dual_bound", "mip_gap", "running_time"):
        value = getattr(info, key, None)
        if value is not None and math.isfinite(value):
            values[key] = value
    if len(values) > 2:
        store.put("solver_progress", values)


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=3,
            )
        except (subprocess.TimeoutExpired, OSError):
            process.kill()
    else:
        try:
            getattr(os, "killpg")(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            except ProcessLookupError:
                pass
        process.wait(timeout=2)


def run_isolated(store, identity, cancel, work=None):
    """Return only after completion or confirmed worker termination, never a fake cancel."""
    if cancel.is_set():
        raise CancelledError("Calculation cancelled before starting")
    directory = (store.root / "jobs" / identity).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    marker, result = directory / "cancel", directory / "result.json"
    source = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    options: dict[str, Any] = (
        {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fpl_intelligence.job_worker",
            str(store.root.resolve()),
            identity,
            str(os.getpid()),
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **options,
    )
    from .storage import now

    try:
        store.put("job_worker", {"job_id": identity, "pid": process.pid, "at": now()})
        store.put("job_phase", {"job_id": identity, "phase": "Starting calculation worker", "at": now()})
        while process.poll() is None:
            # The atomic result is written only after the operation has ended.
            # Native-library or interpreter teardown must not hold the UI hostage.
            if result.is_file():
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    stop_process(process)
                break
            if cancel.wait(0.05) or marker.exists():
                marker.touch(exist_ok=True)
                try:
                    process.wait(timeout=0.75)
                except subprocess.TimeoutExpired:
                    stop_process(process)
                raise CancelledError("Calculation stopped")
        if cancel.is_set() or marker.exists():
            raise CancelledError("Calculation stopped")
        if not result.is_file():
            raise RuntimeError(
                f"Calculation worker exited ({process.returncode}); previous accepted artifacts retained"
            )
        body = json.loads(result.read_text())
        if "error" in body:
            raise RuntimeError(body["error"])
        if body.get("cancelled"):
            raise CancelledError("Calculation stopped")
        return body["result"]
    finally:
        stop_process(process)
        if process.stdin:
            process.stdin.close()
        # Job state/progress stays in SQLite; no request/results or marker litter.
        for path in directory.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
        directory.rmdir()
