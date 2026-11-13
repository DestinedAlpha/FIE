"""Windows-spawn-safe entry point; reuse API handlers without a second job queue."""

from __future__ import annotations

import json
import sys
from concurrent.futures import CancelledError
from pathlib import Path

from .job_execution import FileCancellation, record_progress
from .search_control import search_session
from .settings import redact
from .storage import Store, atomic_json


def run(root, identity):
    store = Store(root)
    with store.db() as db:
        row = db.execute("SELECT request FROM jobs WHERE id=?", (identity,)).fetchone()
    if row is None:
        raise ValueError("Unknown calculation job")
    request = json.loads(row["request"])
    directory = store.root / "jobs" / identity
    result = directory / "result.json"
    cancel = FileCancellation(directory / "cancel")
    try:
        with search_session(
            request.get("_unlimited", False), cancel, lambda info: record_progress(store, identity, info)
        ):
            if request["kind"] == "startup-refresh":
                from .api import initialize_store

                value = initialize_store(store)
            else:
                from fastapi.testclient import TestClient

                from .api import create_app

                execution = request.get("_execution")
                if execution is None:
                    if request["kind"] != "scheduled-refresh":
                        raise ValueError("Calculation route unavailable; no other operation was substituted")
                    execution = {"path": "/api/refresh-cycle", "payload": {}}
                # No lifespan: the supervisor already owns startup/recovery/scheduling.
                client = TestClient(
                    create_app(
                        root,
                        _worker=True,
                        _job_inputs=request.get("input_versions"),
                        _worker_job={"id": identity, "unlimited": request.get("_unlimited", False)},
                    ),
                    base_url="http://127.0.0.1",
                )
                response = client.post(execution["path"], json=execution["payload"])
                if response.status_code >= 400:
                    raise ValueError(
                        response.json().get("detail", f"Calculation failed: {response.status_code}")
                    )
                value = response.json()
            if cancel.is_set():
                raise CancelledError()
            atomic_json(result, {"result": value})
    except CancelledError:
        atomic_json(result, {"cancelled": True})
    except BaseException as error:
        atomic_json(result, {"error": redact(f"{type(error).__name__}: {error}")[:600]})
        raise SystemExit(1)


def watch_parent(parent_pid):
    """No blocking stdin reads: closing buffered stdin can deadlock on Windows."""
    import os
    import time

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        handle = kernel.OpenProcess(0x00100000, False, parent_pid)
        if not handle:
            os._exit(2)
        if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) == 0:
            os._exit(2)
    else:
        while os.getppid() == parent_pid:
            time.sleep(0.5)
        os._exit(2)


if __name__ == "__main__":
    import os
    from threading import Thread

    Thread(
        target=watch_parent, args=(int(sys.argv[3]) if len(sys.argv) > 3 else os.getppid(),), daemon=True
    ).start()
    run(Path(sys.argv[1]), sys.argv[2])
