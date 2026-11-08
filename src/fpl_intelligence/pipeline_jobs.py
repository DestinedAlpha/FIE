"""One cross-process writer for expensive historical replay artifacts."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def replay_lock(path, busy_message="The integrated replay is already running; inspect its progress"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if path.stat().st_size == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(busy_message) from error
        try:
            yield
        finally:
            if sys.platform == "win32":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_pipeline_job(store, **options):
    from .pipeline_replay import run_pipeline_replay

    with replay_lock(store.root / "pipeline-replay.lock"):
        if store.get("decision_model").get("future_horizon_model_id"):
            from .future_policy_jobs import run_future_suite

            return run_future_suite(store, **options)
        return run_pipeline_replay(store, **options)
