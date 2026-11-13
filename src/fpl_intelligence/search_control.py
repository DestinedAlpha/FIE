"""Scoped live search policy; historical experiments retain explicit budgets."""

from concurrent.futures import CancelledError
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

control: ContextVar[Any] = ContextVar("fpl_search_control", default=None)
deadline: ContextVar[Any] = ContextVar("fpl_planning_deadline", default=None)


@contextmanager
def planning_budget(seconds):
    """Share one interactive search budget across seeds and counterfactuals."""
    import time

    previous = deadline.get()
    expires = None if seconds is None else time.monotonic() + seconds
    if previous is not None:
        expires = previous if expires is None else min(previous, expires)
    token = deadline.set(expires)
    try:
        yield
    finally:
        deadline.reset(token)


@contextmanager
def auxiliary_search():
    """Warm starts retain their small budget even during an unlimited main solve."""
    current = control.get()
    with search_session(False, current[1] if current else None, current[2] if current else None):
        yield


@contextmanager
def search_session(unlimited=False, cancel=None, progress=None):
    token = control.set((unlimited, cancel, progress))
    try:
        check_cancelled()
        yield
    finally:
        control.reset(token)


def check_cancelled():
    current = control.get()
    if current and current[1] is not None and current[1].is_set():
        raise CancelledError("Calculation cancelled")


def report_phase(phase, **details):
    from types import SimpleNamespace

    check_cancelled()
    current = control.get()
    if current and current[2]:
        current[2](SimpleNamespace(phase=phase, **details))


@contextmanager
def live_search():
    current = control.get()
    with search_session(True, current[1] if current else None, current[2] if current else None):
        yield


def effective_time_limit(seconds):
    import time

    current = control.get()
    value = None if seconds is None or bool(current and current[0]) else seconds
    expires = deadline.get()
    if expires is not None:
        remaining = max(0.001, expires - time.monotonic())
        value = remaining if value is None else min(value, remaining)
    return value


def configure_solver(solver, seconds, gap):
    import time

    current = control.get()
    seconds = effective_time_limit(seconds)
    unlimited = seconds is None
    solver.setOptionValue("time_limit", float("inf") if unlimited else seconds)
    solver.setOptionValue("mip_rel_gap", 0.0 if unlimited else gap)
    if unlimited:
        solver.setOptionValue("mip_abs_gap", 0.0)
    last = [0.0]

    def callback(event):
        if current and current[1] is not None and current[1].is_set():
            event.interrupt()
        if current and current[2] and time.monotonic() - last[0] >= 2:
            last[0] = time.monotonic()
            current[2](event.data_out)

    for name in ("cbMipInterrupt", "cbSimplexInterrupt", "cbIpmInterrupt"):
        getattr(solver, name).subscribe(callback)
    return unlimited
