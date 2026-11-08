"""Reconcile revealed deadline squads without inventing hidden current transfers.

Public picks reveal a squad only after its deadline. A public reconciliation is
therefore a planning snapshot, not confirmation that no later transfers exist.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from threading import RLock

from .domain import ChipUse, Lot, TeamState
from .ingestion import load_market
from .rules import Rules, quote, selling_price, validate_squad
from .storage import now

state_lock = RLock()


_held_account_locks: ContextVar[frozenset[str]] = ContextVar("account_locks", default=frozenset())


@contextmanager
def locked_state(store):
    """Coordinate API edits with calculation processes; OS releases on cancellation."""
    from .pipeline_jobs import replay_lock
    from .search_control import check_cancelled

    key = str(store.root.resolve())
    with state_lock:
        check_cancelled()
        if key in _held_account_locks.get():
            yield
            return
        token = _held_account_locks.set(_held_account_locks.get() | {key})
        try:
            with replay_lock(
                store.root / "account-write.lock", busy_message="Account publication in progress; try again"
            ):
                yield
        finally:
            _held_account_locks.reset(token)


def revealed_pick_events(history: dict, released: int) -> list[int]:
    """Request only deadlines this account actually entered, including late joiners."""
    events = sorted({int(r["event"]) for r in history.get("current", []) if r["event"] <= released})
    if not events or events[-1] != released:
        raise ValueError(f"Missing revealed account history for GW{released}")
    return events[-2:]


def next_event(market: dict, at: str | None = None) -> int | None:
    instant = datetime.fromisoformat((at or now()).replace("Z", "+00:00"))
    return next(
        (
            e["id"]
            for e in sorted(market["events"], key=lambda e: e["id"])
            if datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00")) > instant
        ),
        None,
    )


def free_transfers_after(history: dict, through: int, rules: Rules) -> int:
    """Replay all revealed deadlines; chips preserve saved transfers."""
    chips = {r["event"]: r["name"] for r in history.get("chips", [])}
    rows = {r["event"]: r for r in history.get("current", [])}
    if not rows or through < min(rows):
        raise ValueError("No revealed account deadlines to replay")
    first = min(rows)
    ft = 1
    for event in range(first + 1, through + 1):
        if event not in rows:
            raise ValueError(f"Missing public account history for GW{event}")
        if chips.get(event) not in ("freehit", "wildcard"):
            ft = min(rules.ft_cap, max(0, ft - rows[event]["event_transfers"]) + 1)
        allocation = rules.event_overrides.get(event + 1, {}).get("rules", {}).get("free_transfers")
        if allocation is not None:
            ft = int(allocation)
    if through == first:
        ft = int(rules.event_overrides.get(first + 1, {}).get("rules", {}).get("free_transfers", 1))
    return ft


def _lot(identity, player, old, transfers, chip_by_event, through):
    acquisitions = sorted(
        (
            r
            for r in transfers
            if r["element_in"] == identity
            and r["event"] <= through
            and chip_by_event.get(r["event"]) != "freehit"
        ),
        key=lambda r: (r["event"], r.get("time", "")),
    )
    purchase = acquisitions[-1]["element_in_cost"] if acquisitions else None
    if purchase is not None:
        return Lot(
            player_id=identity,
            purchase_price=purchase,
            sell_price=selling_price(purchase, player.price),
            quote_market_price=player.price,
            provenance="official_public_transfer_cost",
        )
    if old is not None:
        try:
            sale = quote(old, player.price)
        except ValueError:
            from .scenarios import sale_bounds

            if old.quote_confirmed:
                low, high = sale_bounds(old, old.quote_market_price, player.price)
                if low == high:
                    return old.model_copy(
                        update={
                            "sell_price": low,
                            "quote_market_price": player.price,
                            "provenance": "Unique sale value implied by prior exact quote and current price",
                        }
                    )
            # Ambiguous sale values retain their dated basis and require a new
            # quote; no private purchase price is invented.
            return old
        return old.model_copy(update={"sell_price": sale, "quote_market_price": player.price})
    # This quote is explicitly unresolved. No optimizer may treat the current
    # price as an exact sale receipt for an unknown acquisition lot.
    return Lot(
        player_id=identity,
        purchase_price=None,
        sell_price=player.price,
        quote_market_price=player.price,
        quote_confirmed=False,
        provenance="public_roster_sell_quote_unresolved",
    )


def reconcile_public(store, expected_state_id: str | None = None) -> dict:
    with locked_state(store):
        return _reconcile_public(store, expected_state_id)


def _reconcile_public(store, expected_state_id: str | None = None) -> dict:
    saved = store.get("state")
    if expected_state_id is not None and saved["id"] != expected_state_id:
        return {"status": "Account edited during refresh; preserved newer account", "account_changed": False}
    entry = saved.get("entry_id")
    if not entry:
        return {"status": "No entry ID configured", "account_changed": False}
    players, market = load_market(store)
    if saved["season"] != market["season"]:
        return {"status": "New season requires a new account import", "account_changed": False}
    upcoming = next_event(market)
    if upcoming is None:
        return {"status": "Season finished; account retained", "account_changed": False}
    history, meta = store.latest(f"entry_history_{entry}")
    revealed = max((r["event"] for r in history.get("current", []) if r["event"] < upcoming), default=0)
    if not revealed:
        return {"status": "No revealed deadline squad yet", "account_changed": False}
    if revealed != upcoming - 1:
        return {
            "status": "Latest deadline squad not yet publicly available; preserved account",
            "account_changed": False,
            "revealed_event": revealed,
            "expected_revealed_event": upcoming - 1,
        }
    deadline = next(e["deadline_time"] for e in market["events"] if e["id"] == revealed)
    if saved["event"] == upcoming and saved["status"] == "confirmed" and saved["known_at"] >= deadline:
        return {
            "status": "Preserved confirmed account and any unrevealed transfers",
            "account_changed": False,
        }
    chips = {r["event"]: r["name"] for r in history.get("chips", [])}
    roster_event = revealed - 1 if chips.get(revealed) == "freehit" else revealed
    picks, picks_meta = store.latest(f"picks_{entry}_{roster_event}")
    if len(picks.get("picks", [])) != 15:
        raise ValueError("Public deadline roster is incomplete; account preserved")
    transfers, transfer_meta = store.latest(f"transfers_{entry}")
    original = {r["player_id"]: Lot.model_validate(r) for r in saved["lots"]}
    lots = [
        _lot(r["element"], players[r["element"]], original.get(r["element"]), transfers, chips, revealed)
        for r in picks["picks"]
    ]
    bank_row = next(r for r in history["current"] if r["event"] == roster_event)
    last = next(r for r in history["current"] if r["event"] == revealed)
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    state = TeamState(
        season=market["season"],
        event=upcoming,
        name=saved["name"],
        entry_id=entry,
        bank=bank_row["bank"],
        free_transfers=free_transfers_after(history, revealed, rules),
        lots=lots,
        used_chips=[ChipUse(chip=r["name"], event=r["event"]) for r in history.get("chips", [])],
        points=last["total_points"],
        status="public_deadline",
        known_at=meta["captured_at"],
        grandfathered_clubs=[
            club
            for club, count in Counter(players[r.player_id].team for r in lots).items()
            if count > rules.club_limit
        ],
    )
    validate_squad(state, players, rules)
    # Do not replace a newer editor import that completed during source reads.
    if store.get("state")["id"] != saved["id"]:
        return {"status": "Account edited during refresh; preserved newer account", "account_changed": False}
    # Idempotence: a timestamp-only refresh must not invalidate every plan.
    payload = state.model_dump()
    if {k: v for k, v in payload.items() if k != "known_at"} == {
        k: v for k, v in saved.items() if k not in ("id", "known_at")
    }:
        return {
            "status": "Revealed account already synchronized",
            "account_changed": False,
            "state_id": saved["id"],
        }
    identity = store.put("state", payload)
    receipt = {
        "at": now(),
        "previous_state_id": saved["id"],
        "state_id": identity,
        "revealed_event": revealed,
        "planning_event": upcoming,
        "source_hashes": [meta["hash"], picks_meta["hash"], transfer_meta["hash"]],
        "account_changed": True,
        "status": "Synchronized revealed deadline squad",
        "limitation": "Transfers after the latest deadline are private. Confirm any further moves and sell quotes in My squad.",
    }
    store.put("account_sync", receipt)
    store.event("public_account_sync", identity, receipt)
    return receipt
