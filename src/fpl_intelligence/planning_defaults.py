"""Choose a horizon that can value remaining chips before they expire."""


def defaults(store, state, market):
    event = int(state.get("event", 1))
    end = min(38, event + 7)
    reason = "Next eight gameweeks"
    if state and market.get("source_hash"):
        from .rules import Rules

        bootstrap, _ = store.latest("bootstrap", content_hash=market["source_hash"])
        if not all(k in bootstrap for k in ("game_settings", "element_types", "events", "chips")):
            return {"end_event": end, "search_seconds": 180, "reason": reason}
        rules = Rules.from_bootstrap(bootstrap)
        for chip in ("wildcard", "freehit", "bboost", "3xc"):
            try:
                first, last = rules.chip_window(chip, event)
            except ValueError:
                continue
            if any(c["chip"] == chip and first <= c["event"] <= last for c in state.get("used_chips", [])):
                continue
            if last > end:
                end = min(38, last)
                reason = "Includes the remaining chip window"
    return {"end_event": end, "search_seconds": 180, "reason": reason}


def chips_assessed(store, plan, pinned=None):
    """An automatic policy cannot spend an entitlement outside its assessed window."""
    if not any(r.get("chip") for r in plan.get("ledger", [])):
        return True
    if not store.list("market", 1) or not store.list("state", 1):
        return True
    from .rules import Rules

    market = store.get("market", plan.get("market_id"))
    bootstrap, _ = store.latest("bootstrap", content_hash=market["source_hash"])
    rules = Rules.from_bootstrap(bootstrap)
    state = store.get("state", plan["state_id"])
    requested = pinned if pinned is not None else plan.get("chip_constraints", {}).get("pinned", {})
    requested = {int(event): chip for event, chip in requested.items()}
    for row in plan["ledger"]:
        chip = row.get("chip")
        if not chip or requested.get(row["event"]) == chip:
            continue
        if state.get("active_chip") == chip and row["event"] == state["event"]:
            continue
        _, last = rules.chip_window(chip, row["event"])
        if last > plan["end_event"]:
            return False
    return True
