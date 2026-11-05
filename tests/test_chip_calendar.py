"""Exercise chip-calendar constraints through the real MIP and account engine."""

import numpy as np
import pytest

from fpl_intelligence.domain import ChipUse, Lot
from fpl_intelligence.optimization import optimize
from fpl_intelligence.storage import Store


@pytest.fixture
def planning_store(tmp_path, market, rules, state, monkeypatch):
    def make(start=9, end=10, prior_transfer=False):
        current = state.model_copy(update={"event": start, "status": "dated_seed"})
        if prior_transfer:
            current = current.model_copy(
                update={
                    "transfers_made": 1,
                    "bank": 28,
                    "deadline_bank": 30,
                    "deadline_lots": state.lots,
                    "lots": [lot for lot in state.lots if lot.player_id != 3]
                    + [Lot(player_id=18, purchase_price=51, sell_price=51, quote_market_price=51)],
                }
            )
        store = Store(tmp_path)
        frozen = {
            "id": "market",
            "season": current.season,
            "source_hash": "bootstrap",
            "events": [],
            "teams": [{"id": team} for team in sorted({p.team for p in market.values()})],
        }
        for module in ("ingestion", "optimization", "simulation"):
            monkeypatch.setattr(f"fpl_intelligence.{module}.load_market", lambda *a: (market, frozen))
        monkeypatch.setattr("fpl_intelligence.rules.Rules.from_bootstrap", lambda *a: rules)
        monkeypatch.setattr(store, "latest", lambda *a, **kw: ({}, {}))
        ids = sorted(market)
        means = {p: float(1 + p % 7) for p in ids}
        path = tmp_path / "scenarios" / "worlds.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            path,
            **{
                f"{name}_{event}": np.tile(
                    [means[p] if name == "points" else 90 for p in ids], (24, 1)
                ).astype(np.int16)
                for event in range(start, end + 1)
                for name in ("points", "minutes")
            },
        )
        model_id = store.put("decision_model", {"promoted": True})
        fid = store.put(
            "forecast",
            {
                "market_id": "market",
                "rules_hash": rules.hash,
                "start_event": start,
                "end_event": end,
                "worlds": 24,
                "seed": 11,
                "scenario_path": str(path),
                "player_ids": ids,
                "players": [
                    {"event": e, "player_id": p, "mean": means[p]} for e in range(start, end + 1) for p in ids
                ],
                "input_hashes": [],
                "model_version": "learned-points-v1",
                "decision_model_id": model_id,
            },
        )
        return store, store.put("state", current.model_dump()), fid, current

    return make


def test_chip_calendar_covers_all_four_chips_and_independent_periods(planning_store):
    from fpl_intelligence.chips import compare_chips, rebase_chip_comparison

    store, sid, fid, _ = planning_store(19, 20)
    base = optimize(store, sid, fid, 20, 2, chips={})
    result = compare_chips(store, base["id"], 2)
    assert {(r["chip"], r["event"]) for r in result["calendar"]} == {
        (c, t) for c in ("wildcard", "freehit", "bboost", "3xc") for t in (19, 20)
    }
    assert len(result["comparisons"]) == 8
    selected = store.get("plan", result["best_candidate_plan_id"])
    rebased = rebase_chip_comparison(store, result, selected["id"])
    assert rebased["base_plan_id"] == selected["id"] and rebased["dates_evaluated"] == 8
    other_id = store.put("plan", {**selected, "forecast_id": "different"})
    with pytest.raises(ValueError, match="different frozen inputs"):
        rebase_chip_comparison(store, result, other_id)
    for row in result["comparisons"]:
        alternative = store.get("plan", row["without_plan_id"])
        lo, hi = row["window"]
        assert not any(r["chip"] == row["chip"] and lo <= r["event"] <= hi for r in alternative["ledger"])
    # Disabling the first entitlement leaves the second available.
    second = optimize(store, sid, fid, 20, 2, blocked_chips={19: ("bboost",)}, pin_chips={20: "bboost"})
    assert second["ledger"][1]["chip"] == "bboost"


def test_freehit_after_prior_transfers_restores_bank_and_preserves_tail_constraints(planning_store):
    store, sid, fid, current = planning_store(prior_transfer=True)
    result = optimize(store, sid, fid, 10, 2, chips={9: "freehit"}, max_transfers=0)
    first, tail = result["ledger"]
    assert first["chip"] == "freehit"
    assert set(first["permanent_squad"]) == {lot.player_id for lot in current.deadline_lots}
    assert not tail["buy"] and tail["chip"] is None
    assert tail["bank_before"] == current.deadline_bank
    assert store.get("state")["id"] == sid


def test_prior_transfer_does_not_disable_future_freehit(planning_store):
    store, _, fid, state = planning_store(prior_transfer=True)
    state = state.model_copy(update={"deadline_lots": None, "deadline_bank": None})
    sid = store.put("state", state.model_dump())
    result = optimize(store, sid, fid, 10, 2, chips={10: "freehit"})
    assert result["ledger"][1]["chip"] == "freehit"


def test_chip_calendar_omits_spent_entitlement_and_consecutive_freehit(planning_store):
    from fpl_intelligence.chips import compare_chips

    store, sid, fid, state = planning_store(19, 20)
    state = state.model_copy(
        update={"used_chips": [ChipUse(chip="3xc", event=3), ChipUse(chip="freehit", event=18)]}
    )
    sid = store.put("state", state.model_dump())
    base = optimize(store, sid, fid, 20, 2, chips={})
    result = compare_chips(store, base["id"], 2)
    dates = {(r["chip"], r["event"]) for r in result["calendar"]}
    assert ("3xc", 19) not in dates and ("freehit", 19) not in dates
    assert ("3xc", 20) in dates and ("freehit", 20) in dates
