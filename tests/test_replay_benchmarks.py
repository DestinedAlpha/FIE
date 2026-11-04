import numpy as np
import pytest

from fpl_intelligence.replay_benchmarks import BASELINES, baseline_signals, compare_weekly_baselines
from fpl_intelligence.storage import Store


def test_baselines_use_archived_signals_and_do_not_double_adjust_ep(market):
    player = market[1].model_copy(
        update={
            "chance": 0.5,
            "raw": {
                "ep_next": "3",
                "points_per_game": "6",
                "selected_by_percent": "40",
                "target_points": 1000,
            },
        }
    )
    assert baseline_signals({1: player}, "official_expected") == {1: 3}
    assert baseline_signals({1: player}, "points_per_game") == {1: 3}
    assert baseline_signals({1: player}, "ownership") == {1: 20}
    assert all(baseline_signals({1: player}, method, scheduled=set())[1] == 0 for method in BASELINES)
    unavailable = player.model_copy(update={"chance": 0})
    assert all(baseline_signals({1: unavailable}, method)[1] == 0 for method in BASELINES)
    with pytest.raises(ValueError, match="Missing"):
        baseline_signals({1: player.model_copy(update={"raw": {}})}, "official_expected")
    with pytest.raises(ValueError, match="Nonfinite"):
        baseline_signals({1: player.model_copy(update={"raw": {"ep_next": "nan"}})}, "official_expected")


def test_all_baseline_decisions_freeze_before_target_labels_load(tmp_path, market, rules, state, monkeypatch):
    import fpl_intelligence.replay_benchmarks as module

    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "data")
    state_id = store.put("state", state.model_dump())
    store.put("market", {"events": [{"id": i, "average_entry_score": 50} for i in (1, 2)]})
    players = {
        p: v.model_copy(
            update={"raw": {"ep_next": str(p), "points_per_game": str(p), "selected_by_percent": str(p)}}
        )
        for p, v in market.items()
    }
    monkeypatch.setattr(module, "deadline_rows", lambda *args: (players, []))
    monkeypatch.setattr(module.Rules, "from_bootstrap", lambda *args: rules)
    weeks = []
    for event in (1, 2):
        source_id = store.put(
            "replay_inputs",
            {"bootstrap": {}, "fixture_proof": {"fixtures": []}, "deadline": "2026-08-01T11:00:00Z"},
        )
        weeks.append({"season": "2026/27", "event": event, "input_id": source_id, "actual_points": 12})
    store.put("weekly_replay", {"season": "2026/27", "weeks": weeks, "total_points": 24})

    def select(*args, diagnostics=None, **kwargs):
        diagnostics.update(gap=0)
        return list(range(1, 16))

    monkeypatch.setattr(module, "select_squad", select)

    def labels(*args):
        assert len(store.list("weekly_baseline_decision", 100)) == 6
        return (
            [
                {"season": "2026/27", "round": event, "element": p, "total_points": 1, "minutes": 90}
                for event in (1, 2)
                for p in range(1, 16)
            ],
            {},
            {},
        )

    monkeypatch.setattr(module, "canonical_rows", labels)
    result = compare_weekly_baselines(store)
    assert result["account_state_unchanged"] and store.get("state")["id"] == state_id
    assert all(r["baseline_total"] == 24 for r in result["comparisons"].values())
    assert all(r["reference_minus_baseline_total"] == 0 for r in result["comparisons"].values())
    assert all(
        np.array_equal(r["mean_weekly_difference_two_week_block_95ci"], [0, 0])
        for r in result["comparisons"].values()
    )
    assert all(
        "actual_points" not in store.get("weekly_baseline_decision", row["id"])
        for row in store.list("weekly_baseline_decision", 100)
    )
