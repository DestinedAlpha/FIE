from collections import defaultdict

import numpy as np
import pytest

from fpl_intelligence.consensus import (
    WEIGHTS,
    DeadlineInputs,
    comparison_is_current,
    mix_distributions,
    published_vector,
    reusable_decision,
    select_weight,
)
from fpl_intelligence.replay import archive_integer
from fpl_intelligence.replay_benchmarks import BASELINES


def test_announced_double_allocates_weekly_ep_once_and_missing_is_masked():
    packet = {"ep_next": "8", "form": "5.5", "points_per_game": "6"}
    one, double = published_vector(packet, 1), published_vector(packet, 2)
    assert one == [8, 1, 6, 1, 5.5, 1, 1]
    assert double == [4, 1, 6, 1, 5.5, 1, 2]
    assert published_vector({}, 0) == [0, 0, 0, 0, 0, 0, 0]
    assert one != published_vector({"ep_next": 0, "form": 0, "points_per_game": 0}, 1)
    with pytest.raises(ValueError, match="Nonfinite"):
        published_vector({"ep_next": "nan"}, 1)


def test_mixture_preserves_mass_and_total_point_expectation():
    rng = np.random.default_rng(77)
    rp, cp = rng.dirichlet(np.ones(4), 10), rng.dirichlet(np.ones(4), 10)
    rc, cc = rng.normal(4, 3, (10, 4)), rng.normal(4, 3, (10, 4))
    rs, cs = rng.random(10), rng.random(10)
    for weight in (0, 0.25, 0.5, 1):
        p, c, s = mix_distributions((rp, rc, rs), (cp, cc, cs), weight)
        np.testing.assert_allclose(p.sum(1), 1)
        np.testing.assert_allclose(
            (p * c).sum(1), (1 - weight) * (rp * rc).sum(1) + weight * (cp * cc).sum(1)
        )
        np.testing.assert_allclose(s, (1 - weight) * rs + weight * cs)
    p, c, _ = mix_distributions((np.zeros((1, 4)), np.zeros((1, 4)), rs[:1]), (cp[:1], cc[:1], cs[:1]), 0)
    assert np.isfinite(c).all() and not p.any()


def test_historical_tensors_ignore_target_and_future_history(market):
    builder = object.__new__(DeadlineInputs)
    builder.history = defaultdict(list)
    # Exact four-hour completion boundary is excluded, as are future seasons.
    builder.history["code", 1] = [
        {"season": "2026/27", "round": 8, "kickoff_time": "2026-10-24T15:00:00Z", "minutes": 90},
        {"season": "2026/27", "round": 8, "kickoff_time": "2026-10-31T07:00:00Z", "minutes": 999},
        {"season": "2026/27", "round": 9, "kickoff_time": "2026-10-30T15:00:00Z", "minutes": 888},
        {"season": "2027/28", "round": 1, "kickoff_time": "2026-10-20T15:00:00Z", "minutes": 777},
    ]
    builder.experts, builder.samples, builder.workload = {}, {}, defaultdict(list)

    class Enrichment:
        def vector(self, *args):
            return np.zeros(31)

    builder.enrichment = Enrichment()
    rows = [{"player_id": 1, "x": [0] * 26}]
    arrays, _ = builder.build(
        rows, market, {"season": "2026/27", "event": 9, "deadline": "2026-10-31T11:00:00Z"}
    )
    assert arrays["sequence_mask"].sum() == 1
    assert arrays["sequence"][0, -1, 0] == 1  # 90 / 90


@pytest.mark.parametrize("value,expected", [("1.0", 1), ("38", 38), ("", None), (None, None)])
def test_archived_nullable_integer_columns(value, expected):
    assert archive_integer(value) == expected


@pytest.mark.parametrize("value", ["1.5", "inf", "bad", "NaN"])
def test_archive_identifiers_cannot_be_rounded_or_nonfinite(value):
    with pytest.raises(ValueError):
        archive_integer(value)


def test_weight_selection_ignores_acceptance_and_current_scores():
    names = [f"mix_{w:g}" for w in WEIGHTS] + list(BASELINES)
    decisions = [
        {
            "season": "2025/26",
            "event": event,
            "policy": name,
            "actual_points": 60 if name == "mix_0.5" else 50,
        }
        for event in range(1, 39)
        for name in names
    ]
    selected, totals = select_weight(decisions, "2025/26")
    assert selected == "mix_0.5"
    for r in decisions:
        if r["event"] > 19:
            r["actual_points"] = 100000 if r["policy"] == "mix_1" else -100000
    decisions.extend({"season": "2026/27", "event": 1, "policy": p, "actual_points": 1e9} for p in names)
    assert select_weight(decisions, "2025/26") == (selected, totals)
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        select_weight(decisions + [decisions[0]], "2025/26")


def test_cached_decisions_require_reproduced_full_market_utilities():
    cached = {"input_id": "before-deadline", "signals": {"1": 5.0, "2": 4.0}, "actual_points": 999}
    assert reusable_decision(cached, {1: 5.0, 2: 4.0}, "before-deadline")
    assert not reusable_decision(cached, {1: 5.1, 2: 4.0}, "before-deadline")
    assert not reusable_decision(cached, {1: 5.0}, "before-deadline")
    assert not reusable_decision(cached, {1: 5.0, 2: 4.0}, "new-schedule")
    assert not reusable_decision(None, {1: 5.0}, "before-deadline")


def test_expired_or_differently_scoped_comparisons_cannot_be_current_actions():
    report = {
        "market_id": "m",
        "state_id": "s",
        "primary_model_id": "p",
        "consensus_model_id": "c",
        "event": 9,
    }
    market = {"id": "m", "events": [{"id": 9, "deadline_time": "2026-10-31T11:00:00Z"}]}
    assert comparison_is_current(report, market, "s", "p", "c", "2026-10-31T10:59:59Z")
    assert not comparison_is_current(report, market, "s", "p", "c", "2026-10-31T11:00:00Z")
    assert not comparison_is_current(report, market, "s", "p", "c", "2026-11-04T11:00:00Z")
    for field in ("market_id", "state_id", "primary_model_id", "consensus_model_id"):
        assert not comparison_is_current(
            report | {field: "old"}, market, "s", "p", "c", "2026-10-31T10:59:59Z"
        )
