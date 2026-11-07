import numpy as np

from fpl_intelligence.role_balance import reconcile


def cohort(count=16):
    return [
        {"season": "2025/26", "event": 1, "fixture_id": 8, "team": "Club", "position": 2}
        for _ in range(count)
    ]


def test_team_reconciliation_preserves_hard_absence_mass_and_player_symmetry():
    p = np.tile([0.5, 0.0, 0.2, 0.3], (16, 1))
    p[0] = [1, 0, 0, 0]
    q = reconcile(p, cohort())
    assert np.allclose(q.sum(1), 1) and np.array_equal(q[0], p[0])
    assert np.all(q[:, 1] == 0)
    assert np.isclose((q @ [0, 24, 74, 90]).sum(), 900)
    assert np.allclose(q[1:], q[1])
    permutation = np.random.default_rng(8).permutation(16)
    assert np.allclose(reconcile(p[permutation], cohort()), q[permutation])


def test_incomplete_or_impossible_rosters_never_invent_players():
    p = np.tile([1.0, 0, 0, 0], (16, 1))
    assert np.array_equal(reconcile(p, cohort()), p)
    p = np.tile([0.5, 0, 0, 0.5], (10, 1))
    assert np.array_equal(reconcile(p, cohort(10)), p)
