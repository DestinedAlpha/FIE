import numpy as np
import pytest

from fpl_intelligence.future_distribution import crps


def test_crps_matches_the_independent_empirical_pairwise_definition():
    draws = np.array([-2, 0, 0, 3, 8])
    target = 4
    expected = np.mean(np.abs(draws - target)) - 0.5 * np.mean(np.abs(draws[:, None] - draws))
    assert crps(draws, target) == pytest.approx(expected)
    assert crps([4, 4], 4) == 0 and crps([4], 1) == 3
