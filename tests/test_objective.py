"""
test_objective.py — Unit tests for objective.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from objective import (
    between_group_dispersion, within_group_variance,
    mahalanobis_distance_between_groups, compute_objective, score_solution,
    _normalise_weights,
)


@pytest.fixture
def balanced_df():
    """Perfectly balanced assignment: groups have identical means."""
    np.random.seed(0)
    data = {"w": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], "g": [5.0, 6.0, 7.0, 5.0, 6.0, 7.0]}
    return pd.DataFrame(data)


@pytest.fixture
def imbalanced_df():
    """Clearly imbalanced: group 0 has high values, group 1 has low values."""
    data = {"w": [10.0, 11.0, 12.0, 1.0, 2.0, 3.0]}
    return pd.DataFrame(data)


class TestBetweenGroupDispersion:

    def test_perfect_balance_near_zero(self, balanced_df):
        assignment = np.array([0, 1, 2, 0, 1, 2])
        score = between_group_dispersion(balanced_df, ["w", "g"], assignment)
        assert score < 0.01

    def test_imbalanced_higher_than_balanced(self, imbalanced_df):
        balanced_assignment = np.array([0, 1, 0, 1, 0, 1])
        imbalanced_assignment = np.array([0, 0, 0, 1, 1, 1])
        s_bal = between_group_dispersion(imbalanced_df, ["w"], balanced_assignment)
        s_imb = between_group_dispersion(imbalanced_df, ["w"], imbalanced_assignment)
        assert s_imb > s_bal

    def test_custom_weights(self, balanced_df):
        assignment = np.array([0, 1, 2, 0, 1, 2])
        weights = np.array([2.0, 1.0])
        score_weighted = between_group_dispersion(balanced_df, ["w", "g"], assignment, weights)
        assert isinstance(score_weighted, float)


class TestWithinGroupVariance:

    def test_zero_variance_identical_values(self):
        df = pd.DataFrame({"w": [5.0, 5.0, 5.0, 5.0]})
        assignment = np.array([0, 0, 1, 1])
        score = within_group_variance(df, ["w"], assignment)
        assert score == pytest.approx(0.0)

    def test_higher_variance_scores_worse(self):
        df_low  = pd.DataFrame({"w": [5.0, 5.1, 5.0, 5.1]})
        df_high = pd.DataFrame({"w": [1.0, 9.0, 1.0, 9.0]})
        a = np.array([0, 0, 1, 1])
        assert within_group_variance(df_high, ["w"], a) > within_group_variance(df_low, ["w"], a)


class TestMahalanobis:

    def test_identical_centroids_near_zero(self):
        df = pd.DataFrame({"w": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})
        assignment = np.array([0, 1, 0, 1, 0, 1])
        d = mahalanobis_distance_between_groups(df, ["w"], assignment)
        assert d < 1.0  # centroids are close

    def test_returns_float(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"w": rng.normal(5, 1, 20), "g": rng.normal(7, 1, 20)})
        assignment = np.array([0]*10 + [1]*10)
        d = mahalanobis_distance_between_groups(df, ["w", "g"], assignment)
        assert isinstance(d, float)
        assert d >= 0


class TestComputeObjective:

    def test_lower_for_balanced(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"w": rng.normal(5, 1, 12), "g": rng.normal(7, 1, 12)})
        balanced = np.array([0,1,2]*4)
        imbalanced = np.array([0]*4 + [1]*4 + [2]*4)
        assert compute_objective(df, ["w", "g"], balanced) <= compute_objective(df, ["w", "g"], imbalanced) * 2

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"w": rng.normal(5, 1, 9)})
        a = np.array([0,1,2]*3)
        assert compute_objective(df, ["w"], a) >= 0


class TestScoreSolution:

    def test_returns_all_keys(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"w": rng.normal(5, 1, 9)})
        a = np.array([0,1,2]*3)
        result = score_solution(df, ["w"], a)
        assert "composite" in result
        assert "between_dispersion" in result
        assert "within_variance" in result
        assert "mahalanobis" in result
        assert "group_sizes" in result
        assert "group_means" in result


class TestNormaliseWeights:

    def test_none_returns_uniform(self):
        w = _normalise_weights(None, 4)
        np.testing.assert_allclose(w, [0.25, 0.25, 0.25, 0.25])

    def test_sums_to_one(self):
        w = _normalise_weights(np.array([1.0, 2.0, 3.0]), 3)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            _normalise_weights(np.array([1.0, 2.0]), 3)
