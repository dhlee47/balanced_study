"""
test_algorithms.py — Unit tests for algorithms.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from algorithms import stratified_clustering_hybrid, run_algorithm


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "animal_id": range(1, 31),
        "weight": rng.normal(24, 3, 30),
        "glucose": rng.normal(7, 1, 30),
        "activity": rng.normal(1500, 300, 30),
    })


METRIC_COLS = ["weight", "glucose", "activity"]


class TestStratifiedHybrid:

    def test_returns_correct_shape(self, sample_df):
        a = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3, max_sa_iter=100)
        assert len(a) == 30

    def test_all_groups_present(self, sample_df):
        a = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3, max_sa_iter=100)
        assert set(a) == {0, 1, 2}

    def test_balanced_group_sizes(self, sample_df):
        a = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3, max_sa_iter=100)
        sizes = [int((a == g).sum()) for g in range(3)]
        assert max(sizes) - min(sizes) <= 1

    def test_reproducible_with_seed(self, sample_df):
        a1 = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3, max_sa_iter=50, random_seed=42)
        a2 = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3, max_sa_iter=50, random_seed=42)
        np.testing.assert_array_equal(a1, a2)

    def test_different_seeds_may_differ(self, sample_df):
        a1 = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3, max_sa_iter=200, random_seed=1)
        a2 = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3, max_sa_iter=200, random_seed=99)
        assert set(a1).issubset({0, 1, 2})
        assert set(a2).issubset({0, 1, 2})

    def test_custom_group_sizes(self, sample_df):
        a = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3,
                                         group_sizes=[12, 10, 8], max_sa_iter=50)
        assert (a == 0).sum() == 12
        assert (a == 1).sum() == 10
        assert (a == 2).sum() == 8

    def test_two_groups(self, sample_df):
        a = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=2, max_sa_iter=50)
        assert set(a) == {0, 1}

    def test_single_metric(self, sample_df):
        a = stratified_clustering_hybrid(sample_df, ["weight"], k=3, max_sa_iter=50)
        assert len(a) == 30

    def test_improves_over_random(self, sample_df):
        """Hybrid should produce a lower objective score than a random assignment."""
        from objective import compute_objective
        rng = np.random.default_rng(0)
        random_assignment = rng.integers(0, 3, len(sample_df))
        random_score = compute_objective(sample_df, METRIC_COLS, random_assignment)

        _, score, _ = run_algorithm(sample_df, METRIC_COLS, k=3, max_sa_iter=500)
        assert score <= random_score * 2


class TestRunAlgorithm:

    def test_returns_correct_types(self, sample_df):
        assignment, score, elapsed = run_algorithm(sample_df, METRIC_COLS, k=3, max_sa_iter=50)
        assert len(assignment) == 30
        assert score >= 0
        assert elapsed >= 0

    def test_score_detail_keys(self, sample_df):
        from objective import score_solution
        assignment, _, _ = run_algorithm(sample_df, METRIC_COLS, k=3, max_sa_iter=50)
        result = score_solution(sample_df, METRIC_COLS, assignment)
        for key in ("composite", "between_dispersion", "within_variance", "mahalanobis"):
            assert key in result
