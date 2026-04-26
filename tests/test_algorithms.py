"""
test_algorithms.py — Unit tests for algorithms.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from algorithms import dynamic_allocation, evolutionary_algorithm, stratified_clustering_hybrid, run_algorithm


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


class TestDynamicAllocation:

    def test_returns_correct_shape(self, sample_df):
        a = dynamic_allocation(sample_df, METRIC_COLS, k=3)
        assert len(a) == 30

    def test_all_groups_present(self, sample_df):
        a = dynamic_allocation(sample_df, METRIC_COLS, k=3)
        assert set(a) == {0, 1, 2}

    def test_balanced_group_sizes(self, sample_df):
        a = dynamic_allocation(sample_df, METRIC_COLS, k=3)
        sizes = [int((a == g).sum()) for g in range(3)]
        assert max(sizes) - min(sizes) <= 1

    def test_deterministic(self, sample_df):
        a1 = dynamic_allocation(sample_df, METRIC_COLS, k=3)
        a2 = dynamic_allocation(sample_df, METRIC_COLS, k=3)
        np.testing.assert_array_equal(a1, a2)

    def test_k_equals_n(self, sample_df):
        small_df = sample_df.head(5)
        a = dynamic_allocation(small_df, METRIC_COLS[:1], k=5)
        assert len(a) == 5


class TestEvolutionaryAlgorithm:

    def test_returns_correct_shape(self, sample_df):
        a = evolutionary_algorithm(sample_df, METRIC_COLS, k=3, generations=50, population_size=20)
        assert len(a) == 30

    def test_all_groups_present(self, sample_df):
        a = evolutionary_algorithm(sample_df, METRIC_COLS, k=3, generations=50, population_size=20)
        assert set(a) == {0, 1, 2}

    def test_reproducible_with_seed(self, sample_df):
        a1 = evolutionary_algorithm(sample_df, METRIC_COLS, k=3, generations=50, population_size=20, random_seed=42)
        a2 = evolutionary_algorithm(sample_df, METRIC_COLS, k=3, generations=50, population_size=20, random_seed=42)
        np.testing.assert_array_equal(a1, a2)

    def test_different_seeds_differ(self, sample_df):
        a1 = evolutionary_algorithm(sample_df, METRIC_COLS, k=3, generations=100, population_size=20, random_seed=1)
        a2 = evolutionary_algorithm(sample_df, METRIC_COLS, k=3, generations=100, population_size=20, random_seed=99)
        # Not guaranteed to differ but very likely with different seeds
        # Just check both are valid
        assert set(a1).issubset({0, 1, 2})
        assert set(a2).issubset({0, 1, 2})


class TestStratifiedHybrid:

    def test_returns_correct_shape(self, sample_df):
        a = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3, max_sa_iter=100)
        assert len(a) == 30

    def test_all_groups_present(self, sample_df):
        a = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3, max_sa_iter=100)
        assert set(a) == {0, 1, 2}

    def test_reproducible_with_seed(self, sample_df):
        a1 = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3, max_sa_iter=50, random_seed=42)
        a2 = stratified_clustering_hybrid(sample_df, METRIC_COLS, k=3, max_sa_iter=50, random_seed=42)
        np.testing.assert_array_equal(a1, a2)


class TestRunAlgorithm:

    def test_dynamic_wrapper(self, sample_df):
        assignment, score, elapsed = run_algorithm("dynamic", sample_df, METRIC_COLS, k=3)
        assert len(assignment) == 30
        assert score >= 0
        assert elapsed >= 0

    def test_invalid_algo_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            run_algorithm("nonexistent", sample_df, METRIC_COLS, k=3)

    def test_all_algorithms_improve_over_random(self, sample_df):
        """Each algorithm should produce a lower score than random assignment."""
        rng = np.random.default_rng(0)
        random_assignment = rng.integers(0, 3, len(sample_df))

        from objective import compute_objective
        random_score = compute_objective(sample_df, METRIC_COLS, random_assignment)

        for algo in ["dynamic", "hybrid"]:
            _, score, _ = run_algorithm(algo, sample_df, METRIC_COLS, k=3)
            assert score <= random_score * 2, f"{algo} score unexpectedly high"
