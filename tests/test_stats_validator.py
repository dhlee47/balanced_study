"""
test_stats_validator.py — Unit tests for stats_validator.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stats_validator import StatisticalValidator, ValidationReport, MetricResult


@pytest.fixture
def well_balanced_df():
    """DataFrame where groups are very similar — should PASS all tests."""
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame({
        "weight": rng.normal(24, 0.5, n),   # tiny std → easy to balance
        "glucose": rng.normal(7, 0.2, n),
    })
    # Perfectly interleaved assignment
    assignment = np.array([i % 3 for i in range(n)])
    return df, assignment


@pytest.fixture
def poorly_balanced_df():
    """DataFrame where groups are clearly different — should likely FAIL."""
    df = pd.DataFrame({
        "weight": [10.0]*10 + [20.0]*10 + [30.0]*10,
        "glucose": [5.0]*10 + [7.0]*10 + [9.0]*10,
    })
    # Group 0 gets low, group 2 gets high
    assignment = np.array([0]*10 + [1]*10 + [2]*10)
    return df, assignment


class TestStatisticalValidator:

    def test_returns_validation_report(self, well_balanced_df):
        df, assignment = well_balanced_df
        validator = StatisticalValidator()
        report = validator.validate(df, ["weight", "glucose"], assignment, k=3)
        assert isinstance(report, ValidationReport)

    def test_report_has_correct_n_metrics(self, well_balanced_df):
        df, assignment = well_balanced_df
        validator = StatisticalValidator()
        report = validator.validate(df, ["weight", "glucose"], assignment, k=3)
        assert report.n_metrics == 2
        assert len(report.metric_results) == 2

    def test_well_balanced_passes(self, well_balanced_df):
        df, assignment = well_balanced_df
        validator = StatisticalValidator()
        report = validator.validate(df, ["weight", "glucose"], assignment, k=3)
        # Well-balanced data should pass most tests
        # (not guaranteed but expected with tiny std)
        assert report.n_metrics_failed == 0

    def test_poorly_balanced_fails(self, poorly_balanced_df):
        df, assignment = poorly_balanced_df
        validator = StatisticalValidator()
        report = validator.validate(df, ["weight", "glucose"], assignment, k=3)
        assert not report.overall_pass
        assert report.n_metrics_failed > 0

    def test_metric_result_fields(self, well_balanced_df):
        df, assignment = well_balanced_df
        validator = StatisticalValidator()
        report = validator.validate(df, ["weight", "glucose"], assignment, k=3)
        mr = report.metric_results[0]
        assert isinstance(mr, MetricResult)
        assert mr.metric == "weight"
        assert mr.raw_p_value >= 0
        assert mr.corrected_p_value >= mr.raw_p_value  # Bonferroni should inflate

    def test_summary_line_contains_pass_or_fail(self, well_balanced_df):
        df, assignment = well_balanced_df
        validator = StatisticalValidator()
        report = validator.validate(df, ["weight", "glucose"], assignment, k=3)
        line = report.summary_line()
        assert "PASS" in line or "FAIL" in line

    def test_format_report_returns_string(self, well_balanced_df):
        df, assignment = well_balanced_df
        validator = StatisticalValidator()
        report = validator.validate(df, ["weight", "glucose"], assignment, k=3)
        text = validator.format_report(report)
        assert isinstance(text, str)
        assert "STATISTICAL VALIDATION REPORT" in text

    def test_permutation_fallback_small_n(self):
        """With n=9, MANOVA should not run → permutation test used."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"w": rng.normal(5, 1, 9)})
        assignment = np.array([0, 1, 2] * 3)
        validator = StatisticalValidator()
        report = validator.validate(df, ["w"], assignment, k=3)
        # Should have used permutation test, not MANOVA
        assert report.manova_result is None or "error" in report.manova_result
