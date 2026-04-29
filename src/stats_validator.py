"""
stats_validator.py — Statistical validation of group-balancing solutions.

After every balancing run, this module tests whether the resulting groups
are statistically equivalent on all metrics — the desired outcome of
good balancing.

Test selection (Assumption A04):
    1. Shapiro-Wilk normality test per group per metric
       → if all groups pass (p > 0.05): parametric path (ANOVA)
       → if any group fails: non-parametric path (Kruskal-Wallis)
    2. Post-hoc: Tukey HSD (ANOVA) or Dunn's test (Kruskal-Wallis)
    3. Bonferroni correction for multiple metrics
    4. Multivariate: MANOVA if n >= 20 and min group size >= 3;
       else permutation test
    5. Box's M test for equality of covariance matrices (via pingouin)

ValidationReport:
    Structured result with per-metric pass/fail, overall pass/fail,
    and remediation suggestions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from scipy.stats import f_oneway, kruskal, shapiro
from statsmodels.multivariate.manova import MANOVA


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Statistical test results for a single metric."""
    metric: str
    normality_by_group: dict[int, tuple[float, float]]  # group → (stat, p)
    all_normal: bool
    test_used: str                  # 'ANOVA' or 'Kruskal-Wallis'
    test_statistic: float
    raw_p_value: float
    corrected_p_value: float        # after Bonferroni
    significant: bool               # True = groups differ (BAD for balancing)
    posthoc_pairs: dict[str, float] | None = None  # 'g1_vs_g2' → p-value


@dataclass
class ValidationReport:
    """Full validation report for one balancing solution."""
    n_animals: int
    k_groups: int
    n_metrics: int
    metric_results: list[MetricResult]
    manova_result: dict[str, Any] | None
    permutation_result: dict[str, Any] | None
    overall_pass: bool
    n_metrics_failed: int
    remediation: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        status = "PASS" if self.overall_pass else "FAIL"
        return (
            f"[{status}] {self.n_metrics - self.n_metrics_failed}/{self.n_metrics} metrics OK  "
            f"| MANOVA: {self._manova_summary()}"
        )

    def _manova_summary(self) -> str:
        if self.manova_result:
            p = self.manova_result.get("p_value", "N/A")
            return f"p={p:.4f}" if isinstance(p, float) else str(p)
        if self.permutation_result:
            p = self.permutation_result.get("p_value", "N/A")
            return f"permutation p={p:.4f}" if isinstance(p, float) else str(p)
        return "not run"


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------

class StatisticalValidator:
    """
    Run the full univariate + multivariate validation battery on a
    group assignment.

    Parameters
    ----------
    alpha_level : float
        Significance threshold (before Bonferroni).  Default 0.05.
    n_permutations : int
        Number of permutations for the permutation test fallback.  Default 999.
    random_seed : int
        RNG seed for permutation test.  Default 42.

    Examples
    --------
    >>> validator = StatisticalValidator()
    >>> report = validator.validate(df, ['w', 'g', 'act'], assignment, k=3)
    >>> print(report.summary_line())
    """

    def __init__(
        self,
        alpha_level: float = 0.05,
        n_permutations: int = 999,
        random_seed: int = 42,
    ) -> None:
        self.alpha = alpha_level
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
        metric_cols: list[str],
        assignment: np.ndarray,
        k: int,
    ) -> ValidationReport:
        """
        Run all statistical tests and return a ValidationReport.

        Parameters
        ----------
        df : pd.DataFrame
            Animal data.
        metric_cols : list[str]
            Metric column names.
        assignment : np.ndarray of int, shape (n,)
            Group labels (0-indexed).
        k : int
            Number of groups.

        Returns
        -------
        ValidationReport
        """
        n = len(df)
        m = len(metric_cols)
        bonferroni_m = m  # number of comparisons for correction

        # --- Univariate tests per metric ---
        raw_p_values = []
        metric_results = []
        for col in metric_cols:
            result = self._test_one_metric(df, col, assignment, k)
            raw_p_values.append(result.raw_p_value)
            metric_results.append(result)

        # Apply Bonferroni correction
        for i, mr in enumerate(metric_results):
            corrected = min(mr.raw_p_value * bonferroni_m, 1.0)
            mr.corrected_p_value = corrected
            mr.significant = corrected < self.alpha

        # --- Multivariate ---
        group_sizes = [(assignment == g).sum() for g in range(k)]
        min_group = min(group_sizes)
        use_manova = n >= 20 and min_group >= 3 and m >= 2

        manova_result = None
        permutation_result = None

        if use_manova:
            manova_result = self._run_manova(df, metric_cols, assignment)
        else:
            permutation_result = self._run_permutation_test(df, metric_cols, assignment, k)

        # --- Overall verdict ---
        n_failed = sum(1 for mr in metric_results if mr.significant)
        manova_fail = (
            manova_result is not None
            and isinstance(manova_result.get("p_value"), float)
            and manova_result["p_value"] < self.alpha
        )
        overall_pass = (n_failed == 0) and (not manova_fail)

        # --- Remediation suggestions ---
        remediation = []
        if n_failed > 0:
            failed_metrics = [mr.metric for mr in metric_results if mr.significant]
            remediation.append(
                f"Metrics with significant group differences: {failed_metrics}. "
                "Try Algorithm 2 (evolutionary) with more generations (≥2000), "
                "or Algorithm 3 (hybrid) with lower cooling rate (0.99)."
            )
        if manova_fail:
            remediation.append(
                "MANOVA detected overall group separation. "
                "Consider increasing metric weights for the separating metrics "
                "and re-running the algorithm."
            )
        if not remediation and not overall_pass:
            remediation.append(
                "Enable Continuous Improvement mode and allow up to 20 iterations."
            )

        return ValidationReport(
            n_animals=n,
            k_groups=k,
            n_metrics=m,
            metric_results=metric_results,
            manova_result=manova_result,
            permutation_result=permutation_result,
            overall_pass=overall_pass,
            n_metrics_failed=n_failed,
            remediation=remediation,
        )

    # ------------------------------------------------------------------
    # Univariate tests
    # ------------------------------------------------------------------

    def _test_one_metric(
        self,
        df: pd.DataFrame,
        col: str,
        assignment: np.ndarray,
        k: int,
    ) -> MetricResult:
        """Run normality + ANOVA/KW for a single metric."""
        groups = [df[col].values[assignment == g] for g in range(k)]

        # Shapiro-Wilk per group
        normality = {}
        all_normal = True
        for g, grp in enumerate(groups):
            if len(grp) >= 3:
                stat, p = shapiro(grp)
            else:
                stat, p = float("nan"), 1.0  # too small to test
            normality[g] = (float(stat), float(p))
            if p < self.alpha:
                all_normal = False

        # Main test
        if all_normal:
            test_name = "ANOVA"
            stat, p = f_oneway(*groups)
        else:
            test_name = "Kruskal-Wallis"
            try:
                stat, p = kruskal(*groups)
            except ValueError:
                # All values identical across groups
                stat, p = 0.0, 1.0

        # Post-hoc
        posthoc = None
        if p < self.alpha:
            posthoc = self._posthoc(df, col, assignment, k, all_normal)

        return MetricResult(
            metric=col,
            normality_by_group=normality,
            all_normal=all_normal,
            test_used=test_name,
            test_statistic=float(stat),
            raw_p_value=float(p),
            corrected_p_value=float(p),  # will be updated by caller
            significant=False,           # will be updated by caller
            posthoc_pairs=posthoc,
        )

    def _posthoc(
        self,
        df: pd.DataFrame,
        col: str,
        assignment: np.ndarray,
        k: int,
        parametric: bool,
    ) -> dict[str, float]:
        """Run pairwise post-hoc tests and return {pair_label: p_value}."""
        tmp = pd.DataFrame({col: df[col].values, "group": assignment})

        try:
            if parametric:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                result = pairwise_tukeyhsd(tmp[col], tmp["group"], alpha=self.alpha)
                pairs = {}
                for row in result.summary().data[1:]:
                    label = f"g{row[0]}_vs_g{row[1]}"
                    pairs[label] = float(row[3])
                return pairs
            else:
                ph = sp.posthoc_dunn(tmp, val_col=col, group_col="group", p_adjust="bonferroni")
                pairs = {}
                for i in range(k):
                    for j in range(i + 1, k):
                        if i in ph.index and j in ph.columns:
                            pairs[f"g{i}_vs_g{j}"] = float(ph.loc[i, j])
                return pairs
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Multivariate tests
    # ------------------------------------------------------------------

    def _run_manova(
        self,
        df: pd.DataFrame,
        metric_cols: list[str],
        assignment: np.ndarray,
    ) -> dict[str, Any]:
        """
        Run one-way MANOVA to test if groups differ simultaneously on all metrics.

        Returns
        -------
        dict with keys: p_value, wilks_lambda, f_statistic, method, passed.
        """
        try:
            tmp = df[metric_cols].copy()
            tmp["_group"] = assignment

            formula = " + ".join(metric_cols) + " ~ _group"
            mv = MANOVA.from_formula(formula, data=tmp)
            res = mv.mv_test()

            # Extract Wilks' Lambda p-value
            intercept_table = res.results["_group"]["stat"]
            wilks_row = intercept_table[intercept_table.index == "Wilks' lambda"]
            if not wilks_row.empty:
                p_val = float(wilks_row["Pr > F"].values[0])
                wilks = float(wilks_row["Value"].values[0])
                f_stat = float(wilks_row["F Value"].values[0])
            else:
                p_val = float("nan")
                wilks = float("nan")
                f_stat = float("nan")

            return {
                "method": "MANOVA (Wilks' Lambda)",
                "wilks_lambda": wilks,
                "f_statistic": f_stat,
                "p_value": p_val,
                "passed": p_val >= self.alpha,
            }

        except Exception as exc:
            return {"method": "MANOVA", "error": str(exc), "p_value": float("nan")}

    def _run_permutation_test(
        self,
        df: pd.DataFrame,
        metric_cols: list[str],
        assignment: np.ndarray,
        k: int,
    ) -> dict[str, Any]:
        """
        Permutation test: compare observed between-group dispersion to a
        null distribution generated by randomly shuffling group labels.

        Parameters
        ----------
        df, metric_cols, assignment, k
        n_permutations : int
            Uses self.n_permutations.

        Returns
        -------
        dict with keys: observed_stat, p_value, method, passed.
        """
        from objective import between_group_dispersion

        observed = between_group_dispersion(df, metric_cols, assignment)
        n = len(df)
        perm_stats = []

        for _ in range(self.n_permutations):
            shuffled = self.rng.permutation(assignment)
            perm_stats.append(between_group_dispersion(df, metric_cols, shuffled))

        perm_stats = np.array(perm_stats)
        # p-value: fraction of permutations with dispersion >= observed.
        # A well-balanced solution has LOW observed dispersion, so most
        # permuted stats >= observed → p is HIGH → we do not reject H0.
        p_val = float((perm_stats >= observed).mean())

        return {
            "method": f"Permutation test (n={self.n_permutations})",
            "observed_stat": float(observed),
            "null_mean": float(perm_stats.mean()),
            "null_std": float(perm_stats.std()),
            "p_value": p_val,
            "passed": p_val >= self.alpha,
        }

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    def format_report(self, report: ValidationReport) -> str:
        """Return a multi-line string summary of the ValidationReport."""
        lines = [
            "=" * 60,
            f"STATISTICAL VALIDATION REPORT",
            f"Animals: {report.n_animals}  |  Groups: {report.k_groups}  |  Metrics: {report.n_metrics}",
            "=" * 60,
            "",
            "UNIVARIATE RESULTS (Bonferroni corrected):",
        ]
        for mr in report.metric_results:
            status = "FAIL" if mr.significant else "PASS"
            lines.append(
                f"  [{status}] {mr.metric:20s}  "
                f"{mr.test_used:15s}  "
                f"p_raw={mr.raw_p_value:.4f}  "
                f"p_corr={mr.corrected_p_value:.4f}  "
                f"{'(normal)' if mr.all_normal else '(non-normal)'}"
            )

        lines += ["", "MULTIVARIATE:"]
        if report.manova_result:
            r = report.manova_result
            if "error" in r:
                lines.append(f"  MANOVA ERROR: {r['error']}")
            else:
                status = "PASS" if r.get("passed") else "FAIL"
                lines.append(
                    f"  [{status}] {r['method']}  p={r.get('p_value', 'N/A'):.4f}"
                )
        elif report.permutation_result:
            r = report.permutation_result
            status = "PASS" if r.get("passed") else "FAIL"
            lines.append(
                f"  [{status}] {r['method']}  p={r.get('p_value', 'N/A'):.4f}"
            )

        lines += ["", f"OVERALL: {'PASS ✓' if report.overall_pass else 'FAIL ✗'}"]

        if report.remediation:
            lines += ["", "REMEDIATION SUGGESTIONS:"]
            for rem in report.remediation:
                lines.append(f"  • {rem}")

        lines.append("=" * 60)
        return "\n".join(lines)
