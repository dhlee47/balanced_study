"""
objective.py — Unified objective function for evaluating group-assignment quality.

All three balancing algorithms optimise the same composite score so results are
directly comparable.

Score components
----------------
1. Between-group dispersion  (lower = better balanced means across groups)
2. Within-group variance      (lower = tighter, more homogeneous groups)
3. Mahalanobis distance       (lower = more similar covariance structure)

Composite score F = alpha * D_between + beta * V_within + gamma * M_mahal

Assumption A06: When n < m+2 (rank-deficient covariance), Mahalanobis is
replaced by scaled Euclidean distance and a warning is emitted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis


# ---------------------------------------------------------------------------
# Component functions
# ---------------------------------------------------------------------------

def between_group_dispersion(
    df: pd.DataFrame,
    metric_cols: list[str],
    assignment: np.ndarray,
    metric_weights: np.ndarray | None = None,
) -> float:
    """
    Compute weighted variance of per-group means for each metric, then average.

    This penalises solutions where group means differ substantially — the ideal
    solution has identical group means for every metric.

    Parameters
    ----------
    df : pd.DataFrame
        Animal data (rows = animals, columns include metric_cols).
    metric_cols : list[str]
        Names of the metric columns to evaluate.
    assignment : np.ndarray of int, shape (n,)
        Group label for each animal (0-indexed).
    metric_weights : np.ndarray | None
        Per-metric importance weights.  None → uniform weights.

    Returns
    -------
    float
        Weighted mean of per-metric between-group variance of means.
    """
    m = len(metric_cols)
    weights = _normalise_weights(metric_weights, m)
    k = len(np.unique(assignment))

    dispersions = []
    for col, w in zip(metric_cols, weights):
        vals = df[col].values.astype(float)
        std = vals.std()
        # Normalise by pooled std so high-range metrics (e.g. locomotor_activity ~1000-2500)
        # don't dominate over low-range metrics (e.g. body_weight ~18-35).
        # Without this, the objective is ~10,000x larger for locomotor_activity, causing
        # algorithms 2 & 3 to over-optimise for it in absolute terms while neglecting others.
        # Algorithm 1 already z-scores internally — this makes 2 & 3 consistent. (Fix A11)
        if std < 1e-10:
            dispersions.append(0.0)
            continue
        group_means = [vals[assignment == g].mean() / std for g in range(k)]
        dispersions.append(w * np.var(group_means, ddof=0))

    return float(np.sum(dispersions))


def within_group_variance(
    df: pd.DataFrame,
    metric_cols: list[str],
    assignment: np.ndarray,
    metric_weights: np.ndarray | None = None,
) -> float:
    """
    Compute weighted mean of within-group variance per metric.

    Parameters
    ----------
    df : pd.DataFrame
    metric_cols : list[str]
    assignment : np.ndarray of int, shape (n,)
    metric_weights : np.ndarray | None

    Returns
    -------
    float
        Weighted average within-group variance across all metrics and groups.
    """
    m = len(metric_cols)
    weights = _normalise_weights(metric_weights, m)
    k = len(np.unique(assignment))

    variances = []
    for col, w in zip(metric_cols, weights):
        pooled_std = df[col].values.astype(float).std()
        if pooled_std < 1e-10:
            variances.append(0.0)
            continue
        group_vars = []
        for g in range(k):
            vals = df[col].values[assignment == g].astype(float) / pooled_std
            if len(vals) > 1:
                group_vars.append(np.var(vals, ddof=1))
            else:
                group_vars.append(0.0)
        variances.append(w * np.mean(group_vars))

    return float(np.sum(variances))


def mahalanobis_distance_between_groups(
    df: pd.DataFrame,
    metric_cols: list[str],
    assignment: np.ndarray,
) -> float:
    """
    Compute mean pairwise Mahalanobis distance between group centroids.

    Uses the pooled covariance matrix across all groups.  Falls back to
    scaled Euclidean if the covariance matrix is singular (rank-deficient).

    Parameters
    ----------
    df : pd.DataFrame
    metric_cols : list[str]
    assignment : np.ndarray of int, shape (n,)

    Returns
    -------
    float
        Mean pairwise Mahalanobis distance (0 = identical centroids).
    """
    k = len(np.unique(assignment))
    X = df[metric_cols].values.astype(float)
    m = len(metric_cols)
    n = len(X)

    # Compute group centroids
    centroids = np.array([X[assignment == g].mean(axis=0) for g in range(k)])

    # Pooled covariance
    if n > m + 1:
        try:
            cov = np.cov(X.T, ddof=1)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            inv_cov = np.linalg.inv(cov + np.eye(m) * 1e-8)  # ridge for stability

            total_dist = 0.0
            n_pairs = 0
            for i in range(k):
                for j in range(i + 1, k):
                    diff = centroids[i] - centroids[j]
                    d = float(diff @ inv_cov @ diff) ** 0.5
                    total_dist += d
                    n_pairs += 1
            return total_dist / n_pairs if n_pairs > 0 else 0.0

        except np.linalg.LinAlgError:
            pass  # fall through to Euclidean fallback

    # Euclidean fallback when covariance is singular
    warnings.warn(
        "Covariance matrix is rank-deficient (n < m+2). "
        "Using scaled Euclidean distance instead of Mahalanobis. (Assumption A06)",
        RuntimeWarning,
        stacklevel=2,
    )
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero for constant columns

    total_dist = 0.0
    n_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            diff = (centroids[i] - centroids[j]) / stds
            total_dist += np.linalg.norm(diff)
            n_pairs += 1
    return total_dist / n_pairs if n_pairs > 0 else 0.0


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_objective(
    df: pd.DataFrame,
    metric_cols: list[str],
    assignment: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    metric_weights: np.ndarray | None = None,
) -> float:
    """
    Compute the composite objective score F for a group assignment.

    F = alpha * D_between + beta * V_within + gamma * M_mahal

    Lower is better.

    Parameters
    ----------
    df : pd.DataFrame
        Animal data.
    metric_cols : list[str]
        Metric column names.
    assignment : np.ndarray of int, shape (n,)
        Group label (0-indexed) for each animal.
    alpha : float
        Weight on between-group mean dispersion.  Default 1.0.
    beta : float
        Weight on within-group variance.  Default 1.0.
    gamma : float
        Weight on Mahalanobis distance between group centroids.  Default 0.5.
    metric_weights : np.ndarray | None
        Per-metric importance weights (length m).  None → uniform.

    Returns
    -------
    float
        Composite score.  0 = perfect balance; larger = worse.

    Examples
    --------
    >>> import numpy as np
    >>> score = compute_objective(df, ['w', 'g'], np.array([0,1,0,1,0,1]))
    """
    d = between_group_dispersion(df, metric_cols, assignment, metric_weights)
    v = within_group_variance(df, metric_cols, assignment, metric_weights)
    mh = mahalanobis_distance_between_groups(df, metric_cols, assignment)
    return alpha * d + beta * v + gamma * mh


def score_solution(
    df: pd.DataFrame,
    metric_cols: list[str],
    assignment: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    metric_weights: np.ndarray | None = None,
) -> dict:
    """
    Return a detailed breakdown of the objective score components.

    Parameters
    ----------
    df : pd.DataFrame
    metric_cols : list[str]
    assignment : np.ndarray
    alpha, beta, gamma : float
        Component weights.
    metric_weights : np.ndarray | None

    Returns
    -------
    dict
        Keys: 'composite', 'between_dispersion', 'within_variance',
              'mahalanobis', 'group_sizes', 'group_means'.
    """
    k = len(np.unique(assignment))
    d = between_group_dispersion(df, metric_cols, assignment, metric_weights)
    v = within_group_variance(df, metric_cols, assignment, metric_weights)
    mh = mahalanobis_distance_between_groups(df, metric_cols, assignment)
    composite = alpha * d + beta * v + gamma * mh

    group_sizes = {g: int((assignment == g).sum()) for g in range(k)}
    group_means = {
        g: {col: float(df[col].values[assignment == g].mean()) for col in metric_cols}
        for g in range(k)
    }

    return {
        "composite": composite,
        "between_dispersion": d,
        "within_variance": v,
        "mahalanobis": mh,
        "group_sizes": group_sizes,
        "group_means": group_means,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_weights(weights: np.ndarray | None, m: int) -> np.ndarray:
    """
    Return a normalised weight array of length m.

    Parameters
    ----------
    weights : np.ndarray | None
        Raw weights.  None → uniform (all 1/m).
    m : int
        Number of metrics.

    Returns
    -------
    np.ndarray
        Weights summing to 1.0.
    """
    if weights is None:
        return np.ones(m) / m
    w = np.asarray(weights, dtype=float)
    if len(w) != m:
        raise ValueError(f"metric_weights length {len(w)} != number of metrics {m}.")
    if w.sum() == 0:
        return np.ones(m) / m
    return w / w.sum()
