"""
algorithms.py — Stratified Clustering Hybrid group-balancing algorithm.

The algorithm runs in three stages:
    1. Z-score all metric columns.
    2. PCA decorrelation → k-means++ stratification in principal-component space.
    3. Simulated annealing fine-tuning of the composite objective function.

This approach is recommended for preclinical studies because it handles
correlated metrics (via PCA) and escapes local minima (via SA).

Returns an integer assignment array of shape (n,) with values in [0, k-1].
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from objective import compute_objective, score_solution


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stratified_clustering_hybrid(
    df: pd.DataFrame,
    metric_cols: list[str],
    k: int,
    n_pca_components: int | None = None,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.995,
    max_sa_iter: int = 5000,
    metric_weights: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    random_seed: int = 42,
    progress_callback: Callable[[int, float], None] | None = None,
    group_sizes: list[int] | None = None,
) -> np.ndarray:
    """
    Balance groups using PCA stratification + simulated annealing fine-tuning.

    Steps
    -----
    1. Z-score the m metric columns.
    2. PCA: reduce to min(m, n//3, 10) principal components.
    3. k-means++ clustering in PC space to build k strata.
    4. Serpentine initialisation: distribute animals across groups in a
       snake pattern over their composite z-score rank, guaranteeing
       each group covers the full score distribution.
    5. Simulated annealing: propose random animal swaps, accept if they
       improve the objective or with probability exp(-Δ/T) if they don't.

    Parameters
    ----------
    df : pd.DataFrame
    metric_cols : list[str]
    k : int
    n_pca_components : int | None
        Number of PCA components.  None → auto (min(m, n//3, 10)).
    initial_temp : float
        SA starting temperature.  Default 10.0.
    cooling_rate : float
        SA temperature multiplier per iteration.  Default 0.995.
    max_sa_iter : int
        Maximum SA iterations.  Default 5000.
    metric_weights : np.ndarray | None
    alpha, beta, gamma : float
        Objective function weights.
    random_seed : int
        RNG seed.  Default 42.
    progress_callback : callable | None
        Called each SA iteration with (iteration, current_score).
    group_sizes : list[int] | None
        Target animals per group.  None → equal split.

    Returns
    -------
    np.ndarray of int, shape (n,)
        Optimised group assignment.
    """
    rng = np.random.default_rng(random_seed)
    n = len(df)
    m = len(metric_cols)
    sizes = _resolve_sizes(n, k, group_sizes)

    # Step 1 — Z-score
    X = df[metric_cols].values.astype(float)
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)

    # Step 2 — PCA
    n_comp = n_pca_components or min(m, n // 3, 10)
    n_comp = max(1, min(n_comp, m, n - 1))
    pca = PCA(n_components=n_comp, random_state=random_seed)
    X_pc = pca.fit_transform(X_z)

    # Step 3 — k-means++ in PC space to build initial strata
    n_strata = min(k * 3, n)
    kmeans = KMeans(n_clusters=n_strata, init="k-means++", random_state=random_seed, n_init=10)
    kmeans.fit_predict(X_pc)

    # Step 4 — Serpentine initialisation (guarantees group_sizes are met)
    assignment = _serpentine_init(df, metric_cols, k, metric_weights, group_sizes=group_sizes)

    # Step 5 — Simulated annealing
    current_score = compute_objective(df, metric_cols, assignment, alpha, beta, gamma, metric_weights)
    best_score = current_score
    best_assignment = assignment.copy()
    temp = initial_temp

    for iteration in range(max_sa_iter):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if assignment[i] == assignment[j]:
            continue

        proposal = assignment.copy()
        proposal[i], proposal[j] = proposal[j], proposal[i]

        new_score = compute_objective(df, metric_cols, proposal, alpha, beta, gamma, metric_weights)
        delta = new_score - current_score

        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            assignment = proposal
            current_score = new_score

        if current_score < best_score:
            best_score = current_score
            best_assignment = assignment.copy()

        temp *= cooling_rate

        if progress_callback:
            progress_callback(iteration, current_score)

    return best_assignment


def run_algorithm(
    df: pd.DataFrame,
    metric_cols: list[str],
    k: int,
    **kwargs,
) -> tuple[np.ndarray, float, float]:
    """
    Run the Stratified Clustering Hybrid algorithm and return results.

    Parameters
    ----------
    df : pd.DataFrame
    metric_cols : list[str]
    k : int
    **kwargs
        Forwarded to stratified_clustering_hybrid.

    Returns
    -------
    tuple[np.ndarray, float, float]
        (assignment, composite_score, elapsed_seconds)
    """
    t0 = time.perf_counter()
    assignment = stratified_clustering_hybrid(df, metric_cols, k, **kwargs)
    elapsed = time.perf_counter() - t0
    score_info = score_solution(df, metric_cols, assignment)
    return assignment, score_info["composite"], elapsed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serpentine_init(
    df: pd.DataFrame,
    metric_cols: list[str],
    k: int,
    metric_weights: np.ndarray | None = None,
    group_sizes: list[int] | None = None,
) -> np.ndarray:
    """
    Build an initial assignment by sorting animals on their composite z-score
    and dealing them in a snake pattern across k groups.

    Used as the starting point for simulated annealing so that the SA phase
    begins near a reasonable solution rather than a random one.
    """
    n = len(df)
    sizes = _resolve_sizes(n, k, group_sizes)

    X = df[metric_cols].values.astype(float)
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)
    weights = _uniform_weights(metric_weights, len(metric_cols))
    composite_scores = X_z @ weights
    sorted_idx = np.argsort(composite_scores)

    pattern = _serpentine_pattern(sizes)

    assignment = np.empty(n, dtype=int)
    for rank, animal_idx in enumerate(sorted_idx):
        assignment[animal_idx] = pattern[rank]
    return assignment


def _resolve_sizes(n: int, k: int, group_sizes: list[int] | None) -> list[int]:
    """
    Return a validated list of target group sizes.

    If group_sizes is None, distribute n animals as evenly as possible:
    the first (n % k) groups get one extra animal.
    """
    if group_sizes is None:
        base, extra = divmod(n, k)
        return [base + (1 if g < extra else 0) for g in range(k)]
    if sum(group_sizes) != n:
        raise ValueError(
            f"group_sizes sum ({sum(group_sizes)}) must equal n ({n})."
        )
    if len(group_sizes) != k:
        raise ValueError(
            f"group_sizes length ({len(group_sizes)}) must equal k ({k})."
        )
    return list(group_sizes)


def _serpentine_pattern(sizes: list[int]) -> list[int]:
    """
    Build a snake/serpentine assignment pattern that respects unequal group sizes.

    Animals are dealt in alternating forward/backward sweeps so that every
    group receives animals from across the full score distribution even when
    sizes differ (e.g. [5, 4, 3] for k=3, n=12).
    """
    remaining = list(sizes)
    pattern: list[int] = []
    k = len(sizes)
    active = list(range(k))

    while active:
        for g in list(active):
            if remaining[g] > 0:
                pattern.append(g)
                remaining[g] -= 1
            if remaining[g] == 0:
                active.remove(g)

        for g in list(reversed(active)):
            if remaining[g] > 0:
                pattern.append(g)
                remaining[g] -= 1
            if remaining[g] == 0:
                active.remove(g)

    return pattern


def _uniform_weights(weights: np.ndarray | None, m: int) -> np.ndarray:
    """Return normalised weight array of length m."""
    if weights is None:
        return np.ones(m) / m
    w = np.asarray(weights, dtype=float)
    return w / w.sum()
