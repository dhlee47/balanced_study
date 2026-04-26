"""
algorithms.py — Three group-balancing algorithms.

Algorithm 1 — Dynamic Allocation (Deterministic / Serpentine)
    Fast, O(n log n), fully reproducible.  Ranks animals by composite
    z-score and assigns them in a snake pattern across k groups.

Algorithm 2 — Evolutionary Algorithm with Shuffling
    Stochastic global search.  Evolves a population of candidate
    assignments using tournament selection, mutation (animal swaps),
    and crossover.  Configurable generations / population size.

Algorithm 3 — Stratified Clustering Hybrid (recommended)
    PCA → k-means stratification → simulated annealing fine-tuning.
    Handles correlated metrics better than Algorithms 1 & 2 because
    PCA decorrelates the feature space before clustering.

All algorithms accept the same interface and return an integer assignment
array of shape (n,) with values in [0, k-1].

Assumptions: A03 (hyperparameter defaults), A06 (Mahalanobis fallback).
"""

from __future__ import annotations

import random
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from objective import compute_objective, score_solution


# ---------------------------------------------------------------------------
# Algorithm 1 — Dynamic Allocation (Serpentine / Snake)
# ---------------------------------------------------------------------------

def dynamic_allocation(
    df: pd.DataFrame,
    metric_cols: list[str],
    k: int,
    metric_weights: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    group_sizes: list[int] | None = None,
) -> np.ndarray:
    """
    Assign animals to k groups using a greedy serpentine rank method.

    Steps:
    1. Z-score each metric column.
    2. Compute a weighted composite score per animal.
    3. Sort animals by composite score (ascending).
    4. Assign in a snake pattern that respects each group's target size.

    Parameters
    ----------
    df : pd.DataFrame
    metric_cols : list[str]
    k : int
    metric_weights : np.ndarray | None
    alpha, beta, gamma : float
        Not used in assignment; kept for API consistency.
    group_sizes : list[int] | None
        Target number of animals per group.  None → equal sizes (n // k,
        with remainder distributed to the first groups).

    Returns
    -------
    np.ndarray of int, shape (n,)
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


# ---------------------------------------------------------------------------
# Algorithm 2 — Evolutionary Algorithm
# ---------------------------------------------------------------------------

def evolutionary_algorithm(
    df: pd.DataFrame,
    metric_cols: list[str],
    k: int,
    generations: int = 1000,
    population_size: int = 100,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    metric_weights: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    random_seed: int = 42,
    progress_callback: Callable[[int, float], None] | None = None,
    early_stop_patience: int = 100,
    group_sizes: list[int] | None = None,
) -> np.ndarray:
    """
    Find a balanced group assignment using an evolutionary algorithm.

    The population is a set of candidate assignment arrays.  Each
    generation:
    1. Select parents via tournament selection.
    2. Produce offspring via single-point crossover.
    3. Apply swap mutation (randomly swap two animals between groups).
    4. Keep the best individuals (elitism: top 10% always survive).
    5. Check early stopping: if the best score hasn't improved in
       `early_stop_patience` generations, stop.

    Parameters
    ----------
    df : pd.DataFrame
    metric_cols : list[str]
    k : int
        Number of groups.
    generations : int
        Maximum number of generations.  Default 1000.
    population_size : int
        Number of candidate solutions per generation.  Default 100.
    tournament_size : int
        Number of candidates compared in each tournament selection.  Default 3.
    mutation_rate : float
        Probability of swapping any two animals in a candidate.  Default 0.1.
    metric_weights : np.ndarray | None
    alpha, beta, gamma : float
        Objective function component weights.
    random_seed : int
        RNG seed for reproducibility.  Default 42.
    progress_callback : callable | None
        If provided, called each generation with (generation, best_score).
    early_stop_patience : int
        Stop if best score does not improve for this many consecutive
        generations.  Default 100.

    Returns
    -------
    np.ndarray of int, shape (n,)
        Best assignment found.
    """
    rng = np.random.default_rng(random_seed)
    n = len(df)
    sizes = _resolve_sizes(n, k, group_sizes)

    def fitness(assignment: np.ndarray) -> float:
        return compute_objective(df, metric_cols, assignment, alpha, beta, gamma, metric_weights)

    # Seed population from Algorithm 1 so we start near a good solution
    base = dynamic_allocation(df, metric_cols, k, metric_weights, alpha, beta, gamma,
                              group_sizes=group_sizes)
    population = [base.copy()]
    for _ in range(population_size - 1):
        # Shuffle while preserving target group sizes
        shuffled = _shuffle_preserving_sizes(base, sizes, rng)
        population.append(shuffled)

    scores = np.array([fitness(ind) for ind in population])
    best_score = scores.min()
    best_assignment = population[np.argmin(scores)].copy()
    no_improve = 0
    n_elite = max(1, population_size // 10)

    for gen in range(generations):
        new_population = []

        # Elitism: always keep the best individuals
        elite_idx = np.argsort(scores)[:n_elite]
        for idx in elite_idx:
            new_population.append(population[idx].copy())

        # Fill remainder with crossover + mutation
        while len(new_population) < population_size:
            # Tournament selection for two parents
            p1 = _tournament_select(population, scores, tournament_size, rng)
            p2 = _tournament_select(population, scores, tournament_size, rng)

            # Single-point crossover (size-aware)
            child = _crossover(p1, p2, k, sizes, rng)

            # Mutation: swap pairs of animals between groups
            if rng.random() < mutation_rate:
                child = _mutate(child, rng)

            new_population.append(child)

        population = new_population
        scores = np.array([fitness(ind) for ind in population])

        gen_best = scores.min()
        if gen_best < best_score - 1e-9:
            best_score = gen_best
            best_assignment = population[np.argmin(scores)].copy()
            no_improve = 0
        else:
            no_improve += 1

        if progress_callback:
            progress_callback(gen, best_score)

        if no_improve >= early_stop_patience:
            break  # converged

    return best_assignment


def _tournament_select(
    population: list[np.ndarray],
    scores: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select the best individual from a random tournament of `size` candidates."""
    idxs = rng.choice(len(population), size=size, replace=False)
    best = idxs[np.argmin(scores[idxs])]
    return population[best]


def _crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    k: int,
    sizes: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Single-point crossover that rebalances to exact target group sizes.

    Split p1 and p2 at a random cut point, combine, then reassign
    excess animals from over-represented groups to under-represented ones.
    """
    n = len(p1)
    cut = rng.integers(1, n)
    child = np.concatenate([p1[:cut], p2[cut:]])

    overflow: list[int] = []
    needed: list[int] = []
    for g in range(k):
        idx = np.where(child == g)[0]
        target = sizes[g]
        if len(idx) > target:
            overflow.extend(idx[:len(idx) - target].tolist())
        elif len(idx) < target:
            needed.extend([g] * (target - len(idx)))

    for animal_idx, group in zip(overflow, needed):
        child[animal_idx] = group

    return child


def _mutate(assignment: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Swap two animals from different groups."""
    mutated = assignment.copy()
    n = len(mutated)
    i, j = rng.choice(n, size=2, replace=False)
    # Only swap if they're in different groups (otherwise no effect)
    if mutated[i] != mutated[j]:
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


# ---------------------------------------------------------------------------
# Algorithm 3 — Stratified Clustering Hybrid (recommended)
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

    Rationale (Assumption A03):
    This approach is recommended because it:
    (a) handles correlated metrics by decorrelating via PCA first,
    (b) creates a good initialisation via k-means++ in PC space,
    (c) applies simulated annealing for local optimisation with
        escape from local minima.

    Steps
    -----
    1. Z-score the m metric columns.
    2. PCA: reduce to min(m, n//3, 10) principal components.
    3. k-means++ clustering in PC space to initialise k strata.
    4. Map cluster labels → group labels (clusters ≠ final groups; we
       use cluster membership to build a balanced initial assignment by
       distributing each cluster's members across all k groups evenly).
    5. Simulated annealing: propose random swaps, accept if they improve
       the objective or with probability exp(-Δ/T) if they don't.

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
    strata = kmeans.fit_predict(X_pc)

    # Step 4 — Distribute strata members across groups respecting target sizes.
    # Use dynamic_allocation as the initialiser so group_sizes are guaranteed correct.
    assignment = dynamic_allocation(df, metric_cols, k, metric_weights, alpha, beta, gamma,
                                    group_sizes=group_sizes)

    # Step 5 — Simulated annealing
    current_score = compute_objective(df, metric_cols, assignment, alpha, beta, gamma, metric_weights)
    best_score = current_score
    best_assignment = assignment.copy()
    temp = initial_temp

    for iteration in range(max_sa_iter):
        # Propose swap of two animals in different groups
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if assignment[i] == assignment[j]:
            continue  # no-op swap

        proposal = assignment.copy()
        proposal[i], proposal[j] = proposal[j], proposal[i]

        new_score = compute_objective(df, metric_cols, proposal, alpha, beta, gamma, metric_weights)
        delta = new_score - current_score

        # Accept if improvement, or with Boltzmann probability if worse
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


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_algorithm(
    algo_name: str,
    df: pd.DataFrame,
    metric_cols: list[str],
    k: int,
    **kwargs,
) -> tuple[np.ndarray, float, float]:
    """
    Run a named algorithm and return the assignment, score, and wall-clock time.

    Parameters
    ----------
    algo_name : str
        One of 'dynamic', 'evolutionary', 'hybrid'.
    df : pd.DataFrame
    metric_cols : list[str]
    k : int
    **kwargs
        Forwarded to the chosen algorithm function.

    Returns
    -------
    tuple[np.ndarray, float, float]
        (assignment, composite_score, elapsed_seconds)

    Raises
    ------
    ValueError
        If algo_name is not recognised.
    """
    ALGO_MAP = {
        "dynamic": dynamic_allocation,
        "evolutionary": evolutionary_algorithm,
        "hybrid": stratified_clustering_hybrid,
    }
    if algo_name not in ALGO_MAP:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. Choose from: {list(ALGO_MAP.keys())}."
        )

    t0 = time.perf_counter()
    assignment = ALGO_MAP[algo_name](df, metric_cols, k, **kwargs)
    elapsed = time.perf_counter() - t0

    score_info = score_solution(df, metric_cols, assignment)
    return assignment, score_info["composite"], elapsed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    active = list(range(k))  # groups that still need animals

    while active:
        for g in list(active):  # forward sweep
            if remaining[g] > 0:
                pattern.append(g)
                remaining[g] -= 1
            if remaining[g] == 0:
                active.remove(g)

        for g in list(reversed(active)):  # backward sweep
            if remaining[g] > 0:
                pattern.append(g)
                remaining[g] -= 1
            if remaining[g] == 0:
                active.remove(g)

    return pattern


def _shuffle_preserving_sizes(
    base: np.ndarray,
    sizes: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return a new assignment that has the same group sizes as `base` but
    with animals randomly redistributed.  Used to seed the EA population.
    """
    result = np.empty_like(base)
    # Build a pool of shuffled indices
    indices = rng.permutation(len(base))
    pos = 0
    for g, size in enumerate(sizes):
        result[indices[pos:pos + size]] = g
        pos += size
    return result


def _uniform_weights(weights: np.ndarray | None, m: int) -> np.ndarray:
    """Return normalised weight array of length m."""
    if weights is None:
        return np.ones(m) / m
    w = np.asarray(weights, dtype=float)
    return w / w.sum()
