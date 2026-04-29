# Algorithm Details — Stratified Clustering Hybrid

**Complexity:** O(n × m × PCA) + O(max_sa_iter)  
**Type:** Stochastic with structured initialisation  
**Reproducible:** Yes — same random seed always produces the same result (default seed = 42)

---

## Why this approach?

In preclinical studies, baseline metrics are often correlated — for example, heavier animals
tend to have higher blood glucose. An algorithm that treats each metric independently can
produce groups that look balanced metric-by-metric but still differ in their overall biological
profile.

The Stratified Clustering Hybrid solves this by:

1. **Decorrelating metrics with PCA** so that "distance" between animals reflects their full
   biological similarity, not just individual metric differences.
2. **Stratifying with k-means++** to group biologically similar animals together before
   distributing them.
3. **Fine-tuning with simulated annealing** to escape local minima and reach a near-optimal
   assignment.

---

## Step-by-step

### Step 1 — Z-score normalisation
Each metric column is standardised (mean 0, std 1) so that high-range metrics
(e.g. locomotor activity ~500–2500) do not dominate over low-range metrics
(e.g. body weight ~18–35 g).

### Step 2 — PCA decorrelation
The m standardised metrics are projected into uncorrelated principal components.
The number of components is set automatically to min(m, n // 3, 10) — enough to
capture most variance without overfitting to small datasets.

### Step 3 — k-means++ stratification
Animals are clustered in PCA space using k-means++ initialisation. This groups
animals that are biologically similar. We use k × 3 clusters so that each final
group receives a mix of animal types rather than a single cluster.

### Step 4 — Serpentine initialisation
Animals are sorted by their composite z-score and assigned to groups in a snake
pattern (0→1→2→2→1→0→…). This guarantees:
- Each group has exactly the target number of animals.
- Each group covers the full range of the score distribution.

This serves as the starting point for simulated annealing.

### Step 5 — Simulated annealing fine-tuning
Starting from the initialised assignment, the algorithm proposes random animal
swaps between groups. A swap is accepted if it improves the objective score, or
with a small probability even if it does not (the "temperature" parameter controls
this probability and decreases over time). This lets the algorithm escape local
minima rather than getting stuck in the first reasonable solution it finds.

---

## The objective function

F = α · D_between + β · V_within + γ · M_mahal

| Term | What it penalises | Default weight |
|------|--------------------|----------------|
| D_between | Variance of group means per metric | α = 1.0 |
| V_within | Average variance within each group per metric | β = 1.0 |
| M_mahal | Mahalanobis distance between group centroids | γ = 0.5 |

All three terms are scale-invariant (normalised by pooled standard deviation or
inverse covariance matrix), so metrics with very different units are treated equally.

---

## The temperature analogy

Simulated annealing is named after the physical process of slowly cooling a metal.
At high temperature, atoms move freely and the metal is malleable. As it cools, atoms
settle into a low-energy (stable) configuration.

In the algorithm: "temperature" starts high (the algorithm accepts many imperfect
swaps, exploring widely) and decreases each iteration. At low temperature, only
improvements are accepted. This helps find a global minimum rather than getting stuck
in the first local valley encountered.

**Tuning:**
- Lower cooling rate (e.g. 0.99 vs 0.995) → slower cooling → more thorough search, slower
- Higher initial temperature → more exploration early on
- If the first run fails validation, lower the cooling rate or enable Continuous Improvement mode

---

## Hyperparameter defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| SA initial temperature | 10.0 | Accepts ~e^(−1/10) ≈ 90% of bad moves initially |
| SA cooling rate | 0.995 | ~1,380 iterations to reach T = 0.001 |
| SA max iterations | 5,000 | Sufficient for n ≤ 120; avoids runaway on small datasets |
| PCA components | min(m, n//3, 10) | Captures variance without overfitting |
| α (between-group) | 1.0 | Equal emphasis by default |
| β (within-group) | 1.0 | Equal emphasis by default |
| γ (Mahalanobis) | 0.5 | Slightly less than dispersion to avoid domination |
