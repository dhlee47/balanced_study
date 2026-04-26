# Algorithm Details

## Algorithm 1 — Dynamic Allocation (Serpentine)

**Complexity:** O(n log n)  
**Type:** Deterministic — same input always produces the same output  
**Best for:** Quick runs, large n, when you need reproducibility

### How it works

Think of it like dealing cards around a table, but instead of random order, the cards are first sorted by "how extreme" each animal is.

1. Each animal gets a composite score = weighted average of its z-scored metrics
2. Animals are sorted from lowest to highest composite score
3. They are assigned to groups in a snake pattern: 0→1→2→2→1→0→0→1→2→...

The snake pattern ensures each group gets animals from every part of the score distribution — one "low" animal, one "medium", one "high" — just like dealing cards evenly.

### Objective function not used during assignment
Algorithm 1 doesn't iteratively optimise — it's a one-shot heuristic. The objective function is only computed afterward for reporting.

---

## Algorithm 2 — Evolutionary Algorithm

**Complexity:** O(generations × population_size × n)  
**Type:** Stochastic — results vary by random seed  
**Best for:** Small-to-medium n where thorough search is feasible

### How it works

Inspired by biological evolution:

- **Population:** Start with 100 candidate group assignments
- **Fitness:** Score each candidate using the objective function (lower = better)
- **Selection:** Run tournaments — pick 3 random candidates, keep the best one
- **Crossover:** Combine two "parent" assignments at a random cut point to make a child
- **Mutation:** With 10% probability, swap two randomly chosen animals between groups
- **Elitism:** Always keep the top 10% of candidates unchanged each generation
- **Early stopping:** If no improvement after 100 generations, stop

After 1000 generations (default), return the best assignment found.

### Hyperparameter guidance
- More generations → better quality, slower runtime
- Larger population → more diversity, slower per-generation
- Lower mutation rate → exploitation (refine); higher rate → exploration (diversify)

---

## Algorithm 3 — Stratified Clustering Hybrid (Recommended)

**Complexity:** O(n × m × PCA) + O(max_sa_iter)  
**Type:** Stochastic with structured initialisation  
**Best for:** Datasets with correlated metrics; general use

### Why it's recommended

Algorithms 1 and 2 treat each metric independently. But in preclinical studies, metrics are often correlated (e.g., heavier animals tend to have higher blood glucose). Ignoring correlations can lead to groups that look balanced metric-by-metric but differ in their overall biological profile.

Algorithm 3 solves this by:

1. **PCA decorrelation:** Transform the m correlated metrics into uncorrelated principal components. Now "distance" in this space is meaningful.

2. **k-means++ stratification:** Cluster animals in PCA space. This groups animals that are biologically similar. We then distribute members of each cluster evenly across all k groups — like assigning litter-mates to different groups.

3. **Simulated annealing fine-tuning:** Starting from the k-means assignment, randomly swap pairs of animals. Accept swaps that improve the objective score. Occasionally accept worse swaps (with decreasing probability as the "temperature" cools) to escape local minima.

### The temperature analogy
Imagine a landscape of scores. SA starts at "high temperature" — it walks around randomly, willing to climb hills. As temperature decreases, it becomes more selective. At "cold" temperature it only accepts improvements. This lets it find a global minimum rather than getting stuck in the first valley it encounters.

### Objective function
F = α·D_between + β·V_within + γ·M_mahal

| Term | Meaning | Want it to be |
|------|---------|---------------|
| D_between | Variance of group means per metric | Low (similar means) |
| V_within | Average variance within each group | Low (tight groups) |
| M_mahal | Mahalanobis distance between group centroids | Low (similar covariance) |
