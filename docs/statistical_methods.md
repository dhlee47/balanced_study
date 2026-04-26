# Statistical Methods

## Why validate after balancing?

A good group assignment should produce groups where no metric differs significantly between groups. Validation checks this formally using the same statistical tests a reviewer would expect to see in your methods section.

---

## Normality Testing — Shapiro-Wilk

**What it asks:** Is this data consistent with a normal (bell-curve) distribution?  
**Why Shapiro-Wilk:** More powerful than Kolmogorov-Smirnov for small samples (n < 50 per group), which is typical in preclinical work.  
**Threshold:** p < 0.05 → treat as non-normal

---

## Univariate Group Comparison

### If data is normal: One-way ANOVA
Tests whether the means of k groups are all equal.
- H₀: μ₁ = μ₂ = ... = μₖ (what we want — all group means equal)
- For a well-balanced assignment, p should be HIGH (> 0.05 after Bonferroni)

### If data is non-normal: Kruskal-Wallis Test
The non-parametric equivalent of ANOVA. Compares group median ranks rather than means. More robust when data is skewed or has outliers.

---

## Multiple Testing Correction — Bonferroni

When testing m metrics simultaneously, the chance of finding at least one false positive grows. Bonferroni correction multiplies each p-value by m (the number of metrics). This is conservative — it controls the family-wise error rate.

If a corrected p-value is still < 0.05, the groups genuinely differ on that metric.

---

## Post-Hoc Tests

Run only when the omnibus test (ANOVA or KW) is significant — identifies which pairs of groups differ.

| Omnibus | Post-hoc | Correction |
|---------|----------|------------|
| ANOVA | Tukey HSD | Built-in |
| Kruskal-Wallis | Dunn's test | Bonferroni |

---

## Multivariate Testing

### MANOVA (used when n ≥ 20 and min group size ≥ 3)
Tests whether group centroids differ in the full m-dimensional space simultaneously. More powerful than running m separate ANOVAs when metrics are correlated.

**Wilks' Lambda (λ):** Ranges from 0 to 1. Values near 1 → groups are similar (good). Small λ with small p → groups differ.

### Permutation Test (fallback for small n)
When sample sizes are too small for MANOVA:
- Compute the observed between-group dispersion score
- Shuffle group labels 999 times and recompute each time
- p-value = fraction of shuffled scores ≤ observed
- A high p-value means the observed balance is at least as good as random chance

---

## Box's M Test
Tests whether the covariance matrices are equal across groups. Used as a supplementary check. Less critical for balancing but useful for detecting distributional differences not captured by means alone.

---

## Interpretation Guide

| Outcome | Meaning | Action |
|---------|---------|--------|
| All p > 0.05 (corrected) | Groups are statistically equivalent | Report as PASS; use this assignment |
| 1 metric p < 0.05 | One metric is unbalanced | Increase that metric's weight and re-run |
| Multiple metrics p < 0.05 | Poor overall balance | Run Algorithm 3 with more SA iterations |
| MANOVA p < 0.05 | Multivariate separation detected | Enable Continuous Improvement mode |

---

## Manuscript Methods Paragraph

> Animals were assigned to experimental groups using the balanced_study software (v1.0.0). Briefly, baseline metrics were z-scored and submitted to the Stratified Clustering Hybrid algorithm, which applies Principal Component Analysis for dimensionality reduction, k-means++ stratification in principal component space, and simulated annealing local optimisation of a composite objective function penalising between-group mean dispersion, within-group variance, and Mahalanobis distance between group centroids (weights α=β=1.0, γ=0.5). Statistical validity of the final assignment was confirmed by one-way ANOVA or Kruskal-Wallis test (selected based on per-group Shapiro-Wilk normality testing) for each metric, with Bonferroni correction for multiple comparisons, and multivariate MANOVA (Wilks' lambda) across all metrics simultaneously. The null hypothesis of group equivalence was not rejected for any metric (all corrected p > 0.05).
