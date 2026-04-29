# ASSUMPTIONS LOG

Generated automatically during project scaffolding.
All assumptions are numbered for traceability.

---

## A01 — Example CSV Schema Inference

**Observed schema:** `animal id, metric 1, metric 2, metric 3, metric 4`
- `animal id`: integer, sequential 1–30 → treated as the ID column (non-numeric role)
- `metric 1`: integer, range 1–69 → assumed continuous numeric baseline metric
- `metric 2`: float ~6.5–8.0 → assumed continuous numeric baseline metric
- `metric 3`: integer, values {1, 2, 3} only → **assumed ordinal/categorical**
  - Treated as numeric for z-scoring but flagged in validation
- `metric 4`: float ~0.5–1.5 → assumed continuous numeric baseline metric

**n = 30 animals, m = 4 metrics. No missing values in example.**

### Schema improvement suggestions (logged, not auto-applied):
1. Rename `animal id` → `animal_id` (no spaces in column names avoids quoting issues)
2. Rename `metric 1–4` to domain-specific names (e.g., `body_weight_g`, `blood_glucose_mmolL`,
   `litter_number`, `locomotor_activity`) once the user confirms what each metric represents
3. `metric 3` should be declared as `category` dtype if truly categorical, to prevent
   spurious distance calculations in PCA/clustering steps

---

## A02 — GUI Library

**Choice: PyQt6**
- Rationale: richer widget set than tkinter, native look on Windows, better support for
  embedded matplotlib figures via `matplotlib.backends.backend_qtagg`
- Fallback: tkinter is available system-wide if PyQt6 fails to install

---

## A03 — Algorithm Hyperparameter Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| EA generations | 1000 | Sufficient convergence for n≤120; user-adjustable |
| EA population size | 100 | Balances diversity vs. runtime |
| EA tournament size | 3 | Standard competitive selection pressure |
| EA mutation rate | 0.1 | 10% swap probability per individual |
| SA initial temperature | 10.0 | Accepts ~e^(-1/10) ≈ 90% of bad moves initially |
| SA cooling rate | 0.995 | ~1380 iterations to reach T=0.001 |
| SA max iterations | 5000 | Prevents runaway on small datasets |
| PCA components | min(m, n//3, 10) | Enough to capture variance without overfitting |
| Alpha (between-group weight) | 1.0 | Equal emphasis by default |
| Beta (within-group weight) | 1.0 | Equal emphasis by default |
| Gamma (Mahalanobis weight) | 0.5 | Slightly less than dispersion to avoid domination |

---

## A04 — Statistical Test Selection

- **Normality**: Shapiro-Wilk chosen over Kolmogorov-Smirnov because it has higher power for
  small samples (n < 50 per group), which is typical in preclinical studies
- **ANOVA threshold**: p < 0.05 from Shapiro-Wilk → non-parametric (Kruskal-Wallis); else ANOVA
- **MANOVA fallback**: if total n < 20 or any group has < 3 animals, fall back to permutation test
- **Bonferroni correction** applied across m metrics to control family-wise error rate
- **Post-hoc**: Tukey HSD for ANOVA (assumes equal variance), Dunn's test for Kruskal-Wallis

---

## A05 — Missing Data

- NaN rows flagged with a boolean column `_had_missing` appended to the cleaned DataFrame
- Default strategy: median imputation (robust to outliers in small samples)
- KNN imputation uses k=5 neighbors

---

## A06 — Objective Function

- Mahalanobis distance computed between group centroids using pooled covariance
- When n < m+2 (rank-deficient covariance), Mahalanobis term is replaced by Euclidean distance
  with a warning

---

## A07 — Benchmark

- All benchmark runs use k=3 groups (as specified in the prompt)
- Timeout per run: 120 seconds (EA with 1000 generations on large datasets can be slow)
- Random seed: 42 for all stochastic algorithms (reproducibility)

---

## A08 — Synthetic Variant Generation

- Realistic value ranges derived from published mouse/rat preclinical study norms:
  - Body weight (g): 18–35 (mice), 200–350 (rats) — variants use mice range
  - Blood glucose (mmol/L): 4.0–10.0
  - Locomotor activity (beam breaks/h): 500–2500
  - Organ weight (% body weight): 0.3–1.2
  - Plasma corticosterone (ng/mL): 50–400
  - Latency to escape (s): 10–120
  - % Time in open arms (EPM): 5–50
  - Body temperature (°C): 36.0–38.5
- metric 3 in example (ordinal 1–3) corresponds to litter number in synthetic variants

---

## A09 — PDF Report Generation

- reportlab used for PDF export; matplotlib figures are rasterised to PNG then embedded
- Interactive HTML versions use plotly with full offline bundle for portability

---

## A11 — Objective Function Scale Normalisation (Bug Fix)

**Problem identified from benchmark results:**
`p_locomotor_activity = 0.0` (statistically significant group differences) for Algorithms 2 and 3
across variants 03, 04, 05, 06, 08, 09, despite those algorithms explicitly minimising the
objective function. Algorithm 1 always passed because it z-scores internally during serpentine ranking.

**Root cause:**
`between_group_dispersion()` and `within_group_variance()` in `objective.py` operated on raw
metric values. `locomotor_activity` (values ~800–2500, std ~450–600) contributed a raw variance
of magnitude ~10,000–100,000, while `body_weight_g` (std ~3–5) contributed ~1–25. The objective
was therefore ~10,000× dominated by locomotor_activity. Algorithms 2 and 3 over-optimised for
locomotor_activity in absolute terms while barely improving other metrics, yet even with
near-optimal locomotor balance, bimodal structure (variant_08) or residual imbalance still
produced statistically significant ANOVA differences.

**Fix applied (2026-04-25):**
Each metric is now normalised by its pooled standard deviation before computing variance of group
means and within-group variance. This makes the objective function scale-invariant and consistent
with Algorithm 1's internal z-scoring. The Mahalanobis term was already scale-invariant via the
inverse covariance matrix and is unchanged.

**How to apply:**
Any new metric column added by users will automatically benefit from this normalisation regardless
of its unit scale.

---

## A10 — Virtual Environment

- Created at: `C:\Users\dhlee\balanced_study\venv`
- Python interpreter: system Python 3.13 (Microsoft Store)
- All benchmark and generation scripts must be run with:
  `C:\Users\dhlee\balanced_study\venv\Scripts\python.exe`
