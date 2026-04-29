# GUI Walkthrough

## Launching the Application

```
# From the project root
venv\Scripts\python.exe src\gui.py
```

---

## Panel 1 — Input

**[Browse CSV…]** Opens a file picker. Select your study CSV.

After loading:
- The first 10 rows appear in the preview table
- Auto-detected ID and metric columns are shown
- Any warnings (e.g., ordinal columns, missing data) appear in the Warnings box

**ID column dropdown** — Override the auto-detected ID column if needed.  
**Missing data handling** — Choose how to handle NaN values before balancing:
  - `median` (default): fill NaN with column median — robust for small samples
  - `mean`: fill with column mean
  - `exclude`: drop any row with at least one NaN
  - `knn`: K-nearest-neighbour imputation (most sophisticated, slower)

Click **[Confirm Data & Proceed to Configuration →]** when ready.

---

## Panel 2 — Configuration

**Number of groups (k)** — How many groups to create (2–20).  
**Group Names** — Click each box to rename groups (e.g., "Control", "Treatment A").

**Algorithm:** Stratified Clustering Hybrid — PCA decorrelation, k-means++ stratification,
simulated annealing fine-tuning. See [algorithm_details.md](algorithm_details.md) for a full explanation.

**Algorithm parameters:**
- **Initial temperature**: how freely the algorithm explores at the start (default 10.0)
- **Cooling rate**: how quickly it narrows its search (default 0.995); lower values (e.g. 0.99) give a more thorough but slower search

**Advanced Weights (click to expand):**
- **Alpha**: how much to penalise different group *means* (default 1.0)
- **Beta**: how much to penalise *spread within groups* (default 1.0)  
- **Gamma**: how much to penalise different *covariance structure* (default 0.5)
- **Per-metric sliders**: give more weight to metrics that matter most biologically (e.g., body weight might get 2.0× if it's the primary endpoint)

**Continuous Improvement Mode:**
When enabled, the algorithm reruns up to N times, each time with a different random seed. Stops as soon as all statistical tests pass. Use this when the first run fails validation.

---

## Panel 3 — Run & Monitor

Click **[▶ Run Balancing]** to start. The GUI stays responsive while the algorithm runs in a background thread.

- **Progress bar** shows percent completion
- **Live log** shows iteration updates and scores
- **Composite Score** badge appears when done
- **PASS ✓ / FAIL ✗** badge (green/red) shows statistical validation result
- **Convergence plot** (CI mode): score over iterations — should decrease and plateau

---

## Panel 4 — Results

**Groups Table** — All animals with their assigned group. Sortable.

**Distributions tab** — KDE + rug plots per metric per group. Look for overlapping distributions (good) vs. separated distributions (bad).

**Covariance tab** — Correlation heatmaps per group. Similar heatmaps across groups = well-balanced correlation structure.

**PCA tab** — Animals in PC1–PC2 space, coloured by group. Well-balanced groups should interleave rather than cluster by group.

**Stats Report tab** — Full text of the statistical validation report with per-metric p-values and overall pass/fail.

**[Export Group CSVs…]** — Save each group as a separate CSV file.  
**[Export PDF Report…]** — Save a PDF with all figures embedded.

---

## Tips

- If you get FAIL on one metric, increase that metric's importance slider (e.g., to 2.0) and rerun
- If MANOVA fails but all univariate tests pass, the groups differ in their *joint* distribution — try Algorithm 3 with a lower cooling rate (0.99)
- For n < 12, statistical tests have low power — a PASS result is expected even with random assignment
