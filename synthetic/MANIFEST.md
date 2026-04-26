# Synthetic Variant Manifest

| # | File | n (animals) | m (metrics) | Missing % | Distributions |
|---|------|-------------|-------------|-----------|---------------|
| 1 | variant_01.csv | 10 | 2 | 0% | normal (all metrics) |
| 2 | variant_02.csv | 12 | 4 | 0% | normal body_weight; skewed corticosterone, latency, open_arm |
| 3 | variant_03.csv | 16 | 7 | 0% | mixed: normal + ordinal + skewed |
| 4 | variant_04.csv | 30 | 3 | 8% | normal + bimodal blood_glucose; 8% NaN |
| 5 | variant_05.csv | 36 | 5 | 0% | all normal |
| 6 | variant_06.csv | 48 | 8 | 12% | bimodal blood_glucose; skewed corticosterone; 12% NaN |
| 7 | variant_07.csv | 72 | 2 | 0% | bimodal body_weight; normal blood_glucose |
| 8 | variant_08.csv | 96 | 6 | 6% | normal + skewed + bimodal locomotor; 6% NaN |
| 9 | variant_09.csv | 120 | 12 | 0% | all distributions represented; 12 metrics |

## Notes
- All variants use realistic preclinical mouse values (see ASSUMPTIONS.md A08).
- Missing values are injected at random cell positions, not full rows.
- Seed: 42 for reproducibility.