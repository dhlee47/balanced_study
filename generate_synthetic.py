"""
generate_synthetic.py — Generate 9 synthetic study CSV variants.

Run with: python generate_synthetic.py

Variants systematically cover:
    - n: small (8-16), medium (24-48), large (60-120)
    - m: few (2-3), moderate (4-6), many (7-12)
    - Missing data: ~20% of variants have 5-15% NaN
    - Distributions: normal, skewed, bimodal
    - Realistic preclinical value ranges (Assumption A08)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "synthetic"
OUTPUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def normal_col(n, mean, std):
    return RNG.normal(mean, std, n)

def skewed_col(n, a, scale, loc=0):
    from scipy.stats import skewnorm
    return skewnorm.rvs(a=a, loc=loc, scale=scale, size=n, random_state=42)

def bimodal_col(n, mu1, mu2, std, mix=0.5):
    mask = RNG.random(n) < mix
    vals = np.where(mask, RNG.normal(mu1, std, n), RNG.normal(mu2, std, n))
    return vals

def ordinal_col(n, levels=3):
    return RNG.integers(1, levels + 1, n).astype(float)

def inject_missing(df, metric_cols, pct):
    df = df.copy()
    n_cells = int(len(df) * len(metric_cols) * pct)
    for _ in range(n_cells):
        r = RNG.integers(0, len(df))
        c = RNG.choice(metric_cols)
        df.at[r, c] = np.nan
    return df

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
# Each variant: (n, metric_specs, missing_pct)
# metric_spec: (col_name, generator_fn)

VARIANTS = [
    # 1: small n, few metrics, normal
    dict(n=10, missing=0.0, metrics=[
        ("body_weight_g",       lambda n: normal_col(n, 24, 3)),
        ("blood_glucose_mmolL", lambda n: normal_col(n, 6.5, 0.8)),
    ]),
    # 2: small n, moderate metrics, skewed
    dict(n=12, missing=0.0, metrics=[
        ("body_weight_g",       lambda n: normal_col(n, 22, 4)),
        ("corticosterone_ngmL", lambda n: skewed_col(n, 5, 80, 100)),
        ("latency_s",           lambda n: skewed_col(n, 3, 20, 15)),
        ("open_arm_pct",        lambda n: skewed_col(n, 2, 10, 10)),
    ]),
    # 3: small n, many metrics, no missing
    dict(n=16, missing=0.0, metrics=[
        ("body_weight_g",       lambda n: normal_col(n, 23, 3)),
        ("blood_glucose_mmolL", lambda n: normal_col(n, 7.0, 1.0)),
        ("litter_number",       lambda n: ordinal_col(n, 3)),
        ("locomotor_activity",  lambda n: normal_col(n, 1500, 300)),
        ("body_temp_C",         lambda n: normal_col(n, 37.2, 0.4)),
        ("organ_weight_pct",    lambda n: normal_col(n, 0.7, 0.1)),
        ("latency_s",           lambda n: skewed_col(n, 3, 15, 20)),
    ]),
    # 4: medium n, few metrics, missing 8%
    dict(n=30, missing=0.08, metrics=[
        ("body_weight_g",       lambda n: normal_col(n, 25, 4)),
        ("blood_glucose_mmolL", lambda n: bimodal_col(n, 5.5, 8.5, 0.6)),
        ("locomotor_activity",  lambda n: normal_col(n, 1800, 400)),
    ]),
    # 5: medium n, moderate metrics, normal
    dict(n=36, missing=0.0, metrics=[
        ("body_weight_g",       lambda n: normal_col(n, 26, 3.5)),
        ("blood_glucose_mmolL", lambda n: normal_col(n, 6.8, 0.9)),
        ("litter_number",       lambda n: ordinal_col(n, 4)),
        ("locomotor_activity",  lambda n: normal_col(n, 1600, 350)),
        ("body_temp_C",         lambda n: normal_col(n, 37.0, 0.5)),
    ]),
    # 6: medium n, many metrics, missing 12%
    dict(n=48, missing=0.12, metrics=[
        ("body_weight_g",       lambda n: normal_col(n, 27, 4)),
        ("blood_glucose_mmolL", lambda n: bimodal_col(n, 5.0, 9.0, 0.8)),
        ("corticosterone_ngmL", lambda n: skewed_col(n, 4, 90, 120)),
        ("latency_s",           lambda n: normal_col(n, 60, 20)),
        ("open_arm_pct",        lambda n: normal_col(n, 25, 10)),
        ("locomotor_activity",  lambda n: normal_col(n, 1700, 450)),
        ("body_temp_C",         lambda n: normal_col(n, 37.1, 0.4)),
        ("organ_weight_pct",    lambda n: normal_col(n, 0.75, 0.12)),
    ]),
    # 7: large n, few metrics, bimodal
    dict(n=72, missing=0.0, metrics=[
        ("body_weight_g",       lambda n: bimodal_col(n, 20, 30, 2)),
        ("blood_glucose_mmolL", lambda n: normal_col(n, 7.2, 1.1)),
    ]),
    # 8: large n, moderate metrics, skewed + missing 6%
    dict(n=96, missing=0.06, metrics=[
        ("body_weight_g",       lambda n: normal_col(n, 28, 5)),
        ("blood_glucose_mmolL", lambda n: skewed_col(n, 6, 1.5, 5)),
        ("corticosterone_ngmL", lambda n: skewed_col(n, 5, 100, 80)),
        ("locomotor_activity",  lambda n: bimodal_col(n, 1000, 2200, 200)),
        ("body_temp_C",         lambda n: normal_col(n, 37.3, 0.3)),
        ("litter_number",       lambda n: ordinal_col(n, 5)),
    ]),
    # 9: large n, many metrics, normal (stress test)
    dict(n=120, missing=0.0, metrics=[
        ("body_weight_g",       lambda n: normal_col(n, 25, 3.5)),
        ("blood_glucose_mmolL", lambda n: normal_col(n, 6.9, 0.8)),
        ("litter_number",       lambda n: ordinal_col(n, 4)),
        ("locomotor_activity",  lambda n: normal_col(n, 1900, 500)),
        ("body_temp_C",         lambda n: normal_col(n, 37.1, 0.45)),
        ("organ_weight_pct",    lambda n: normal_col(n, 0.72, 0.10)),
        ("corticosterone_ngmL", lambda n: skewed_col(n, 4, 85, 100)),
        ("latency_s",           lambda n: normal_col(n, 55, 18)),
        ("open_arm_pct",        lambda n: bimodal_col(n, 15, 35, 5)),
        ("plasma_insulin_ngmL", lambda n: skewed_col(n, 3, 0.4, 0.5)),
        ("adipose_mass_g",      lambda n: normal_col(n, 1.8, 0.5)),
        ("liver_mass_g",        lambda n: normal_col(n, 1.2, 0.2)),
    ]),
]

# Distribution descriptions for MANIFEST
DIST_NOTES = [
    "normal (all metrics)",
    "normal body_weight; skewed corticosterone, latency, open_arm",
    "mixed: normal + ordinal + skewed",
    "normal + bimodal blood_glucose; 8% NaN",
    "all normal",
    "bimodal blood_glucose; skewed corticosterone; 12% NaN",
    "bimodal body_weight; normal blood_glucose",
    "normal + skewed + bimodal locomotor; 6% NaN",
    "all distributions represented; 12 metrics",
]

def generate_variants():
    paths = []
    for i, spec in enumerate(VARIANTS, start=1):
        n = spec["n"]
        missing = spec["missing"]
        metric_specs = spec["metrics"]

        data = {"animal_id": list(range(1, n + 1))}
        for col_name, gen_fn in metric_specs:
            vals = gen_fn(n)
            # Clip to realistic ranges (no negatives for weights etc.)
            vals = np.clip(vals, 0, None)
            data[col_name] = np.round(vals, 4)

        df = pd.DataFrame(data)
        metric_cols = [col for col, _ in metric_specs]

        if missing > 0:
            df = inject_missing(df, metric_cols, missing)

        out_path = OUTPUT_DIR / f"variant_{i:02d}.csv"
        df.to_csv(str(out_path), index=False)
        paths.append(out_path)
        print(f"  variant_{i:02d}.csv  n={n}  m={len(metric_specs)}  missing={missing*100:.0f}%")

    return paths


def write_manifest(paths):
    lines = [
        "# Synthetic Variant Manifest",
        "",
        "| # | File | n (animals) | m (metrics) | Missing % | Distributions |",
        "|---|------|-------------|-------------|-----------|---------------|",
    ]
    for i, (spec, path, note) in enumerate(zip(VARIANTS, paths, DIST_NOTES), start=1):
        n = spec["n"]
        m = len(spec["metrics"])
        miss = f"{spec['missing']*100:.0f}%"
        lines.append(f"| {i} | {path.name} | {n} | {m} | {miss} | {note} |")

    lines += [
        "",
        "## Notes",
        "- All variants use realistic preclinical mouse values (see ASSUMPTIONS.md A08).",
        "- Missing values are injected at random cell positions, not full rows.",
        "- Seed: 42 for reproducibility.",
    ]
    manifest_path = OUTPUT_DIR / "MANIFEST.md"
    manifest_path.write_text("\n".join(lines))
    print(f"\nManifest written: {manifest_path}")


if __name__ == "__main__":
    print("Generating 9 synthetic variants…")
    paths = generate_variants()
    write_manifest(paths)
    print("Done.")
