"""
data_loader.py — CSV ingestion, column detection, and missing-data handling.

Responsibilities:
    - Load any CSV with variable n animals and m metrics
    - Auto-detect which column is the animal ID vs. which are numeric metrics
    - Detect and handle missing data (flag, impute, or exclude)
    - Validate inputs and emit human-readable warnings

Assumption A01: The ID column is the first column whose name contains 'id'
(case-insensitive) or, failing that, the first non-numeric column.
Assumption A05: Default imputation strategy is median.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StudyDataLoader:
    """
    Load a preclinical study CSV and prepare it for balancing.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file.
    id_col : str | None
        Name of the animal-ID column.  If None, auto-detected.
    metric_cols : list[str] | None
        Names of the metric columns.  If None, all numeric columns
        except the ID column are used.

    Attributes
    ----------
    raw_df : pd.DataFrame
        Original data as loaded from disk, before any cleaning.
    df : pd.DataFrame
        Cleaned data after missing-data handling.
    id_col : str
        Detected or user-supplied ID column name.
    metric_cols : list[str]
        Detected or user-supplied metric column names.
    warnings : list[str]
        Human-readable warning messages accumulated during loading.

    Examples
    --------
    >>> loader = StudyDataLoader("example.csv")
    >>> loader.load()
    >>> print(loader.n, loader.m)
    30 4
    """

    def __init__(
        self,
        filepath: str | Path,
        id_col: str | None = None,
        metric_cols: list[str] | None = None,
    ) -> None:
        self.filepath = Path(filepath)
        self._user_id_col = id_col
        self._user_metric_cols = metric_cols

        self.raw_df: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.id_col: str = ""
        self.metric_cols: list[str] = []
        self.warnings: list[str] = []
        self._missing_mask: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Core load pipeline
    # ------------------------------------------------------------------

    def load(self) -> "StudyDataLoader":
        """
        Load the CSV from disk and run the full detection + validation pipeline.

        Returns
        -------
        StudyDataLoader
            Self, for method chaining.

        Raises
        ------
        FileNotFoundError
            If the CSV path does not exist.
        ValueError
            If no numeric metric columns can be detected.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV not found: {self.filepath}")

        self.raw_df = pd.read_csv(self.filepath)
        self.id_col, self.metric_cols = self.detect_columns()
        self._missing_mask = self.raw_df[self.metric_cols].isna()
        self._validate()
        return self

    def detect_columns(self) -> tuple[str, list[str]]:
        """
        Infer the ID column and metric columns from the raw DataFrame.

        ID detection priority:
        1. User-supplied id_col
        2. First column whose name contains 'id' (case-insensitive)
        3. First column that is NOT entirely numeric

        Metric detection: all remaining numeric columns after ID is removed.

        Returns
        -------
        tuple[str, list[str]]
            (id_column_name, [metric_column_names])

        Raises
        ------
        ValueError
            If no metric columns are found.
        """
        df = self.raw_df

        # --- ID column ---
        if self._user_id_col:
            id_col = self._user_id_col
        else:
            id_candidates = [c for c in df.columns if "id" in c.lower()]
            if id_candidates:
                id_col = id_candidates[0]
            else:
                # Fall back to first non-numeric column
                non_numeric = [
                    c for c in df.columns
                    if not pd.api.types.is_numeric_dtype(df[c])
                ]
                id_col = non_numeric[0] if non_numeric else df.columns[0]

        # --- Metric columns ---
        if self._user_metric_cols:
            metric_cols = self._user_metric_cols
        else:
            metric_cols = [
                c for c in df.columns
                if c != id_col and pd.api.types.is_numeric_dtype(df[c])
            ]

        if not metric_cols:
            raise ValueError(
                f"No numeric metric columns found after excluding ID column '{id_col}'."
            )

        return id_col, metric_cols

    # ------------------------------------------------------------------
    # Missing data
    # ------------------------------------------------------------------

    def get_missing_summary(self) -> pd.DataFrame:
        """
        Return a summary of missing values per column.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['column', 'n_missing', 'pct_missing'].
        """
        if self._missing_mask is None:
            raise RuntimeError("Call load() before get_missing_summary().")

        records = []
        for col in self.metric_cols:
            n_miss = self._missing_mask[col].sum()
            pct = 100.0 * n_miss / len(self.raw_df)
            records.append({"column": col, "n_missing": int(n_miss), "pct_missing": round(pct, 1)})
        return pd.DataFrame(records)

    def handle_missing(
        self,
        strategy: Literal["exclude", "mean", "median", "knn"] = "median",
    ) -> pd.DataFrame:
        """
        Handle missing values in the metric columns.

        Parameters
        ----------
        strategy : {'exclude', 'mean', 'median', 'knn'}
            How to handle rows/cells with NaN values:
            - 'exclude' : drop rows that have any NaN in metric columns
            - 'mean'    : replace NaN with column mean
            - 'median'  : replace NaN with column median (default; robust to outliers)
            - 'knn'     : K-nearest-neighbour imputation (k=5)

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with the ID column and imputed/filtered metrics.

        Raises
        ------
        RuntimeError
            If load() has not been called.
        ValueError
            If strategy is not one of the accepted values.
        """
        if self.raw_df is None:
            raise RuntimeError("Call load() before handle_missing().")

        df = self.raw_df.copy()
        any_missing = self._missing_mask.any().any()

        if not any_missing:
            self.df = df
            return self.df

        n_missing_rows = self._missing_mask.any(axis=1).sum()
        self.warnings.append(
            f"Missing data detected: {n_missing_rows} rows have at least one NaN "
            f"across metric columns. Strategy: '{strategy}'."
        )

        if strategy == "exclude":
            mask = ~self._missing_mask.any(axis=1)
            df = df[mask].reset_index(drop=True)
            self.warnings.append(
                f"Excluded {n_missing_rows} rows with missing data. "
                f"Remaining: {len(df)} animals."
            )

        elif strategy == "mean":
            for col in self.metric_cols:
                df[col] = df[col].fillna(df[col].mean())

        elif strategy == "median":
            for col in self.metric_cols:
                df[col] = df[col].fillna(df[col].median())

        elif strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
            df[self.metric_cols] = imputer.fit_transform(df[self.metric_cols])

        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose from: 'exclude', 'mean', 'median', 'knn'."
            )

        # Add a flag column to track which rows had imputed values
        df["_had_missing"] = self._missing_mask.any(axis=1).values

        self.df = df
        return self.df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Run internal checks and populate self.warnings."""
        df = self.raw_df

        # Check for duplicate IDs
        if df[self.id_col].duplicated().any():
            self.warnings.append(
                f"Duplicate values found in ID column '{self.id_col}'. "
                "This may cause issues downstream."
            )

        # Flag ordinal-looking columns (integer, few unique values)
        for col in self.metric_cols:
            n_unique = df[col].nunique()
            if pd.api.types.is_integer_dtype(df[col]) and n_unique <= 10:
                self.warnings.append(
                    f"Column '{col}' has only {n_unique} unique integer values. "
                    "It may be ordinal/categorical. Verify it should be treated as numeric."
                )

        # Warn if n is very small
        n = len(df)
        if n < 6:
            self.warnings.append(
                f"Very small dataset: n={n}. Statistical tests will have very low power."
            )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of animals."""
        if self.raw_df is None:
            raise RuntimeError("Call load() first.")
        return len(self.raw_df)

    @property
    def m(self) -> int:
        """Number of metric columns."""
        return len(self.metric_cols)

    def get_clean_df(self, strategy: str = "median") -> pd.DataFrame:
        """
        Convenience: load + handle missing + return clean DataFrame.

        Parameters
        ----------
        strategy : str
            Missing-data strategy passed to handle_missing().

        Returns
        -------
        pd.DataFrame
            Clean DataFrame ready for balancing.
        """
        if self.df is None:
            self.handle_missing(strategy)
        return self.df

    def get_summary(self) -> dict:
        """
        Return a dictionary summarising the loaded dataset.

        Returns
        -------
        dict
            Keys: n, m, id_col, metric_cols, missing_pct, warnings.
        """
        missing_pct = 0.0
        if self._missing_mask is not None:
            total_cells = self._missing_mask.size
            missing_pct = 100.0 * self._missing_mask.sum().sum() / total_cells

        return {
            "n": self.n,
            "m": self.m,
            "id_col": self.id_col,
            "metric_cols": self.metric_cols,
            "missing_pct": round(missing_pct, 2),
            "warnings": self.warnings,
        }

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        s = self.get_summary()
        print(f"=== Dataset Summary ===")
        print(f"  File       : {self.filepath.name}")
        print(f"  Animals    : {s['n']}")
        print(f"  Metrics    : {s['m']}  →  {s['metric_cols']}")
        print(f"  ID column  : {s['id_col']}")
        print(f"  Missing    : {s['missing_pct']}%")
        if s["warnings"]:
            print("  Warnings:")
            for w in s["warnings"]:
                print(f"    ⚠  {w}")
