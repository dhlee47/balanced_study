"""
test_data_loader.py — Unit tests for data_loader.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import StudyDataLoader


@pytest.fixture
def tmp_csv(tmp_path):
    """Create a minimal valid CSV for testing."""
    df = pd.DataFrame({
        "animal_id": range(1, 11),
        "weight": np.random.normal(24, 3, 10),
        "glucose": np.random.normal(7, 1, 10),
        "score": np.random.randint(1, 4, 10).astype(float),
    })
    path = tmp_path / "test.csv"
    df.to_csv(str(path), index=False)
    return path


@pytest.fixture
def tmp_csv_missing(tmp_path):
    """CSV with missing values."""
    df = pd.DataFrame({
        "animal_id": range(1, 11),
        "weight": [24.0, np.nan, 23.0, 25.0, 22.0, np.nan, 26.0, 24.0, 25.0, 23.0],
        "glucose": np.random.normal(7, 1, 10),
    })
    path = tmp_path / "missing.csv"
    df.to_csv(str(path), index=False)
    return path


class TestStudyDataLoader:

    def test_load_basic(self, tmp_csv):
        loader = StudyDataLoader(tmp_csv)
        loader.load()
        assert loader.n == 10
        assert loader.m == 3
        assert loader.id_col == "animal_id"
        assert "weight" in loader.metric_cols

    def test_auto_id_detection(self, tmp_csv):
        loader = StudyDataLoader(tmp_csv)
        loader.load()
        assert loader.id_col == "animal_id"

    def test_user_override_id_col(self, tmp_csv):
        loader = StudyDataLoader(tmp_csv, id_col="weight")
        loader.load()
        assert loader.id_col == "weight"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            StudyDataLoader("nonexistent.csv").load()

    def test_missing_median_imputation(self, tmp_csv_missing):
        loader = StudyDataLoader(tmp_csv_missing)
        loader.load()
        df = loader.handle_missing("median")
        assert df["weight"].isna().sum() == 0
        assert len(loader.warnings) > 0

    def test_missing_exclusion(self, tmp_csv_missing):
        loader = StudyDataLoader(tmp_csv_missing)
        loader.load()
        df = loader.handle_missing("exclude")
        # 2 rows had NaN → 8 remain
        assert len(df) == 8

    def test_missing_mean_imputation(self, tmp_csv_missing):
        loader = StudyDataLoader(tmp_csv_missing)
        loader.load()
        df = loader.handle_missing("mean")
        assert df["weight"].isna().sum() == 0

    def test_missing_knn_imputation(self, tmp_csv_missing):
        loader = StudyDataLoader(tmp_csv_missing)
        loader.load()
        df = loader.handle_missing("knn")
        assert df["weight"].isna().sum() == 0

    def test_invalid_strategy(self, tmp_csv):
        loader = StudyDataLoader(tmp_csv)
        loader.load()
        with pytest.raises(ValueError):
            loader.handle_missing("invalid_strategy")

    def test_summary_structure(self, tmp_csv):
        loader = StudyDataLoader(tmp_csv)
        loader.load()
        summary = loader.get_summary()
        assert "n" in summary
        assert "m" in summary
        assert "id_col" in summary
        assert "metric_cols" in summary
        assert "missing_pct" in summary

    def test_ordinal_warning(self, tmp_csv):
        """Integer column with few unique values should trigger a warning."""
        loader = StudyDataLoader(tmp_csv)
        loader.load()
        ordinal_warnings = [w for w in loader.warnings if "score" in w.lower() or "unique" in w.lower()]
        assert len(ordinal_warnings) > 0
