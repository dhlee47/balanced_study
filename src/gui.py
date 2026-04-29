"""
gui.py — PyQt6 desktop application for the balanced_study toolkit.

Four-panel layout:
    Panel 1 — Input:        Load CSV, preview data, override column detection
    Panel 2 — Config:       k groups, algorithm, hyperparameters, weights
    Panel 3 — Run/Monitor:  Progress, live log, score + pass/fail badge
    Panel 4 — Results:      Tabs for tables, plots, stats report, export

Continuous Improvement Mode (toggle):
    Automatically re-runs the algorithm until all statistical tests pass
    or max iterations is reached.  Shows a convergence plot.

Assumption A02: PyQt6 is the preferred GUI library.
"""

from __future__ import annotations

import sys
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Matplotlib Qt backend — must be set before pyplot import
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
    QRadioButton, QButtonGroup, QGroupBox, QTextEdit, QProgressBar,
    QTabWidget, QScrollArea, QCheckBox, QLineEdit, QSplitter,
    QSizePolicy, QFrame, QMessageBox, QStatusBar,
)

# Add src/ to path so local imports work when run directly
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import StudyDataLoader
from algorithms import run_algorithm, stratified_clustering_hybrid
from objective import score_solution
from stats_validator import StatisticalValidator
from visualizer import Visualizer


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class BalancingWorker(QThread):
    """
    Run the balancing algorithm (and optional continuous improvement) in a
    background thread so the GUI stays responsive.

    Signals
    -------
    progress_updated(int, str)
        (percent_complete, log_message)
    result_ready(dict)
        Emitted when computation finishes.  Dict contains all results.
    error_occurred(str)
        Emitted on unhandled exception.
    """

    progress_updated = pyqtSignal(int, str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        df: pd.DataFrame,
        metric_cols: list[str],
        k: int,
        algo_kwargs: dict,
        alpha: float,
        beta: float,
        gamma: float,
        metric_weights: np.ndarray | None,
        continuous: bool,
        max_ci_iterations: int,
        output_dir: Path,
        run_id: str,
        group_names: list[str],
        id_col: str,
        group_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.df = df
        self.metric_cols = metric_cols
        self.k = k
        self.algo_kwargs = algo_kwargs
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.metric_weights = metric_weights
        self.continuous = continuous
        self.max_ci_iterations = max_ci_iterations
        self.output_dir = output_dir
        self.run_id = run_id
        self.group_names = group_names
        self.id_col = id_col
        self.group_sizes = group_sizes
        self._stop_requested = False

    def stop(self) -> None:
        """Request graceful stop."""
        self._stop_requested = True

    def run(self) -> None:
        """Main thread body — runs algorithm and validation."""
        try:
            validator = StatisticalValidator()
            convergence_scores = []
            best_assignment = None
            best_score = float("inf")
            best_report = None

            n_iterations = self.max_ci_iterations if self.continuous else 1

            for iteration in range(n_iterations):
                if self._stop_requested:
                    break

                self.progress_updated.emit(
                    int(100 * iteration / n_iterations),
                    f"Iteration {iteration + 1}/{n_iterations} — running Stratified Clustering Hybrid…",
                )

                # Vary seed per iteration for diversity
                kwargs = dict(self.algo_kwargs)
                if "random_seed" in kwargs:
                    kwargs["random_seed"] = 42 + iteration

                assignment, score, elapsed = run_algorithm(
                    self.df, self.metric_cols, self.k,
                    metric_weights=self.metric_weights,
                    alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                    group_sizes=self.group_sizes,
                    **kwargs,
                )

                convergence_scores.append(score)
                self.progress_updated.emit(
                    int(100 * (iteration + 0.5) / n_iterations),
                    f"  Score: {score:.4f}  |  Elapsed: {elapsed:.2f}s",
                )

                # Validate
                report = validator.validate(self.df, self.metric_cols, assignment, self.k)
                self.progress_updated.emit(
                    int(100 * (iteration + 0.7) / n_iterations),
                    f"  Validation: {'PASS' if report.overall_pass else 'FAIL'}",
                )

                if score < best_score:
                    best_score = score
                    best_assignment = assignment.copy()
                    best_report = report

                if self.continuous and report.overall_pass:
                    self.progress_updated.emit(100, "All tests passed — stopping early.")
                    break

            # Each run gets its own subdirectory named by run_id
            run_dir = self.output_dir / self.run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # Auto-save a single master CSV in original row order with group column
            # inserted immediately after the ID column
            self.progress_updated.emit(88, "Saving master CSV…")
            out_df = self.df.copy()
            group_labels = [self.group_names[g] for g in best_assignment]
            id_pos = out_df.columns.get_loc(self.id_col)
            out_df.insert(id_pos + 1, "group", group_labels)
            csv_path = run_dir / f"{self.run_id}_assignments.csv"
            out_df.to_csv(str(csv_path), index=False)

            # Generate figures
            self.progress_updated.emit(90, "Generating figures…")
            viz = Visualizer(self.df, self.metric_cols, best_assignment, self.k,
                             group_names=self.group_names)
            saved_figs = viz.save_all(
                run_dir,
                metric_results=best_report.metric_results if best_report else None,
                manova_result=(best_report.manova_result or best_report.permutation_result) if best_report else None,
                boxm_result=best_report.boxm_result if best_report else None,
                prefix=self.run_id,
            )

            score_details = score_solution(
                self.df, self.metric_cols, best_assignment,
                self.alpha, self.beta, self.gamma, self.metric_weights,
            )

            self.progress_updated.emit(100, f"Done. Outputs saved to: {run_dir}")
            self.result_ready.emit({
                "assignment": best_assignment,
                "score_details": score_details,
                "report": best_report,
                "convergence_scores": convergence_scores,
                "saved_figs": saved_figs,
                "viz": viz,
                "run_id": self.run_id,
                "run_dir": str(run_dir),
            })

        except Exception:
            self.error_occurred.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Main application window with four panels in a tabbed / splitter layout.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Balanced Study — Preclinical Group Balancing Toolkit")
        self.setMinimumSize(1100, 750)

        self._loader: StudyDataLoader | None = None
        self._df: pd.DataFrame | None = None
        self._assignment: np.ndarray | None = None
        self._run_dir: Path | None = None
        self._worker: BalancingWorker | None = None
        self._metric_weight_sliders: list[QSlider] = []
        self._output_dir = Path.home() / "balanced_study_outputs"

        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top-level tabs: one per panel
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_panel_input(),   "1 — Input")
        self.tabs.addTab(self._build_panel_config(),  "2 — Configuration")
        self.tabs.addTab(self._build_panel_run(),     "3 — Run & Monitor")
        self.tabs.addTab(self._build_panel_results(), "4 — Results")
        main_layout.addWidget(self.tabs)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — load a CSV to begin.")

    # --- Panel 1: Input ---

    def _build_panel_input(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # File browser row
        file_row = QHBoxLayout()
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        btn_browse = QPushButton("Browse CSV…")
        btn_browse.clicked.connect(self._browse_csv)
        file_row.addWidget(QLabel("CSV File:"))
        file_row.addWidget(self.lbl_file)
        file_row.addWidget(btn_browse)
        layout.addLayout(file_row)

        # Column override row
        col_row = QHBoxLayout()
        self.combo_id_col = QComboBox()
        self.combo_id_col.setMinimumWidth(120)
        col_row.addWidget(QLabel("ID column:"))
        col_row.addWidget(self.combo_id_col)
        col_row.addStretch()

        self.lbl_metrics = QLabel("Metric columns: (auto-detected)")
        col_row.addWidget(self.lbl_metrics)
        layout.addLayout(col_row)

        # Missing data handling
        miss_row = QHBoxLayout()
        self.combo_missing = QComboBox()
        self.combo_missing.addItems(["median (default)", "mean", "exclude", "knn"])
        miss_row.addWidget(QLabel("Missing data handling:"))
        miss_row.addWidget(self.combo_missing)
        miss_row.addStretch()
        layout.addLayout(miss_row)

        # Preview table
        layout.addWidget(QLabel("Data Preview (first 10 rows):"))
        self.preview_table = QTableWidget()
        self.preview_table.setMinimumHeight(200)
        layout.addWidget(self.preview_table)

        # Warnings box
        layout.addWidget(QLabel("Warnings / Notes:"))
        self.warnings_box = QTextEdit()
        self.warnings_box.setReadOnly(True)
        self.warnings_box.setMaximumHeight(100)
        layout.addWidget(self.warnings_box)

        btn_confirm = QPushButton("Confirm Data & Proceed to Configuration →")
        btn_confirm.clicked.connect(self._confirm_data)
        layout.addWidget(btn_confirm)

        return w

    # --- Panel 2: Configuration ---

    def _build_panel_config(self) -> QWidget:
        w = QScrollArea()
        inner = QWidget()
        layout = QVBoxLayout(inner)

        # k spinner
        k_row = QHBoxLayout()
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 20)
        self.spin_k.setValue(3)
        self.spin_k.valueChanged.connect(self._update_group_names)
        k_row.addWidget(QLabel("Number of groups (k):"))
        k_row.addWidget(self.spin_k)
        k_row.addStretch()
        layout.addLayout(k_row)

        # Group names + sizes
        self.group_names_box = QGroupBox("Group Names & Sizes")
        gn_outer = QVBoxLayout()
        self.group_names_layout = QHBoxLayout()  # populated by _update_group_names
        self._size_total_label = QLabel("Total: 0 / ?")
        self._size_total_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        gn_outer.addLayout(self.group_names_layout)
        gn_outer.addWidget(self._size_total_label)
        self.group_names_box.setLayout(gn_outer)
        layout.addWidget(self.group_names_box)
        self._group_name_edits: list[QLineEdit] = []
        self._group_size_spins: list[QSpinBox] = []
        self._update_group_names(3)

        # Algorithm info
        algo_box = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout()
        algo_label = QLabel(
            "Stratified Clustering Hybrid  —  PCA decorrelation → k-means++ stratification "
            "→ simulated annealing optimisation"
        )
        algo_label.setWordWrap(True)
        algo_layout.addWidget(algo_label)
        algo_box.setLayout(algo_layout)
        layout.addWidget(algo_box)

        # SA parameters
        self.algo_params_box = QGroupBox("Algorithm Parameters")
        self.algo_params_layout = QGridLayout()
        self.algo_params_box.setLayout(self.algo_params_layout)

        self.lbl_temp = QLabel("Initial temperature:")
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(0.1, 100.0)
        self.spin_temp.setValue(10.0)
        self.lbl_cool = QLabel("Cooling rate:")
        self.spin_cool = QDoubleSpinBox()
        self.spin_cool.setRange(0.90, 0.9999)
        self.spin_cool.setValue(0.995)
        self.spin_cool.setDecimals(4)

        for row, (lbl, widget) in enumerate([
            (self.lbl_temp, self.spin_temp),
            (self.lbl_cool, self.spin_cool),
        ]):
            self.algo_params_layout.addWidget(lbl, row, 0)
            self.algo_params_layout.addWidget(widget, row, 1)

        layout.addWidget(self.algo_params_box)

        # Advanced weights — toggle button shows/hides the inner widget
        self.btn_weights_toggle = QPushButton("▶  Advanced Weights (click to expand)")
        self.btn_weights_toggle.setCheckable(True)
        self.btn_weights_toggle.setChecked(False)
        self.btn_weights_toggle.setStyleSheet(
            "QPushButton { text-align: left; padding: 4px 8px; font-weight: bold; }"
            "QPushButton:checked { background: #e0e8f0; }"
        )
        layout.addWidget(self.btn_weights_toggle)

        weights_inner = QWidget()
        weights_inner.setVisible(False)
        w_layout = QGridLayout(weights_inner)

        self.lbl_alpha = QLabel("Alpha (between-group weight): 1.0")
        self.slider_alpha = _make_slider(10, 30, 10)
        self.slider_alpha.valueChanged.connect(
            lambda v: self.lbl_alpha.setText(f"Alpha (between-group weight): {v/10:.1f}")
        )
        self.lbl_beta = QLabel("Beta (within-group weight): 1.0")
        self.slider_beta = _make_slider(10, 30, 10)
        self.slider_beta.valueChanged.connect(
            lambda v: self.lbl_beta.setText(f"Beta (within-group weight): {v/10:.1f}")
        )
        self.lbl_gamma = QLabel("Gamma (covariance weight): 0.5")
        self.slider_gamma = _make_slider(1, 30, 5)
        self.slider_gamma.valueChanged.connect(
            lambda v: self.lbl_gamma.setText(f"Gamma (covariance weight): {v/10:.1f}")
        )

        for row, (lbl, slider) in enumerate([
            (self.lbl_alpha, self.slider_alpha),
            (self.lbl_beta,  self.slider_beta),
            (self.lbl_gamma, self.slider_gamma),
        ]):
            w_layout.addWidget(lbl, row, 0)
            w_layout.addWidget(slider, row, 1)

        # Per-metric weight sliders (populated when data is loaded)
        self.metric_weights_box = QGroupBox("Per-Metric Importance (1.0 = equal)")
        self.metric_weights_layout = QGridLayout()
        self.metric_weights_box.setLayout(self.metric_weights_layout)
        w_layout.addWidget(self.metric_weights_box, 3, 0, 1, 2)

        layout.addWidget(weights_inner)
        self.btn_weights_toggle.toggled.connect(
            lambda checked: (
                self.btn_weights_toggle.setText(
                    ("▼  Advanced Weights (click to collapse)" if checked
                     else "▶  Advanced Weights (click to expand)")
                ),
                weights_inner.setVisible(checked),
            )
        )

        # Continuous improvement
        ci_box = QGroupBox("Continuous Improvement Mode")
        ci_layout = QHBoxLayout()
        self.chk_continuous = QCheckBox("Enable")
        self.spin_max_iter = QSpinBox(); self.spin_max_iter.setRange(1, 50); self.spin_max_iter.setValue(10)
        ci_layout.addWidget(self.chk_continuous)
        ci_layout.addWidget(QLabel("Max iterations:"))
        ci_layout.addWidget(self.spin_max_iter)
        ci_layout.addStretch()
        ci_box.setLayout(ci_layout)
        layout.addWidget(ci_box)

        layout.addStretch()
        w.setWidget(inner)
        w.setWidgetResizable(True)
        return w

    # --- Panel 3: Run & Monitor ---

    def _build_panel_run(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Balancing")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self._run_balancing)
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_worker)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Score display
        score_row = QHBoxLayout()
        self.lbl_score = QLabel("Composite Score: —")
        self.lbl_score.setFont(QFont("Monospace", 12))
        self.lbl_pass = QLabel("   —   ")
        self.lbl_pass.setAutoFillBackground(True)
        self.lbl_pass.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_pass.setMinimumWidth(80)
        score_row.addWidget(self.lbl_score)
        score_row.addWidget(self.lbl_pass)
        score_row.addStretch()
        layout.addLayout(score_row)

        layout.addWidget(QLabel("Run Log:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Monospace", 9))
        layout.addWidget(self.log_box)

        # Convergence plot (shown when CI mode is active)
        self.convergence_canvas = FigureCanvas(plt.Figure(figsize=(6, 2)))
        self.convergence_canvas.setMinimumHeight(150)
        layout.addWidget(QLabel("Convergence (CI mode):"))
        layout.addWidget(self.convergence_canvas)

        return w

    # --- Panel 4: Results ---

    def _build_panel_results(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.results_tabs = QTabWidget()

        # Groups table tab
        self.groups_table = QTableWidget()
        self.results_tabs.addTab(self.groups_table, "Groups Table")

        # Plot tabs — container widgets so we can swap in a fresh canvas each run
        self._dist_container = _PlotContainer()
        self.results_tabs.addTab(self._dist_container, "Distributions")

        self._cov_container = _PlotContainer()
        self.results_tabs.addTab(self._cov_container, "Covariance")

        self._pca_container = _PlotContainer()
        self.results_tabs.addTab(self._pca_container, "PCA")

        # Stats report
        self.stats_report_box = QTextEdit()
        self.stats_report_box.setReadOnly(True)
        self.stats_report_box.setFont(QFont("Monospace", 9))
        self.results_tabs.addTab(self.stats_report_box, "Stats Report")

        layout.addWidget(self.results_tabs)

        # Export row
        export_row = QHBoxLayout()
        btn_export_csv = QPushButton("Export Group CSVs…")
        btn_export_csv.clicked.connect(self._export_csv)
        btn_export_pdf = QPushButton("Export PDF Report…")
        btn_export_pdf.clicked.connect(self._export_pdf)
        export_row.addWidget(btn_export_csv)
        export_row.addWidget(btn_export_pdf)
        export_row.addStretch()
        layout.addLayout(export_row)

        return w

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", str(Path.home()), "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            self._loader = StudyDataLoader(path)
            self._loader.load()
            missing_strategy = self._missing_strategy()
            self._df = self._loader.get_clean_df(missing_strategy)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self.lbl_file.setText(path)

        # Populate ID column combo
        self.combo_id_col.clear()
        if self._loader.raw_df is not None:
            self.combo_id_col.addItems(list(self._loader.raw_df.columns))
            idx = list(self._loader.raw_df.columns).index(self._loader.id_col)
            self.combo_id_col.setCurrentIndex(idx)

        self.lbl_metrics.setText(f"Metric columns: {self._loader.metric_cols}")

        # Preview table
        preview = self._loader.raw_df.head(10)
        self.preview_table.setRowCount(len(preview))
        self.preview_table.setColumnCount(len(preview.columns))
        self.preview_table.setHorizontalHeaderLabels(list(preview.columns))
        for r, row in preview.iterrows():
            for c, val in enumerate(row):
                self.preview_table.setItem(r, c, QTableWidgetItem(str(val)))

        # Warnings
        if self._loader.warnings:
            self.warnings_box.setText("\n".join(self._loader.warnings))
        else:
            self.warnings_box.setText("No warnings.")

        # Refresh group size spinbox defaults and limits to match new n
        self._update_group_names(self.spin_k.value())

        # Populate metric weight sliders
        self._populate_metric_sliders()

        # Reset any previous run results so Panel 4 doesn't show stale data
        self._assignment = None
        self.groups_table.clearContents()
        self.groups_table.setRowCount(0)
        self.groups_table.setColumnCount(0)
        self.stats_report_box.clear()
        self.lbl_score.setText("Composite Score: —")
        self.lbl_pass.setText("   —   ")
        self.lbl_pass.setStyleSheet("")
        self.progress_bar.setValue(0)
        self.log_box.clear()

        self.status_bar.showMessage(
            f"Loaded: {self._loader.n} animals × {self._loader.m} metrics"
        )

    def _missing_strategy(self) -> str:
        mapping = {"median (default)": "median", "mean": "mean",
                   "exclude": "exclude", "knn": "knn"}
        return mapping.get(self.combo_missing.currentText(), "median")

    def _confirm_data(self) -> None:
        if self._df is None:
            QMessageBox.warning(self, "No Data", "Please load a CSV first.")
            return
        self.tabs.setCurrentIndex(1)
        self.status_bar.showMessage("Data confirmed. Configure and run balancing.")

    def _update_group_names(self, k: int) -> None:
        # Clear old widgets
        for edit in self._group_name_edits:
            self.group_names_layout.removeWidget(edit)
            edit.deleteLater()
        for spin in self._group_size_spins:
            self.group_names_layout.removeWidget(spin)
            spin.deleteLater()
        # Also remove any QLabel separators
        while self.group_names_layout.count():
            item = self.group_names_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._group_name_edits.clear()
        self._group_size_spins.clear()

        n = len(self._df) if self._df is not None else 0
        base, extra = divmod(n, k) if n > 0 else (0, 0)

        for g in range(k):
            col_widget = QWidget()
            col_layout = QVBoxLayout(col_widget)
            col_layout.setContentsMargins(4, 0, 4, 0)
            col_layout.setSpacing(2)

            edit = QLineEdit(f"Group {g}")
            edit.setMaximumWidth(110)
            edit.setPlaceholderText(f"Group {g}")

            default_size = base + (1 if g < extra else 0)
            spin = QSpinBox()
            spin.setRange(0, max(n, 9999))
            spin.setValue(default_size)
            spin.setMaximumWidth(110)
            spin.valueChanged.connect(self._update_size_total)

            col_layout.addWidget(QLabel(f"G{g} name:"))
            col_layout.addWidget(edit)
            col_layout.addWidget(QLabel("Size:"))
            col_layout.addWidget(spin)

            self.group_names_layout.addWidget(col_widget)
            self._group_name_edits.append(edit)
            self._group_size_spins.append(spin)

        self._update_size_total()

    def _update_size_total(self) -> None:
        """Refresh the total-vs-n label; highlight red if mismatch."""
        n = len(self._df) if self._df is not None else 0
        total = sum(s.value() for s in self._group_size_spins)
        self._size_total_label.setText(f"Total: {total} / {n}")
        ok = (total == n) or (n == 0)
        self._size_total_label.setStyleSheet(
            "color: #1a9850; font-weight: bold;" if ok else "color: #d73027; font-weight: bold;"
        )

    def _get_group_sizes(self) -> list[int] | None:
        """Return per-group sizes if custom, else None (equal split)."""
        if not self._group_size_spins or self._df is None:
            return None
        sizes = [s.value() for s in self._group_size_spins]
        n = len(self._df)
        k = self.spin_k.value()
        # Check if it's just the default equal split
        base, extra = divmod(n, k)
        default = [base + (1 if g < extra else 0) for g in range(k)]
        return sizes if sizes != default else None

    def _populate_metric_sliders(self) -> None:
        if self._loader is None:
            return
        # Clear old
        for slider in self._metric_weight_sliders:
            slider.deleteLater()
        self._metric_weight_sliders.clear()

        for row, col in enumerate(self._loader.metric_cols):
            lbl = QLabel(f"{col}: 1.0")
            slider = _make_slider(1, 30, 10)
            slider.valueChanged.connect(
                lambda v, l=lbl, c=col: l.setText(f"{c}: {v/10:.1f}")
            )
            self.metric_weights_layout.addWidget(lbl, row, 0)
            self.metric_weights_layout.addWidget(slider, row, 1)
            self._metric_weight_sliders.append(slider)

    def _get_metric_weights(self) -> np.ndarray | None:
        if not self._metric_weight_sliders:
            return None
        vals = np.array([s.value() / 10.0 for s in self._metric_weight_sliders])
        return vals if not np.allclose(vals, 1.0) else None

    def _run_balancing(self) -> None:
        if self._df is None or self._loader is None:
            QMessageBox.warning(self, "No Data", "Load and confirm a CSV first.")
            return

        # Validate group sizes before starting
        group_sizes = self._get_group_sizes()
        if group_sizes is not None:
            if sum(group_sizes) != len(self._df):
                QMessageBox.warning(
                    self, "Size Mismatch",
                    f"Group sizes sum to {sum(group_sizes)} but the dataset has "
                    f"{len(self._df)} animals. Adjust sizes so they sum to {len(self._df)}."
                )
                return

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        algo_kwargs: dict = {
            "initial_temp": self.spin_temp.value(),
            "cooling_rate": self.spin_cool.value(),
            "random_seed": 42,
        }

        self._output_dir.mkdir(parents=True, exist_ok=True)
        group_names = [e.text() or f"Group {i}" for i, e in enumerate(self._group_name_edits)]
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_box.append(f"Run ID: {run_id}")

        # Clean up previous worker before creating a new one
        if self._worker is not None:
            self._worker.wait()
            self._worker = None

        self._worker = BalancingWorker(
            df=self._df,
            metric_cols=self._loader.metric_cols,
            k=self.spin_k.value(),
            algo_kwargs=algo_kwargs,
            alpha=self.slider_alpha.value() / 10.0,
            beta=self.slider_beta.value() / 10.0,
            gamma=self.slider_gamma.value() / 10.0,
            metric_weights=self._get_metric_weights(),
            continuous=self.chk_continuous.isChecked(),
            max_ci_iterations=self.spin_max_iter.value(),
            output_dir=self._output_dir,
            run_id=run_id,
            group_names=group_names,
            id_col=self._loader.id_col,
            group_sizes=group_sizes,
        )
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.result_ready.connect(lambda r: self._on_result(r, group_names))
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _stop_worker(self) -> None:
        if self._worker:
            self._worker.stop()
        self.btn_stop.setEnabled(False)

    def _on_progress(self, pct: int, msg: str) -> None:
        self.progress_bar.setValue(pct)
        self.log_box.append(msg)

    def _on_result(self, results: dict, group_names: list[str]) -> None:
        self._assignment = results["assignment"]
        score_d = results["score_details"]
        report = results["report"]
        scores = results["convergence_scores"]

        # Score label
        self.lbl_score.setText(f"Composite Score: {score_d['composite']:.4f}")

        # Pass/fail badge
        if report and report.overall_pass:
            self.lbl_pass.setText("PASS ✓")
            _set_label_bg(self.lbl_pass, "#1a9850")
        else:
            self.lbl_pass.setText("FAIL ✗")
            _set_label_bg(self.lbl_pass, "#d73027")

        # Convergence plot
        if len(scores) > 1:
            ax = self.convergence_canvas.figure.clear()
            ax = self.convergence_canvas.figure.add_subplot(111)
            ax.plot(scores, marker="o", markersize=3, linewidth=1.5, color="#1f77b4")
            ax.set_xlabel("Iteration"); ax.set_ylabel("Score")
            ax.set_title("Convergence")
            ax.grid(True, alpha=0.3)
            self.convergence_canvas.figure.tight_layout()
            self.convergence_canvas.draw()

        # Populate groups table
        df_with_group = self._df.copy()
        df_with_group["_group_name"] = [group_names[g] for g in self._assignment]
        self.groups_table.clear()
        cols = [self._loader.id_col] + self._loader.metric_cols + ["Group"]
        self.groups_table.setColumnCount(len(cols))
        self.groups_table.setHorizontalHeaderLabels(cols)
        self.groups_table.setRowCount(len(df_with_group))
        for r, (_, row) in enumerate(df_with_group.iterrows()):
            vals = (
                [str(row[self._loader.id_col])]
                + [str(round(row[mc], 4)) for mc in self._loader.metric_cols]
                + [row["_group_name"]]
            )
            for c, val in enumerate(vals):
                self.groups_table.setItem(r, c, QTableWidgetItem(val))

        # Stats report
        if report:
            validator = StatisticalValidator()
            self.stats_report_box.setText(validator.format_report(report))

        # Embed matplotlib figures — each run gets fresh canvases so Qt backend is clean
        viz = results.get("viz")
        if viz:
            self._dist_container.set_figure(viz.plot_distributions()[0])
            self._cov_container.set_figure(viz.plot_covariance()[0])
            self._pca_container.set_figure(viz.plot_pca()[0])

        run_id = results.get("run_id", "")
        run_dir = results.get("run_dir", "")
        self._run_dir = Path(run_dir) if run_dir else None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.tabs.setCurrentIndex(3)
        self.status_bar.showMessage(
            f"[{run_id}] Score: {score_d['composite']:.4f} | "
            f"{'PASS' if report and report.overall_pass else 'FAIL'} | "
            f"Saved: {run_dir}"
        )

    def _on_error(self, tb: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log_box.append(f"ERROR:\n{tb}")
        QMessageBox.critical(self, "Algorithm Error", tb[:500])


    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        if self._assignment is None or self._df is None:
            QMessageBox.warning(self, "No Results", "Run balancing first.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Assignments CSV",
            str(Path.home() / "assignments.csv"),
            "CSV files (*.csv)",
        )
        if not out_path:
            return
        out_df = self._df.copy()
        group_labels = [
            (self._group_name_edits[g].text() or f"Group {g}")
            for g in self._assignment
        ]
        id_pos = out_df.columns.get_loc(self._loader.id_col)
        out_df.insert(id_pos + 1, "group", group_labels)
        out_df.to_csv(out_path, index=False)
        QMessageBox.information(self, "Exported", f"Saved to {out_path}")

    def _export_pdf(self) -> None:
        if self._assignment is None or self._run_dir is None:
            QMessageBox.warning(self, "No Results", "Run balancing first.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", str(Path.home() / "report.pdf"), "PDF (*.pdf)"
        )
        if not out_path:
            return
        try:
            _generate_pdf_report(out_path, self._run_dir)
            QMessageBox.information(self, "Exported", f"PDF saved to {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "PDF Error", str(exc))

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background: #f5f5f5; }
            QTabBar::tab { min-width: 160px; padding: 6px 12px; }
            QPushButton { padding: 6px 14px; border-radius: 4px;
                          background: #2c7bb6; color: white; font-weight: bold; }
            QPushButton:hover { background: #1f5c8a; }
            QPushButton:disabled { background: #aaa; }
            QGroupBox { font-weight: bold; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
        """)


# ---------------------------------------------------------------------------
# Plot container — swaps in a fresh FigureCanvas each run
# ---------------------------------------------------------------------------

class _PlotContainer(QWidget):
    """
    A QWidget that holds exactly one FigureCanvas at a time.

    Calling set_figure() properly closes the old matplotlib figure and
    replaces the canvas so the Qt backend starts clean for each run.
    This prevents rendering artefacts when the same window runs multiple analyses.
    """

    def __init__(self) -> None:
        super().__init__()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._canvas: FigureCanvas | None = None

    def set_figure(self, fig: plt.Figure) -> None:
        """Replace the current canvas with one built from fig."""
        # Remove and destroy the old canvas
        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            plt.close(self._canvas.figure)
            self._canvas.deleteLater()

        self._canvas = FigureCanvas(fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._layout.addWidget(self._canvas)
        self._canvas.draw()


# ---------------------------------------------------------------------------
# PDF report helper
# ---------------------------------------------------------------------------

def _generate_pdf_report(out_path: str, figures_dir: Path) -> None:
    """
    Build a simple PDF report embedding all saved PNG figures.

    Uses reportlab for layout.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    doc = SimpleDocTemplate(out_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Balanced Study — Group Assignment Report", styles["Title"]), Spacer(1, 0.5 * cm)]

    for png in sorted(figures_dir.glob("*.png")):
        story.append(Paragraph(png.stem.replace("_", " ").title(), styles["Heading2"]))
        story.append(Image(str(png), width=16 * cm, height=10 * cm))
        story.append(Spacer(1, 0.5 * cm))

    doc.build(story)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_slider(min_val: int, max_val: int, default: int) -> QSlider:
    """Create a horizontal QSlider."""
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(min_val, max_val)
    s.setValue(default)
    return s


def _set_label_bg(lbl: QLabel, hex_color: str) -> None:
    """Set the background colour of a QLabel."""
    pal = lbl.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(hex_color))
    lbl.setPalette(pal)
    lbl.setStyleSheet(f"background-color: {hex_color}; color: white; font-weight: bold; border-radius: 4px; padding: 4px;")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Balanced Study")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
