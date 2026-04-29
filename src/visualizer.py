"""
visualizer.py — Publication-quality figures and interactive HTML plots.

Produces four figure types for every balancing solution:

Figure 1 — Per-metric distributions per group
    KDE + rug + stem plots; one subplot per (group × metric).

Figure 2 — Covariance / correlation structure
    Pearson correlation heatmap per group side-by-side.

Figure 3 — PCA scatter
    PC1 vs PC2 scatter with 95% confidence ellipses per group.

Figure 4 — Statistical summary table
    Corrected p-values per metric, colour-coded PASS/FAIL status column.

All figures saved as high-res PNG + interactive plotly HTML.
Assumption A09: reportlab used for PDF embedding.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend; replaced by Qt when GUI is running
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Colour palette for groups (up to 20)
_GROUP_PALETTE = sns.color_palette("tab20", 20)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Visualizer:
    """
    Generate and save all visualisations for a group-balancing solution.

    Parameters
    ----------
    df : pd.DataFrame
        Animal data (rows = animals, columns include metric_cols).
    metric_cols : list[str]
        Metric column names.
    assignment : np.ndarray of int, shape (n,)
        Group labels (0-indexed).
    k : int
        Number of groups.
    group_names : list[str] | None
        Display names for each group.  None → ['Group 0', 'Group 1', ...].
    dpi : int
        Resolution for PNG output.  Default 150.

    Examples
    --------
    >>> viz = Visualizer(df, ['w', 'g', 'act'], assignment, k=3)
    >>> viz.save_all(output_dir="outputs/run_01")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metric_cols: list[str],
        assignment: np.ndarray,
        k: int,
        group_names: list[str] | None = None,
        dpi: int = 150,
    ) -> None:
        self.df = df.copy()
        self.metric_cols = metric_cols
        self.assignment = assignment
        self.k = k
        self.group_names = group_names or [f"Group {g}" for g in range(k)]
        self.dpi = dpi
        self.colors = [_GROUP_PALETTE[g % 20] for g in range(k)]
        # Add assignment to working df
        self.df["_group"] = assignment
        self.df["_group_name"] = [self.group_names[g] for g in assignment]

    # ------------------------------------------------------------------
    # Figure 1 — Per-metric distribution plots
    # ------------------------------------------------------------------

    def plot_distributions(self) -> tuple[plt.Figure, go.Figure]:
        """
        Plot box plot + jittered scatter for each metric, all groups overlaid.

        One subplot per metric. Each group is a separate box with individual
        data points jittered alongside it.

        Returns
        -------
        tuple[matplotlib.figure.Figure, plotly.graph_objects.Figure]
            Static (PNG) and interactive (HTML) versions.
        """
        m = len(self.metric_cols)
        fig, axes = plt.subplots(1, m, figsize=(4 * m, 5), squeeze=False)
        fig.suptitle("Per-Metric Distributions by Group", fontsize=13)

        palette = {self.group_names[g]: self.colors[g] for g in range(self.k)}

        for col_idx, col in enumerate(self.metric_cols):
            ax = axes[0][col_idx]

            # Box plot — nearly transparent fill so scatter points remain the focus
            sns.boxplot(
                data=self.df,
                x="_group_name",
                y=col,
                order=self.group_names,
                palette=palette,
                width=0.5,
                fliersize=0,
                linewidth=1.2,
                boxprops=dict(alpha=0.15),
                medianprops=dict(linewidth=1.5),
                whiskerprops=dict(alpha=0.4),
                capprops=dict(alpha=0.4),
                ax=ax,
            )

            # Jittered scatter overlaid on each box
            sns.stripplot(
                data=self.df,
                x="_group_name",
                y=col,
                order=self.group_names,
                palette=palette,
                size=5,
                jitter=True,
                alpha=0.7,
                linewidth=0.4,
                edgecolor="white",
                ax=ax,
            )

            ax.set_title(col, fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel(col, fontsize=8)
            ax.tick_params(axis="x", labelsize=8, rotation=15)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()

        # --- Plotly interactive version (box + all points) ---
        plotly_fig = go.Figure()
        for g in range(self.k):
            grp_data = self.df[self.df["_group"] == g]
            rgb = f"rgb{tuple(int(c * 255) for c in self.colors[g][:3])}"
            for col in self.metric_cols:
                vals = grp_data[col].dropna().values
                plotly_fig.add_trace(
                    go.Box(
                        x=[col] * len(vals),
                        y=vals,
                        name=self.group_names[g],
                        legendgroup=self.group_names[g],
                        showlegend=(col == self.metric_cols[0]),
                        boxpoints="all",
                        jitter=0.4,
                        pointpos=0,
                        marker=dict(size=5, opacity=0.7, color=rgb),
                        line=dict(color=rgb),
                        fillcolor=rgb.replace("rgb", "rgba").replace(")", ",0.25)"),
                    )
                )
        plotly_fig.update_layout(
            title="Per-Metric Distributions by Group (Interactive)",
            boxmode="group",
        )

        return fig, plotly_fig

    # ------------------------------------------------------------------
    # Figure 2 — Covariance / correlation heatmaps
    # ------------------------------------------------------------------

    def plot_covariance(self) -> tuple[plt.Figure, go.Figure]:
        """
        Plot Pearson correlation heatmap for each group side by side.

        Returns
        -------
        tuple[matplotlib.figure.Figure, plotly.graph_objects.Figure]
        """
        fig, axes = plt.subplots(1, self.k, figsize=(4.5 * self.k, 4.5), squeeze=False)
        fig.suptitle("Pearson Correlation Structure by Group", fontsize=14)

        for g in range(self.k):
            ax = axes[0][g]
            grp_data = self.df[self.df["_group"] == g][self.metric_cols]
            if len(grp_data) >= 2:
                corr = grp_data.corr()
            else:
                corr = pd.DataFrame(np.eye(len(self.metric_cols)),
                                    index=self.metric_cols, columns=self.metric_cols)
            sns.heatmap(
                corr, ax=ax, annot=True, fmt=".2f",
                cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, linewidths=0.5, annot_kws={"size": 8},
                cbar=False,
            )
            ax.set_title(self.group_names[g], fontsize=11)
            ax.tick_params(labelsize=8)

        # Single shared horizontal colorbar centred at the bottom
        fig.subplots_adjust(bottom=0.18)
        cbar_ax = fig.add_axes([0.3, 0.06, 0.4, 0.035])
        import matplotlib as _mpl
        sm = _mpl.cm.ScalarMappable(cmap="RdBu_r", norm=_mpl.colors.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="Pearson r")

        fig.tight_layout(rect=[0, 0.12, 1, 1])

        # Plotly version: subplots
        from plotly.subplots import make_subplots
        pfig = make_subplots(rows=1, cols=self.k,
                             subplot_titles=self.group_names)
        for g in range(self.k):
            grp_data = self.df[self.df["_group"] == g][self.metric_cols]
            corr = grp_data.corr() if len(grp_data) >= 2 else pd.DataFrame(
                np.eye(len(self.metric_cols)), index=self.metric_cols, columns=self.metric_cols
            )
            pfig.add_trace(
                go.Heatmap(
                    z=corr.values,
                    x=self.metric_cols,
                    y=self.metric_cols,
                    colorscale="RdBu",
                    zmin=-1, zmax=1,
                    showscale=(g == self.k - 1),
                    text=corr.round(2).values,
                    texttemplate="%{text}",
                    name=self.group_names[g],
                ),
                row=1, col=g + 1,
            )
        pfig.update_layout(title_text="Correlation Heatmaps by Group")

        return fig, pfig

    # ------------------------------------------------------------------
    # Figure 3 — PCA scatter with confidence ellipses
    # ------------------------------------------------------------------

    def plot_pca(self) -> tuple[plt.Figure, go.Figure]:
        """
        Project animals into PC1–PC2 space; colour by group; draw 95% ellipses.

        Returns
        -------
        tuple[matplotlib.figure.Figure, plotly.graph_objects.Figure]
        """
        X = self.df[self.metric_cols].values.astype(float)
        scaler = StandardScaler()
        X_z = scaler.fit_transform(X)

        n_comp = min(2, X_z.shape[1], X_z.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        X_pc = pca.fit_transform(X_z)

        var_exp = pca.explained_variance_ratio_ * 100

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title("PCA Scatter — Animals Coloured by Group", fontsize=13)

        for g in range(self.k):
            idx = self.assignment == g
            pts = X_pc[idx]
            color = self.colors[g]
            ax.scatter(pts[:, 0], pts[:, 1] if n_comp > 1 else np.zeros(len(pts)),
                       color=color, label=self.group_names[g],
                       alpha=0.8, edgecolors="white", linewidths=0.5, s=60)
            if len(pts) >= 3 and n_comp > 1:
                _draw_confidence_ellipse(pts[:, 0], pts[:, 1], ax, color=color)

        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)", fontsize=11)
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)" if n_comp > 1 else "PC2 (0% var)", fontsize=11)
        ax.legend(title="Group", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Plotly version
        pca_df = pd.DataFrame({
            "PC1": X_pc[:, 0],
            "PC2": X_pc[:, 1] if n_comp > 1 else np.zeros(len(X_pc)),
            "Group": self.df["_group_name"].values,
            "Animal": self.df.index.astype(str),
        })
        pfig = px.scatter(
            pca_df, x="PC1", y="PC2", color="Group",
            hover_data=["Animal"],
            title="PCA Scatter (Interactive)",
            labels={"PC1": f"PC1 ({var_exp[0]:.1f}% var)",
                    "PC2": f"PC2 ({var_exp[1]:.1f}% var)" if n_comp > 1 else "PC2"},
        )
        pfig.update_traces(marker=dict(size=10, opacity=0.8))

        return fig, pfig

    # ------------------------------------------------------------------
    # Figure 4 — Statistical p-value table
    # ------------------------------------------------------------------

    def plot_stats_table(
        self,
        metric_results: list,
        manova_result: dict | None = None,
        boxm_result: dict | None = None,
    ) -> tuple[plt.Figure, go.Figure]:
        """
        Readable table of corrected p-values: one row per metric, plus
        MANOVA and Box's M rows at the bottom.

        Columns: Metric | p-value | Test | Normality | Status
        Status cell: green = PASS (p >= 0.05), red = FAIL (p < 0.05).
        Box's M status shown in grey — supplementary, not part of overall verdict.

        Parameters
        ----------
        metric_results : list[MetricResult]
            From StatisticalValidator.validate().
        manova_result : dict | None
            From ValidationReport.manova_result or permutation_result.
        boxm_result : dict | None
            From ValidationReport.boxm_result.
        """
        ALPHA = 0.05
        col_headers = ["Metric / Test", "p-value", "Method", "Notes", "Status"]
        col_widths   = [0.28, 0.15, 0.22, 0.19, 0.16]

        rows, status_flags, is_supplementary = [], [], []

        # Per-metric rows
        for mr in metric_results:
            norm_str = "normal" if getattr(mr, "all_normal", False) else "non-normal"
            is_fail  = mr.significant
            rows.append([
                mr.metric,
                f"{mr.corrected_p_value:.4f}",
                mr.test_used,
                f"Bonferroni; {norm_str}",
                "FAIL" if is_fail else "PASS",
            ])
            status_flags.append(is_fail)
            is_supplementary.append(False)

        # MANOVA / permutation row
        mv = manova_result
        if mv and "error" not in mv:
            p_mv = mv.get("p_value", float("nan"))
            method_mv = mv.get("method", "MANOVA")
            is_fail_mv = isinstance(p_mv, float) and not np.isnan(p_mv) and p_mv < ALPHA
            rows.append([
                "MANOVA",
                f"{p_mv:.4f}" if isinstance(p_mv, float) and not np.isnan(p_mv) else "N/A",
                method_mv,
                "Multivariate (all metrics)",
                "FAIL" if is_fail_mv else "PASS",
            ])
            status_flags.append(is_fail_mv)
            is_supplementary.append(False)

        # Box's M row
        bm = boxm_result
        if bm and "error" not in bm and "skipped" not in bm.get("note", ""):
            p_bm = bm.get("p_value", float("nan"))
            is_fail_bm = isinstance(p_bm, float) and not np.isnan(p_bm) and p_bm < ALPHA
            rows.append([
                "Box's M",
                f"{p_bm:.4f}" if isinstance(p_bm, float) and not np.isnan(p_bm) else "N/A",
                "chi2 approx.",
                "Covariance homogeneity*",
                "FAIL*" if is_fail_bm else "PASS*",
            ])
            status_flags.append(is_fail_bm)
            is_supplementary.append(True)

        n_rows = len(rows)
        fig_h  = max(3.2, 0.55 * n_rows + 1.8)
        fig, ax = plt.subplots(figsize=(11, fig_h))
        ax.axis("off")
        ax.set_title(
            "Statistical Validation Summary",
            fontsize=13, fontweight="bold", pad=12,
        )

        tbl = ax.table(
            cellText=rows,
            colLabels=col_headers,
            colWidths=col_widths,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 2.0)

        # Header row
        for col_idx in range(len(col_headers)):
            cell = tbl[0, col_idx]
            cell.set_facecolor("#2c7bb6")
            cell.set_text_props(color="white", fontweight="bold", fontsize=11)

        # Data rows
        for row_idx, (is_fail, is_supp) in enumerate(zip(status_flags, is_supplementary), start=1):
            for col_idx in range(len(col_headers)):
                cell = tbl[row_idx, col_idx]
                if col_idx == len(col_headers) - 1:  # Status column
                    if is_supp:
                        cell.set_facecolor("#888888")  # grey = supplementary
                    else:
                        cell.set_facecolor("#d73027" if is_fail else "#1a9850")
                    cell.set_text_props(color="white", fontweight="bold")
                else:
                    if is_supp:
                        cell.set_facecolor("#f5f5f5")
                        cell.set_text_props(color="#666")
                    else:
                        cell.set_facecolor("#fff5f5" if is_fail else "#f5fff5")
                        cell.set_text_props(color="#222")
                cell.set_edgecolor("#cccccc")

        if any(is_supplementary):
            fig.text(0.5, 0.01, "* Box's M is supplementary — sensitive to non-normality, not part of PASS/FAIL verdict.",
                     ha="center", fontsize=9, color="#555", style="italic")

        fig.tight_layout(rect=[0, 0.04, 1, 1])

        # Plotly interactive version
        header_color = "#2c7bb6"
        pass_color   = "#1a9850"
        fail_color   = "#d73027"
        supp_color   = "#888888"

        status_colors = [
            supp_color if s else (fail_color if f else pass_color)
            for f, s in zip(status_flags, is_supplementary)
        ]
        row_bg = [
            "#f5f5f5" if s else ("#fff5f5" if f else "#f5fff5")
            for f, s in zip(status_flags, is_supplementary)
        ]

        col_data = list(zip(*rows)) if rows else [[] for _ in col_headers]
        pfig = go.Figure(go.Table(
            columnwidth=[200, 110, 170, 160, 110],
            header=dict(
                values=[f"<b>{h}</b>" for h in col_headers],
                fill_color=header_color,
                font=dict(color="white", size=13),
                align="center",
                height=36,
            ),
            cells=dict(
                values=list(col_data),
                fill_color=[row_bg, row_bg, row_bg, row_bg, status_colors],
                font=dict(color=["#222"] * 4 + ["white"], size=12),
                align="center",
                height=32,
            ),
        ))
        pfig.update_layout(
            title="Statistical Validation Summary",
            margin=dict(l=20, r=20, t=60, b=20),
        )

        return fig, pfig

    # ------------------------------------------------------------------
    # Save all figures
    # ------------------------------------------------------------------

    def save_all(
        self,
        output_dir: str | Path,
        metric_results: list | None = None,
        manova_result: dict | None = None,
        boxm_result: dict | None = None,
        prefix: str = "",
    ) -> dict[str, Path]:
        """
        Generate and save all four figures as PNG + HTML.

        Parameters
        ----------
        output_dir : str | Path
            Directory to save files into.
        metric_results : list[MetricResult] | None
            Required for Figure 4.  If None, Figure 4 is skipped.
        manova_result : dict | None
            From ValidationReport.manova_result or permutation_result.
        boxm_result : dict | None
            From ValidationReport.boxm_result.
        prefix : str
            Optional string prepended to all filenames.

        Returns
        -------
        dict[str, Path]
            Mapping of figure name to saved PNG path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = {}
        p = (prefix + "_") if prefix else ""

        def _save(fig_mpl, fig_plotly, name: str) -> Path:
            png_path = out / f"{p}{name}.png"
            html_path = out / f"{p}{name}.html"
            fig_mpl.savefig(str(png_path), dpi=self.dpi, bbox_inches="tight")
            plt.close(fig_mpl)
            fig_plotly.write_html(str(html_path))
            return png_path

        saved["distributions"] = _save(*self.plot_distributions(), "fig1_distributions")
        saved["covariance"]    = _save(*self.plot_covariance(),    "fig2_covariance")
        saved["pca"]           = _save(*self.plot_pca(),           "fig3_pca")

        if metric_results is not None:
            saved["stats"] = _save(
                *self.plot_stats_table(metric_results, manova_result, boxm_result),
                "fig4_stats",
            )

        return saved


# ---------------------------------------------------------------------------
# Internal helper — 95% confidence ellipse
# ---------------------------------------------------------------------------

def _draw_confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    n_std: float = 1.96,
    color: str = "blue",
) -> None:
    """
    Draw a 95% confidence ellipse (1.96 std) around a 2D point cloud.

    Based on the eigendecomposition of the covariance matrix.
    """
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=2.0,
    )
    ax.add_patch(ellipse)
