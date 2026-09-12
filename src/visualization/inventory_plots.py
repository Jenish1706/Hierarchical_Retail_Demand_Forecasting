"""Inventory-layer visualisation: ABC Pareto, ABC/XYZ heatmap, service-level
trade-off curve and working-capital waterfall.

All plots follow the shared style from :mod:`src.visualization.style` so they
read as part of the same document as the forecasting leaderboard figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config.logging import get_logger  # noqa: E402

log = get_logger(__name__)

# Inventory-layer palette.  Consistent across all four figures so the reader
# builds a visual vocabulary: warm = high value, cool = low value.
SEGMENT_COLORS: dict[str, str] = {
    "A": "#c1444a",   # red -- high-revenue
    "B": "#e07b39",   # orange -- medium-revenue
    "C": "#2a7fb8",   # blue -- low-revenue
}

SL_COLORS: dict[float, str] = {
    0.90: "#9aa0a6",  # grey
    0.95: "#e07b39",  # orange
    0.98: "#c1444a",  # red
}


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
    """Save with tight layout and report the path."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("plot_saved", path=str(path))
    return path


# -----------------------------------------------------------------------
# 1. ABC Pareto curve
# -----------------------------------------------------------------------

def plot_abc_pareto(
    abc_xyz_df: pd.DataFrame,
    output_path: Path | str,
    title: str = "ABC Pareto Curve — Cumulative Revenue vs SKU Rank",
) -> Path:
    """Classic 80/20 Pareto chart showing revenue concentration.

    Parameters
    ----------
    abc_xyz_df : pd.DataFrame
        Output of ABCXYZClassifier.classify() -- must contain
        ``cumulative_revenue_pct`` and ``abc_class``.
    output_path : Path | str
        File path for the saved figure.
    """
    output_path = Path(output_path)
    df = abc_xyz_df.dropna(subset=["cumulative_revenue_pct"]).sort_values(
        "cumulative_revenue_pct"
    ).reset_index(drop=True)

    if len(df) == 0:
        log.warning("pareto_no_data")
        return output_path

    fig, ax = plt.subplots(figsize=(10, 6))

    # Rank index as percentage of SKU count
    n = len(df)
    sku_pct = np.arange(1, n + 1) / n * 100

    # Colour by ABC class
    for cls, color in SEGMENT_COLORS.items():
        mask = df["abc_class"] == cls
        ax.scatter(
            sku_pct[mask], df.loc[mask, "cumulative_revenue_pct"] * 100,
            c=color, s=4, alpha=0.6, label=f"Class {cls}", rasterized=True,
        )

    # Reference lines for ABC thresholds
    ax.axhline(80, color="#666", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(95, color="#666", ls="--", lw=0.8, alpha=0.6)
    ax.text(2, 81, "A threshold (80%)", fontsize=8, color="#666")
    ax.text(2, 96, "B threshold (95%)", fontsize=8, color="#666")

    ax.set_xlabel("SKU Rank (percentile)")
    ax.set_ylabel("Cumulative Revenue (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    sns.despine(ax=ax)

    return _save(fig, output_path)


# -----------------------------------------------------------------------
# 2. ABC/XYZ 3x3 heatmap
# -----------------------------------------------------------------------

def plot_abc_xyz_heatmap(
    abc_xyz_df: pd.DataFrame,
    output_path: Path | str,
    value_col: str = "count",
    title: str = "ABC/XYZ Classification Matrix — SKU Count",
) -> Path:
    """3x3 heatmap of SKU counts (or revenue) by ABC × XYZ class.

    Parameters
    ----------
    abc_xyz_df : pd.DataFrame
        Output of ABCXYZClassifier.classify().
    value_col : str
        ``"count"`` for SKU counts, or a numeric column name to sum.
    """
    output_path = Path(output_path)
    df = abc_xyz_df.dropna(subset=["abc_class", "xyz_class"])

    if len(df) == 0:
        log.warning("heatmap_no_data")
        return output_path

    if value_col == "count":
        pivot = pd.crosstab(df["abc_class"], df["xyz_class"])
    else:
        pivot = df.pivot_table(
            index="abc_class", columns="xyz_class", values=value_col,
            aggfunc="sum", fill_value=0,
        )

    # Ensure row/col order
    for label in ["A", "B", "C"]:
        if label not in pivot.index:
            pivot.loc[label] = 0
    for label in ["X", "Y", "Z"]:
        if label not in pivot.columns:
            pivot[label] = 0
    pivot = pivot.loc[["A", "B", "C"], ["X", "Y", "Z"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".0f" if value_col == "count" else ".1f",
        cmap="YlOrRd", linewidths=1, linecolor="white",
        cbar_kws={"label": "SKU Count" if value_col == "count" else value_col},
        ax=ax,
    )
    ax.set_xlabel("Demand Predictability (XYZ)")
    ax.set_ylabel("Revenue Importance (ABC)")
    ax.set_title(title, fontsize=12, fontweight="bold")

    return _save(fig, output_path)


# -----------------------------------------------------------------------
# 3. Service-level trade-off curve
# -----------------------------------------------------------------------

def plot_service_level_tradeoff(
    segment_summary: pd.DataFrame,
    output_path: Path | str,
    title: str = "Service-Level Trade-off — Holding Cost vs Stockout Cost",
) -> Path:
    """Stacked bar or line chart showing cost components at each service level.

    Parameters
    ----------
    segment_summary : pd.DataFrame
        Output of ServiceLevelAnalyzer.segment_summary(), aggregated by
        (abc_xyz_class, service_level).
    """
    output_path = Path(output_path)

    if len(segment_summary) == 0:
        log.warning("tradeoff_no_data")
        return output_path

    # Aggregate across all segments per service level
    sl_agg = (
        segment_summary.groupby("service_level")[
            ["holding_cost_annual_sum", "stockout_cost_annual_sum", "total_relevant_cost_sum"]
        ].sum().reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sl_agg))
    width = 0.35

    holding = sl_agg["holding_cost_annual_sum"].values
    stockout = sl_agg["stockout_cost_annual_sum"].values
    sls = sl_agg["service_level"].values

    ax.bar(x - width / 2, holding, width, label="Holding Cost", color="#e07b39")
    ax.bar(x + width / 2, stockout, width, label="Stockout Cost", color="#c1444a")
    ax.plot(x, holding + stockout, "ko-", label="Total Relevant Cost", ms=6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{sl:.0%}" for sl in sls])
    ax.set_xlabel("Service Level")
    ax.set_ylabel("Annual Cost ($)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    return _save(fig, output_path)


# -----------------------------------------------------------------------
# 4. Working-capital waterfall chart
# -----------------------------------------------------------------------

def plot_working_capital_waterfall(
    portfolio_summary: pd.DataFrame,
    output_path: Path | str,
    title: str = "Working Capital Release by ABC/XYZ Segment",
) -> Path:
    """Waterfall chart showing capital released per segment when switching
    from a blanket 95% policy to the recommended differentiated policy.

    Parameters
    ----------
    portfolio_summary : pd.DataFrame
        Output of WorkingCapitalAnalyzer.portfolio_summary().
    """
    output_path = Path(output_path)

    if len(portfolio_summary) == 0:
        log.warning("waterfall_no_data")
        return output_path

    df = portfolio_summary.sort_values("capital_released_total", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    segments = df["abc_xyz_class"].values
    releases = df["capital_released_total"].values

    colors = ["#4c9a52" if v >= 0 else "#c1444a" for v in releases]

    # Cumulative waterfall
    cumulative = np.cumsum(releases)
    starts = np.concatenate([[0], cumulative[:-1]])

    ax.bar(segments, releases, bottom=starts, color=colors, edgecolor="white", lw=0.5)

    # Total bar
    ax.bar(
        "TOTAL", cumulative[-1], color="#2a7fb8", edgecolor="white", lw=0.5,
    )

    # Value labels
    for i, (_seg, val) in enumerate(zip(segments, releases)):
        y_pos = starts[i] + val / 2
        ax.text(i, y_pos, f"${val:,.0f}", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(
        len(segments), cumulative[-1] / 2, f"${cumulative[-1]:,.0f}",
        ha="center", va="center", fontsize=9, fontweight="bold", color="white",
    )

    ax.set_ylabel("Capital Released ($)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    sns.despine(ax=ax)

    return _save(fig, output_path)
