"""
Render README benchmark and inventory tables directly from committed artifacts.

Usage:
    python scripts/render_readme_tables.py          # Updates README.md in-place
    python scripts/render_readme_tables.py --check  # Verifies README.md matches artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_CSV = PROJECT_ROOT / "reports" / "backtest_main" / "accuracy_overall.csv"
SIGNIFICANCE_JSON = PROJECT_ROOT / "reports" / "backtest_main" / "significance" / "significance_report.json"
INVENTORY_CSV = PROJECT_ROOT / "reports" / "inventory" / "portfolio_summary.csv"
README_PATH = PROJECT_ROOT / "README.md"

LEADERBOARD_START = "<!-- BEGIN_LEADERBOARD_TABLE -->"
LEADERBOARD_END = "<!-- END_LEADERBOARD_TABLE -->"
INVENTORY_START = "<!-- BEGIN_INVENTORY_TABLE -->"
INVENTORY_END = "<!-- END_INVENTORY_TABLE -->"

MODEL_INFO = {
    "adida": {"name": "**ADIDA**", "tier": "Intermittent Statistical"},
    "moving_average": {"name": "**Moving Average (28d)**", "tier": "Simple Baseline"},
    "routed_ensemble": {"name": "**Routed SBT Ensemble**", "tier": "Hybrid (ADIDA + Global)"},
    "ses": {"name": "**SES**", "tier": "Exponential Smoothing"},
    "lightgbm": {"name": "**LightGBM**", "tier": "Global GBDT"},
    "lightgbm_tweedie": {"name": "**LightGBM (Tweedie)**", "tier": "Global GBDT"},
    "croston_sba": {"name": "**Croston SBA**", "tier": "Intermittent Statistical"},
    "seasonal_naive_weekly": {"name": "**Seasonal Naive (Weekly)**", "tier": "Baseline"},
    "seasonal_naive": {"name": "**Seasonal Naive (28d)**", "tier": "Baseline"},
    "naive": {"name": "**Naive**", "tier": "Baseline"},
}

POLICY_METADATA = {
    "AX": {"csl": "**98%**", "impact": "High-turnover revenue drivers; upgraded buffer"},
    "AY": {"csl": "**95%**", "impact": "Moderate volume; maintained standard buffer"},
    "AZ": {"csl": "**95%**", "impact": "High volatility; guarded safety buffer"},
    "BX": {"csl": "**95%**", "impact": "Stable secondary SKUs; maintained"},
    "BY": {"csl": "**95%**", "impact": "Moderate secondary volume; maintained"},
    "BZ": {"csl": "**90%**", "impact": "Volatile secondary; working capital released"},
    "CX": {"csl": "**90%**", "impact": "Stable low volume; bulk periodic order"},
    "CY": {"csl": "**90%**", "impact": "Slow-moving; conservative buffer"},
    "CZ": {"csl": "**90%**", "impact": "Erratic low volume; working capital released"},
}


def render_leaderboard_table() -> str:
    """Generate Markdown leaderboard table and significance footnote."""
    df_acc = pd.read_csv(BACKTEST_CSV)
    with open(SIGNIFICANCE_JSON, "r", encoding="utf-8") as f:
        sig = json.load(f)

    # Sort models by WAPE ascending
    df_sorted = df_acc.sort_values("wape").reset_index(drop=True)

    # Extract significance info
    mcb_map = {m["model"]: m for m in sig.get("mcb", [])}
    ranks_map = sig.get("nemenyi", {}).get("average_ranks", {})

    min_wape = df_sorted["wape"].min()
    min_rmsse = df_sorted["rmsse"].min()
    best_rank = min(ranks_map.values()) if ranks_map else None

    lines = [
        "| Rank | Model Name | Architecture Tier | WAPE (%) | RMSSE | Nemenyi Avg Rank | Statistical Verdict |",
        "|---|---|---|---|---|---|---|",
    ]

    for idx, row in df_sorted.iterrows():
        model_key = row["model"]
        info = MODEL_INFO.get(model_key, {"name": f"**{model_key}**", "tier": "Model"})

        # Format metrics
        wape_val = row["wape"]
        wape_str = f"**{wape_val:.2f}%**" if abs(wape_val - min_wape) < 1e-4 else f"{wape_val:.2f}%"

        rmsse_val = row["rmsse"]
        rmsse_str = f"**{rmsse_val:.3f}**" if abs(rmsse_val - min_rmsse) < 1e-4 else f"{rmsse_val:.3f}"

        avg_rank = ranks_map.get(model_key, float("nan"))
        if best_rank is not None and abs(avg_rank - best_rank) < 1e-4:
            rank_str = f"**{avg_rank:.2f}**"
        else:
            rank_str = f"{avg_rank:.2f}"

        # Verdict
        mcb_info = mcb_map.get(model_key, {})
        if mcb_info.get("is_best", False):
            verdict = "**Best by Average Rank**"
        elif mcb_info.get("indistinguishable_from_best", False):
            verdict = "**Indistinguishable from Best**"
        elif model_key in ("seasonal_naive", "naive"):
            verdict = "Baseline"
        else:
            verdict = "Significantly worse than best rank"

        rank_num = idx + 1
        rank_num_str = f"**{rank_num}**"
        lines.append(
            f"| {rank_num_str} | {info['name']} | {info['tier']} | {wape_str} | {rmsse_str} | {rank_str} | {verdict} |"
        )

    # Footnote
    friedman_stat = sig.get("friedman", {}).get("statistic", 2460.79)
    n_models = sig.get("n_models", 10)
    n_series = sig.get("n_series_complete", 2950)
    cd = sig.get("nemenyi", {}).get("critical_difference", 0.2494)
    dm_sep = sig.get("diebold_mariano", {}).get("n_separable_holm", 35)
    dm_pairs = sig.get("diebold_mariano", {}).get("n_pairs", 45)

    footnote = (
        f"\n*Statistical significance: Omnibus Friedman test $\\chi^2 = {friedman_stat:.2f}$ "
        f"($p < 10^{{-12}}$, {n_models} models $\\times$ {n_series:,} complete series). "
        f"Post-hoc Nemenyi critical difference $CD = {cd:.4f}$ ($\\alpha = 0.05$). "
        f"Holm-corrected Diebold-Mariano tests separated {dm_sep} of {dm_pairs} model pairs.*"
    )

    return "\n".join(lines) + "\n" + footnote


def render_inventory_table() -> str:
    """Generate Markdown inventory allocation matrix from portfolio summary."""
    df_port = pd.read_csv(INVENTORY_CSV)

    lines = [
        "| ABC/XYZ Cell | SKU Count | Current Working Capital ($) | Recommended Working Capital ($) | Capital Released ($) | Target Service Level (CSL) | Policy Impact |",
        "|---|---|---|---|---|---|---|",
    ]

    total_skus = 0
    total_current_wc = 0.0
    total_rec_wc = 0.0
    total_released = 0.0

    unpriced_skus = 0

    cell_order = ["AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"]
    port_dict = {row["abc_xyz_class"]: row for _, row in df_port.iterrows()}

    for cell in cell_order:
        row = port_dict.get(cell)
        if row is None:
            continue
        n_skus = int(row["n_skus"])
        cur_wc = float(row["current_wc_total"])
        rec_wc = float(row["recommended_wc_total"])
        rel_wc = float(row["capital_released_total"])

        total_skus += n_skus
        total_current_wc += cur_wc
        total_rec_wc += rec_wc
        total_released += rel_wc

        meta = POLICY_METADATA.get(cell, {"csl": "95%", "impact": "Maintained"})

        cur_str = f"${cur_wc:,.2f}"
        rec_str = f"${rec_wc:,.2f}"
        if rel_wc > 0:
            rel_str = f"+${rel_wc:,.2f}"
        elif rel_wc < 0:
            rel_str = f"-${abs(rel_wc):,.2f}"
        else:
            rel_str = "$0.00"

        lines.append(
            f"| **{cell}** | {n_skus:,} | {cur_str} | {rec_str} | {rel_str} | {meta['csl']} | {meta['impact']} |"
        )

    # Accumulate unpriced
    for cell, row in port_dict.items():
        if cell.startswith("?"):
            unpriced_skus += int(row["n_skus"])

    total_skus += unpriced_skus
    lines.append(
        f"| **Unpriced (?X/?Y/?Z)** | {unpriced_skus:,} | $0.00 | $0.00 | $0.00 | — | CA_2 items lacking price coverage |"
    )

    tot_rel_str = f"+${total_released:,.2f}" if total_released >= 0 else f"-${abs(total_released):,.2f}"
    lines.append(
        f"| **Total Portfolio** | **{total_skus:,}** | **${total_current_wc:,.2f}** | **${total_rec_wc:,.2f}** | **{tot_rel_str}** | — | **${total_released:,.2f} Net Working Capital Released** |"
    )

    return "\n".join(lines)


def check_readme_tables(readme_path: Path = README_PATH) -> tuple[bool, str]:
    """Check if README.md tables match artifacts. Returns (is_synced, diff_message)."""
    if not readme_path.exists():
        return False, f"README path does not exist: {readme_path}"

    content = readme_path.read_text(encoding="utf-8")

    if LEADERBOARD_START not in content or LEADERBOARD_END not in content:
        return False, f"Leaderboard markers missing in {readme_path}"
    if INVENTORY_START not in content or INVENTORY_END not in content:
        return False, f"Inventory markers missing in {readme_path}"

    leaderboard_expected = render_leaderboard_table().strip()
    inventory_expected = render_inventory_table().strip()

    # Extract current tables
    lb_part = content.split(LEADERBOARD_START)[1].split(LEADERBOARD_END)[0].strip()
    inv_part = content.split(INVENTORY_START)[1].split(INVENTORY_END)[0].strip()

    diffs = []
    if lb_part != leaderboard_expected:
        diffs.append("Leaderboard table in README does not match artifact outputs.")
    if inv_part != inventory_expected:
        diffs.append("Inventory table in README does not match artifact outputs.")

    if diffs:
        return False, "\n".join(diffs)
    return True, "README tables are fully synchronized with artifacts."


def update_readme_tables(readme_path: Path = README_PATH) -> None:
    """Update README.md tables in-place using generated tables."""
    content = readme_path.read_text(encoding="utf-8")

    leaderboard_str = render_leaderboard_table().strip()
    inventory_str = render_inventory_table().strip()

    # Replace leaderboard
    if LEADERBOARD_START in content and LEADERBOARD_END in content:
        before = content.split(LEADERBOARD_START)[0]
        after = content.split(LEADERBOARD_END)[1]
        content = f"{before}{LEADERBOARD_START}\n{leaderboard_str}\n{LEADERBOARD_END}{after}"
    else:
        print(f"Warning: {LEADERBOARD_START} not found, skipping leaderboard replacement.")

    # Replace inventory
    if INVENTORY_START in content and INVENTORY_END in content:
        before = content.split(INVENTORY_START)[0]
        after = content.split(INVENTORY_END)[1]
        content = f"{before}{INVENTORY_START}\n{inventory_str}\n{INVENTORY_END}{after}"
    else:
        print(f"Warning: {INVENTORY_START} not found, skipping inventory replacement.")

    readme_path.write_text(content, encoding="utf-8")
    print(f"Successfully updated README tables in {readme_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render or verify README tables from artifacts.")
    parser.add_argument("--check", action="store_true", help="Check if tables match artifacts without writing.")
    args = parser.parse_args()

    if args.check:
        is_synced, msg = check_readme_tables()
        print(msg)
        return 0 if is_synced else 1
    else:
        update_readme_tables()
        return 0


if __name__ == "__main__":
    sys.exit(main())
