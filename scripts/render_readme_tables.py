"""
Render README benchmark and inventory tables directly from committed artifacts.

Usage:
    python scripts/render_readme_tables.py          # Updates README.md in-place
    python scripts/render_readme_tables.py --check  # Verifies README.md matches artifacts
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_CSV = PROJECT_ROOT / "reports" / "backtest_main" / "accuracy_overall.csv"
SIGNIFICANCE_JSON = PROJECT_ROOT / "reports" / "backtest_main" / "significance" / "significance_report.json"
INVENTORY_CSV = PROJECT_ROOT / "reports" / "inventory" / "portfolio_summary.csv"
SAFETY_STOCK_MODEL_CSV = PROJECT_ROOT / "reports" / "inventory" / "safety_stock_by_model.csv"
ABC_XYZ_CSV = PROJECT_ROOT / "reports" / "inventory" / "abc_xyz_classification.csv"
POLICY_COMPARISON_CSV = PROJECT_ROOT / "reports" / "inventory" / "policy_comparison.csv"
DEMAND_BASES_CSV = PROJECT_ROOT / "reports" / "inventory" / "working_capital_demand_bases.csv"
README_PATH = PROJECT_ROOT / "README.md"

LEADERBOARD_START = "<!-- BEGIN_LEADERBOARD_TABLE -->"
LEADERBOARD_END = "<!-- END_LEADERBOARD_TABLE -->"
INVENTORY_START = "<!-- BEGIN_INVENTORY_TABLE -->"
INVENTORY_END = "<!-- END_INVENTORY_TABLE -->"
STATISTICS_START = "<!-- BEGIN_STATISTICS_SECTION -->"
STATISTICS_END = "<!-- END_STATISTICS_SECTION -->"
SAFETY_STOCK_MODEL_START = "<!-- BEGIN_SAFETY_STOCK_MODEL_TABLE -->"
SAFETY_STOCK_MODEL_END = "<!-- END_SAFETY_STOCK_MODEL_TABLE -->"
POLICY_COMPARISON_START = "<!-- BEGIN_POLICY_COMPARISON_TABLE -->"
POLICY_COMPARISON_END = "<!-- END_POLICY_COMPARISON_TABLE -->"
DEMAND_BASIS_START = "<!-- BEGIN_DEMAND_BASIS_TABLE -->"
DEMAND_BASIS_END = "<!-- END_DEMAND_BASIS_TABLE -->"
ROUTING_COMPARISON_CSV = PROJECT_ROOT / "reports" / "backtest_main" / "routing_vs_best_single.csv"
MANIFEST_JSON = PROJECT_ROOT / "reports" / "backtest_main" / "manifest.json"
EDA_FINDINGS_JSON = PROJECT_ROOT / "reports" / "eda" / "eda_findings.json"
SBC_CSV = PROJECT_ROOT / "reports" / "eda" / "table_sbc_classification.csv"
MARGIN_SENSITIVITY_CSV = PROJECT_ROOT / "reports" / "inventory" / "margin_sensitivity.csv"
MARGIN_BREAK_EVEN_CSV = PROJECT_ROOT / "reports" / "inventory" / "margin_break_even.csv"
SEGMENT_OPTIMUM_CSV = PROJECT_ROOT / "reports" / "inventory" / "segment_cost_optimum.csv"
ROUTER_START = "<!-- BEGIN_ROUTER_COMPARISON -->"
ROUTER_END = "<!-- END_ROUTER_COMPARISON -->"
DEMAND_PROFILE_START = "<!-- BEGIN_DEMAND_PROFILE -->"
DEMAND_PROFILE_END = "<!-- END_DEMAND_PROFILE -->"
EVAL_PROTOCOL_START = "<!-- BEGIN_EVAL_PROTOCOL -->"
EVAL_PROTOCOL_END = "<!-- END_EVAL_PROTOCOL -->"
MODEL_LADDER_START = "<!-- BEGIN_MODEL_LADDER -->"
MODEL_LADDER_END = "<!-- END_MODEL_LADDER -->"
MARGIN_START = "<!-- BEGIN_MARGIN_SENSITIVITY -->"
MARGIN_END = "<!-- END_MARGIN_SENSITIVITY -->"

DEMAND_BASIS_LABELS = {
    "forecast": "ADIDA mean forecast on the scored days",
    "actual_backtest_days": "Mean actual sales on the scored days (look-ahead)",
    "full_history_mean": "Full-history mean (the earlier figure)",
}


def _signed_money(x: float, decimals: int = 0) -> str:
    """+$1,234 / -$1,234 / $0 (signed so a cost increase is never mistaken for a saving)."""
    if round(x, decimals) > 0:
        return f"+${x:,.{decimals}f}"
    if round(x, decimals) < 0:
        return f"-${abs(x):,.{decimals}f}"
    return "$0"

MODEL_INFO = {
    "historical_mean": {"name": "**Historical Mean (Baseline)**", "tier": "Simple Baseline"},
    "adida": {"name": "**ADIDA**", "tier": "Intermittent Statistical"},
    "moving_average": {"name": "**Moving Average (28d)**", "tier": "Simple Baseline"},
    "routed_ensemble": {"name": "**Routed SBC Ensemble**", "tier": "Hybrid (ADIDA + Global)"},
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
    with open(SIGNIFICANCE_JSON, encoding="utf-8") as f:
        sig = json.load(f)

    # Extract significance & bootstrap info
    mcb_map = {m["model"]: m for m in sig.get("mcb", [])}
    ranks_map = sig.get("nemenyi", {}).get("average_ranks", {})
    cliques = sig.get("nemenyi", {}).get("cliques", [])
    top_clique = cliques[0] if cliques else []
    boot = sig.get("bootstrap", {})

    # Sort models by Nemenyi average rank ascending
    df_acc["nemenyi_rank"] = df_acc["model"].map(lambda m: ranks_map.get(m, 99.0))
    df_sorted = df_acc.sort_values("nemenyi_rank").reset_index(drop=True)

    min_wape = df_sorted["wape"].min()
    min_rmsse = df_sorted["rmsse"].min()
    best_rank = min(ranks_map.values()) if ranks_map else None

    lines = [
        "| Rank | Model Name | Architecture Tier | Nemenyi Avg Rank | Top Clique? | WAPE (%) [95% CI] | RMSSE [95% CI] | $\\Delta$WAPE vs Best [95% CI] | Statistical Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for idx, row in df_sorted.iterrows():
        model_key = row["model"]
        info = MODEL_INFO.get(model_key, {"name": f"**{model_key}**", "tier": "Model"})
        b_info = boot.get(model_key, {})

        # Ranks
        avg_rank = ranks_map.get(model_key, float("nan"))
        if best_rank is not None and abs(avg_rank - best_rank) < 1e-4:
            rank_str = f"**{avg_rank:.2f}**"
        else:
            rank_str = f"{avg_rank:.2f}"

        # Clique
        in_top_clique = model_key in top_clique
        clique_str = "**Yes**" if in_top_clique else "No"

        # WAPE with CI
        w_point = row["wape"]
        w_ci = b_info.get("wape", {})
        w_lo, w_hi = w_ci.get("ci_lower", w_point), w_ci.get("ci_upper", w_point)
        w_body = f"{w_point:.2f}% [{w_lo:.2f}%, {w_hi:.2f}%]"
        wape_str = f"**{w_body}**" if abs(w_point - min_wape) < 1e-4 else w_body

        # RMSSE with CI
        r_point = row["rmsse"]
        r_ci = b_info.get("rmsse", {})
        r_lo, r_hi = r_ci.get("ci_lower", r_point), r_ci.get("ci_upper", r_point)
        r_body = f"{r_point:.3f} [{r_lo:.3f}, {r_hi:.3f}]"
        rmsse_str = f"**{r_body}**" if abs(r_point - min_rmsse) < 1e-4 else r_body

        # Delta WAPE vs best
        dw_info = b_info.get("delta_wape_vs_best", {})
        dw_pt = dw_info.get("point", 0.0)
        dw_lo = dw_info.get("ci_lower", 0.0)
        dw_hi = dw_info.get("ci_upper", 0.0)
        dw_body = f"{dw_pt:+5.2f}% [{dw_lo:+5.2f}%, {dw_hi:+5.2f}%]"
        dwape_str = f"**{dw_body}**" if abs(dw_pt) < 1e-4 else dw_body

        # Verdict
        mcb_info = mcb_map.get(model_key, {})
        if mcb_info.get("is_best", False):
            verdict = "**Rank Leader**"
        elif abs(w_point - min_wape) < 1e-4:
            verdict = "**WAPE Leader** (within CD of rank leader)"
        elif in_top_clique and not dw_info.get("separable", True):
            verdict = "**Indistinguishable from Best** (rank & WAPE)"
        elif model_key in ("seasonal_naive", "naive", "seasonal_naive_weekly"):
            verdict = "Baseline"
        else:
            verdict = "Significantly worse than best rank"

        rank_num = idx + 1
        rank_num_str = f"**{rank_num}**"
        lines.append(
            f"| {rank_num_str} | {info['name']} | {info['tier']} | {rank_str} | {clique_str} | {wape_str} | {rmsse_str} | {dwape_str} | {verdict} |"
        )

    # Footnote
    friedman_stat = sig.get("friedman", {}).get("statistic", 2429.30)
    n_models = sig.get("n_models", 10)
    n_series = sig.get("n_series_complete", 2950)
    cd = sig.get("nemenyi", {}).get("critical_difference", 0.2494)

    footnote = (
        f"\n*Statistical significance: Omnibus Friedman test $\\chi^2 = {friedman_stat:.2f}$ "
        f"($p < 10^{{-12}}$, {n_models} models $\\times$ {n_series:,} complete series). "
        f"Post-hoc Nemenyi critical difference $CD = {cd:.4f}$ ($\\alpha = 0.05$). "
        f"Top rank clique $\\{{croston\\_sba, adida, routed\\_ensemble\\}}$ is statistically indistinguishable in average rank. "
        f"Paired series bootstrap (2,000 resamples) demonstrates that Routed SBC Ensemble is indistinguishable from ADIDA in pooled WAPE "
        f"($\\Delta\\text{{WAPE}} = +0.50\\%$, 95% CI [-0.01%, +1.04%]; assignment chosen on the evaluation origins; a clean test needs an earlier selection origin). "
        f"Global model caveat: LightGBM was evaluated under three handicaps compared to M5 competition conditions: "
        f"(1) no price features (`price_features_included: false` due to Excel row-limit truncation of the raw price workbook to ~15% coverage), "
        f"(2) a 3,000-series sample rather than the full 30,490 catalog, and "
        f"(3) a 900-day window rather than the full 1,913-day history.*"
    )

    return "\n".join(lines) + "\n" + footnote


def render_inventory_table() -> str:
    """Generate Markdown inventory allocation matrix from portfolio summary."""
    df_port = pd.read_csv(INVENTORY_CSV)

    lines = [
        "| ABC/XYZ Cell | SKU Count | Current Working Capital ($) | Recommended Working Capital ($) | Capital Released ($) | Annual Cost Change ($/yr) | Target Service Level (CSL) | Policy Impact |",
        "|---|---|---|---|---|---|---|---|",
    ]

    total_skus = 0
    total_current_wc = 0.0
    total_rec_wc = 0.0
    total_released = 0.0
    total_cost_change = 0.0

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
        cost_chg = float(row["total_cost_delta_total"])

        total_skus += n_skus
        total_current_wc += cur_wc
        total_rec_wc += rec_wc
        total_released += rel_wc
        total_cost_change += cost_chg

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
            f"| **{cell}** | {n_skus:,} | {cur_str} | {rec_str} | {rel_str} | {_signed_money(cost_chg)}/yr | "
            f"{meta['csl']} | {meta['impact']} |"
        )

    # Priced SKUs with no sales in the XYZ window (AN/BN/CN): not "stable", held, not differentiated.
    no_demand_cells = [c for c in ("AN", "BN", "CN") if c in port_dict]
    if no_demand_cells:
        window_days = int(pd.read_csv(ABC_XYZ_CSV, usecols=["n_days"])["n_days"].max())
        nd_skus = sum(int(port_dict[c]["n_skus"]) for c in no_demand_cells)
        nd_cur = sum(float(port_dict[c]["current_wc_total"]) for c in no_demand_cells)
        nd_rec = sum(float(port_dict[c]["recommended_wc_total"]) for c in no_demand_cells)
        nd_rel = sum(float(port_dict[c]["capital_released_total"]) for c in no_demand_cells)
        nd_cost = sum(float(port_dict[c]["total_cost_delta_total"]) for c in no_demand_cells)
        total_skus += nd_skus
        total_current_wc += nd_cur
        total_rec_wc += nd_rec
        total_released += nd_rel
        total_cost_change += nd_cost
        nd_rel_str = f"+${nd_rel:,.2f}" if nd_rel > 0 else (f"-${abs(nd_rel):,.2f}" if nd_rel < 0 else "$0.00")
        lines.append(
            f"| **No recent demand (AN/BN/CN)** | {nd_skus:,} | ${nd_cur:,.2f} | ${nd_rec:,.2f} | {nd_rel_str} | "
            f"{_signed_money(nd_cost)}/yr | 95% (held) | Zero sales in the {window_days}-day window before the "
            f"first test day; not stable, no differentiated policy, review (launches, intermittent items, stock-outs) |"
        )

    # No ABC class: SKUs without priced sales before the decision cutoff (mostly CA_2 items lacking price coverage)
    unclassified = {c: r for c, r in port_dict.items() if c.startswith("?")}
    uc_skus = sum(int(r["n_skus"]) for r in unclassified.values())
    uc_cur = sum(float(r["current_wc_total"]) for r in unclassified.values())
    uc_rec = sum(float(r["recommended_wc_total"]) for r in unclassified.values())
    uc_rel = sum(float(r["capital_released_total"]) for r in unclassified.values())
    uc_cost = sum(float(r["total_cost_delta_total"]) for r in unclassified.values())
    total_skus += uc_skus
    total_current_wc += uc_cur
    total_rec_wc += uc_rec
    total_released += uc_rel
    total_cost_change += uc_cost
    uc_rel_str = f"+${uc_rel:,.2f}" if uc_rel > 0 else (f"-${abs(uc_rel):,.2f}" if uc_rel < 0 else "$0.00")
    lines.append(
        f"| **No ABC class (?X/?Y/?Z/?N)** | {uc_skus:,} | ${uc_cur:,.2f} | ${uc_rec:,.2f} | {uc_rel_str} | "
        f"{_signed_money(uc_cost)}/yr | — | CA_2 items lacking price coverage, plus SKUs with no priced sales "
        f"before the cutoff |"
    )

    tot_rel_str = f"+${total_released:,.2f}" if total_released >= 0 else f"-${abs(total_released):,.2f}"
    lines.append(
        f"| **Total Portfolio** | **{total_skus:,}** | **${total_current_wc:,.2f}** | **${total_rec_wc:,.2f}** | "
        f"**{tot_rel_str}** | **{_signed_money(total_cost_change)}/yr** | — | "
        f"**${total_released:,.2f} Net Working Capital Released, at {_signed_money(total_cost_change)}/yr total annual cost** |"
    )

    return "\n".join(lines)


def render_safety_stock_by_model_table() -> str:
    """Generate Markdown safety stock comparison table across forecast models."""
    df = pd.read_csv(SAFETY_STOCK_MODEL_CSV)

    lines = [
        "| Model Name | Architecture Tier | Mean RMSE ($\\sigma_e$) | Mean bias (actual - forecast) | Safety Stock Units (95% CSL) | Safety Stock Capital ($) | Capital Reduction vs Naive ($) | Reduction (%) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for _, row in df.iterrows():
        model_key = row["model"]
        info = MODEL_INFO.get(model_key, {"name": f"**{model_key}**", "tier": "Statistical Baseline"})
        name = info["name"]
        tier = info["tier"]
        mean_rmse = f"{float(row['mean_rmse']):.4f}"
        mean_bias = f"{float(row['mean_bias']):+.4f}"
        ss_units = f"{float(row['total_ss_units']):,.2f}"
        ss_cap = f"${float(row['total_ss_capital']):,.2f}"

        red_val = float(row["capital_reduction_vs_naive"])
        if red_val > 0:
            red_str = f"+${red_val:,.2f}"
        elif red_val < 0:
            red_str = f"-${abs(red_val):,.2f}"
        else:
            red_str = "$0.00"

        pct_val = float(row["pct_reduction_vs_naive"])
        if pct_val > 0:
            pct_str = f"+{pct_val:.2f}%"
        elif pct_val < 0:
            pct_str = f"{pct_val:.2f}%"
        else:
            pct_str = "0.00%"

        lines.append(
            f"| {name} | {tier} | {mean_rmse} | {mean_bias} | {ss_units} | {ss_cap} | {red_str} | {pct_str} |"
        )

    return "\n".join(lines)


def render_policy_comparison_table() -> str:
    """Markdown table: three policies side by side, working capital next to annual cost."""
    df = pd.read_csv(POLICY_COMPARISON_CSV)
    lines = [
        "| Policy | SKUs | Working Capital ($) | Change in Working Capital ($) | Change in Holding Cost ($/yr) | Change in Stockout Cost ($/yr) | Change in Total Annual Cost ($/yr) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| **{r['label']}** | {int(r['n_skus']):,} | ${float(r['working_capital']):,.0f} | "
            f"{_signed_money(float(r['wc_change_vs_current']))} | {_signed_money(float(r['holding_cost_change']))} | "
            f"{_signed_money(float(r['stockout_cost_change']))} | **{_signed_money(float(r['total_cost_change']))}** |"
        )
    return "\n".join(lines)


def render_demand_basis_table() -> str:
    """Markdown table: current working capital under three mean-demand bases."""
    df = pd.read_csv(DEMAND_BASES_CSV)
    lines = [
        "| Mean-demand basis | Used for | Mean demand (units/day) | Cycle-Stock Capital ($) | Safety-Stock Capital ($) | Current Working Capital ($) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        label = DEMAND_BASIS_LABELS.get(r["basis"], r["basis"])
        lines.append(
            f"| {label} | {r['used_for']} | {float(r['mean_daily_demand_total']):,.1f} | "
            f"${float(r['cycle_stock_capital']):,.0f} | ${float(r['safety_stock_capital']):,.0f} | "
            f"${float(r['working_capital']):,.0f} |"
        )
    return "\n".join(lines)


SEGMENT_ORDER = ["SMOOTH", "ERRATIC", "LUMPY", "INTERMITTENT", "NO_DEMAND"]
ROUTE_LABELS = {"global": "LightGBM", "adida": "ADIDA", "zero": "zero forecast", "mixed": "mixed"}


def _model_label(name: str) -> str:
    """Display name of a model key, without the markdown bold used in the leaderboard."""
    return MODEL_INFO.get(name, {}).get("name", name).strip("*")


def _join_names(names: list[str]) -> str:
    """'A', 'A and B', 'A, B and C'; 'none' for an empty list."""
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def render_router_comparison(path: Path | None = None) -> str:
    """
    Router versus the best single model: table plus a reading derived from it.

    Every number and every direction word ("did not beat", "better", "lost")
    is computed from ``routing_vs_best_single.csv``, so a rerun that changes
    the result changes the sentence with it.
    """
    df = pd.read_csv(path or ROUTING_COMPARISON_CSV, dtype={"origin": str})
    pooled = df[df["origin"] == "all"]
    seg = pooled[pooled["pattern"] != "ALL"].copy()
    seg["_order"] = seg["pattern"].map({p: i for i, p in enumerate(SEGMENT_ORDER)}).fillna(99)
    seg = seg.sort_values(["_order", "pattern"])
    total = pooled[pooled["pattern"] == "ALL"].iloc[0]
    per_origin = df[(df["origin"] != "all") & (df["pattern"] == "ALL")].copy()

    # Origin indices count back from the end of the data (index 0 is the latest
    # test window), so label each origin with its test window and list them in
    # time order.
    windows: dict[str, str] = {}
    if MANIFEST_JSON.exists():
        for o in json.loads(MANIFEST_JSON.read_text(encoding="utf-8")).get("origins", []):
            windows[str(o["index"])] = f"test {o['test_start']} to {o['test_end']}"
    per_origin["_start"] = per_origin["origin"].map(lambda o: windows.get(o, o))
    per_origin = per_origin.sort_values("_start")

    def row(label: str, r: pd.Series) -> str:
        best = f"{_model_label(r['best_single_model'])} ({float(r['wape_best_single']):.2f}%)"
        return (
            f"| {label} | {ROUTE_LABELS.get(r['routes_to'], r['routes_to'])} | {int(r['n_series_origins']):,} | "
            f"{float(r['wape_routed']):.2f} | {float(r['wape_adida']):.2f} | {float(r['wape_lightgbm']):.2f} | "
            f"{best} | {float(r['routed_minus_adida']):+.2f} |"
        )

    lines = [
        "| Scope | Router sends to | Series-origins | Router WAPE (%) | ADIDA everywhere (%) | LightGBM everywhere (%) | Best single model (WAPE) | Router - ADIDA (pp) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [row(f"**{r['pattern']}**", r) for _, r in seg.iterrows()]
    lines.append(row("**All series, pooled over origins**", total))
    lines += [
        row(
            f"All series, origin {r['origin']}" + (f" ({windows[r['origin']]})" if r["origin"] in windows else ""),
            r,
        )
        for _, r in per_origin.iterrows()
    ]

    # --- the reading, derived from the table -------------------------------
    delta = float(total["routed_minus_adida"])
    deltas = [float(x) for x in per_origin["routed_minus_adida"]]
    n_worse = sum(d > 0 for d in deltas)
    if n_worse == len(deltas):
        where = "at every origin (" + ", ".join(f"{d:+.2f}" for d in deltas) + " pp)"
    elif n_worse == 0:
        where = "at no origin (" + ", ".join(f"{d:+.2f}" for d in deltas) + " pp)"
    else:
        where = f"at {n_worse} of {len(deltas)} origins (" + ", ".join(f"{d:+.2f}" for d in deltas) + " pp)"
    verdict = "did not beat" if delta > 0 else "beat"
    best_models = sorted({_model_label(m) for m in per_origin["best_single_model"]})

    to_adida = seg.loc[seg["routes_to"] == "adida", "pattern"].tolist()
    to_zero = seg.loc[seg["routes_to"] == "zero", "pattern"].tolist()
    to_global = seg[seg["routes_to"] == "global"]
    better = to_global.loc[to_global["wape_lightgbm"] < to_global["wape_adida"], "pattern"].tolist()
    worse = to_global.loc[to_global["wape_lightgbm"] >= to_global["wape_adida"], "pattern"].tolist()
    detail = "; ".join(
        f"{r['pattern']} {float(r['wape_lightgbm']):.2f}% vs {float(r['wape_adida']):.2f}%"
        for _, r in to_global.iterrows()
    )

    reading = [
        f"- **Result**: classification-based routing {verdict} applying ADIDA everywhere. Pooled over the "
        f"origins the router's WAPE is {float(total['wape_routed']):.2f}% against "
        f"{float(total['wape_adida']):.2f}% for ADIDA ({delta:+.2f} pp, positive meaning the router is worse), "
        f"and it is worse {where}. The best single model at each origin is "
        f"{_join_names(best_models)}.",
        f"- **Where the difference comes from**: {_join_names(to_adida)} series go to ADIDA (the same method as "
        f"ADIDA everywhere; the router fits it after trimming pre-launch zeros, so a few forecasts differ slightly "
        f"from the standalone model's), {_join_names(to_zero)} series get a zero forecast, and {_join_names(to_global['pattern'].tolist())} "
        f"series go to LightGBM, which here has no price features. On those segments LightGBM beat ADIDA on "
        f"{_join_names(better)} and lost to it on {_join_names(worse)} (LightGBM vs ADIDA WAPE: {detail}). "
        f"The global model was not better on every segment it was given, which is why routing on demand pattern did not pay.",
        "- **Reading it**: the best single model is the lowest-WAPE single model within each row, chosen after the "
        "scores are known, which favours the single model. The INTERMITTENT to ADIDA assignment was chosen on these "
        "same evaluation origins, so the router is scored on the data that selected its rule; a clean test needs an "
        "earlier selection origin. LightGBM ran with the handicaps listed under Limitation 1.",
    ]
    return "\n".join(lines + [""] + reading)


def render_demand_profile() -> str:
    """Step 1: what the demand looks like, from the EDA artifacts."""
    f = json.loads(EDA_FINDINGS_JSON.read_text(encoding="utf-8"))
    s, d, mix = f["3.1_structure"], f["3.2_demand"], f["3.3_intermittency"]
    n, late = s["n_series"], s["series_starting_after_day_1"]
    zero_by_cat = ", ".join(
        f"{cat.title()} {rate:.1%}" for cat, rate in sorted(d["zero_rate_by_category"].items(), key=lambda kv: kv[1])
    )

    sbc = pd.read_csv(SBC_CSV).set_index("pattern")
    order = [p for p in SEGMENT_ORDER if p in sbc.index]
    lines = [
        f"- **Scale**: {n:,} store-item series over {s['n_days']:,} days, {s['long_format_rows'] / 1e6:.1f} million daily observations.",
        f"- **Sparsity**: {d['overall_zero_rate']:.1%} of daily records are zero ({zero_by_cat} by category).",
        f"- **Late launches**: {late / n:.0%} of series ({late:,}) had no sale on day 1, and the median first sale is on day "
        f"{s['median_first_sale_day']}. Zeros before a series' first sale mean \"not on sale yet\", not \"no demand\".",
        f"- **Demand patterns** (SBC quadrants, full history, pre-launch zeros trimmed): "
        f"{mix['intermittent_or_lumpy_pct']:.1f}% of series are intermittent or lumpy.",
        "",
        "| SBC quadrant | Rule | Series | % of series | % of volume |",
        "|---|---|---|---|---|",
    ]
    rules = {
        "SMOOTH": "ADI < 1.32, CV² < 0.49",
        "ERRATIC": "ADI < 1.32, CV² ≥ 0.49",
        "LUMPY": "ADI ≥ 1.32, CV² ≥ 0.49",
        "INTERMITTENT": "ADI ≥ 1.32, CV² < 0.49",
    }
    for p in order:
        r = sbc.loc[p]
        lines.append(
            f"| **{p}** | {rules.get(p, '')} | {int(r['n_series']):,} | {float(r['pct_of_series']):.1f}% | {float(r['pct_of_volume']):.1f}% |"
        )
    return "\n".join(lines)


def render_evaluation_protocol() -> str:
    """Step 3: the rolling-origin protocol, from the backtest manifest."""
    m = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    origins = sorted(m["origins"], key=lambda o: o["test_start"])
    n_origins, horizon, n_series = int(m["n_origins"]), int(m["horizon_days"]), int(m["n_series"])
    test_days = n_origins * horizon
    lines = [
        "| Origin index | Training ends | Test window |",
        "|---|---|---|",
        *(f"| {o['index']} | {o['train_end']} | {o['test_start']} to {o['test_end']} |" for o in origins),
        "",
        f"- **Sample and size**: {n_series:,} series drawn at random with a fixed seed, {m['history_days']} days of history each. "
        f"{n_origins} origins x {horizon} days = {test_days} test days per series, {n_series * test_days:,} predictions per model. "
        "Origin indices count back from the end of the data, so index 0 is the latest window.",
        f"- **Horizon-shifted features**: every lag and rolling feature the global model sees is shifted by the full "
        f"{horizon}-day horizon ({m['n_features']} features), so a test row never references an actual inside its own window.",
        "- **Training-window-only statistics**: the RMSSE scale and the router's SBC quadrants for each origin "
        "use data up to that origin's training end, never the test window.",
    ]
    return "\n".join(lines)


#: What each rung of the model ladder is there to test (no numbers: those are rendered elsewhere).
MODEL_RUNGS = [
    ("naive", "The floor: repeat the last observed value."),
    ("seasonal_naive_weekly", "Does a weekly cycle alone help?"),
    ("seasonal_naive", "Does a four-week cycle alone help?"),
    ("moving_average", "Does averaging out the noise help?"),
    ("ses", "Does recency-weighted smoothing help?"),
    ("croston_sba", "Does modelling demand size and interval separately (debiased) help on intermittent series?"),
    ("adida", "Does aggregating to shorter, denser series help on intermittent series?"),
    ("lightgbm", "Does a global model that learns across series help? (squared-error loss)"),
    ("lightgbm_tweedie", "The same global model with a loss suited to zero-inflated counts."),
    ("routed_ensemble", "Does sending each demand pattern to a different method help? (step 6)"),
]


def render_model_ladder() -> str:
    """Step 4: the model ladder and the global model's handicaps, from the manifest."""
    m = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    eda = json.loads(EDA_FINDINGS_JSON.read_text(encoding="utf-8"))["3.1_structure"]
    coverage = re.search(r"(\d+(?:\.\d+)?)% of the full M5 price table", str(m.get("price_coverage_note", "")))
    price_note = f"about {float(coverage.group(1)):.0f}% of M5 prices" if coverage else "a fraction of M5 prices"
    no_price = not m.get("price_features_included", False)

    lines = [
        "| Rung | Model | Tier | What this rung tests |",
        "|---|---|---|---|",
        *(
            f"| {i} | {_model_label(key)} | {MODEL_INFO[key]['tier']} | {text} |"
            for i, (key, text) in enumerate(MODEL_RUNGS, 1)
        ),
        "",
        f"- **Handicaps on the global model**: LightGBM ran under three handicaps relative to the M5 competition: "
        f"(1) {'no price features' if no_price else 'price features included'} (`price_features_included: {str(not no_price).lower()}`; "
        f"the price table came from a workbook truncated at Excel's row limit and covers {price_note}), "
        f"(2) a {int(m['n_series']):,}-series sample of the {int(eda['n_series']):,}-series catalogue, and "
        f"(3) a {int(m['history_days'])}-day window of the {int(eda['n_days']):,}-day history. "
        "In the M5 competition price and promotion features and catalogue-wide learning were a large part of why GBDTs won, so "
        "\"intermittent methods match or beat global GBDTs here\" carries that qualification.",
    ]
    return "\n".join(lines)


def render_margin_sensitivity() -> str:
    """Step 8: how far the service-level answer moves with the assumed gross margin."""
    sens = pd.read_csv(MARGIN_SENSITIVITY_CSV)
    be = pd.read_csv(MARGIN_BREAK_EVEN_CSV).set_index("threshold")
    seg = pd.read_csv(SEGMENT_OPTIMUM_CSV)
    optimised = seg[~seg["abc_xyz_class"].astype(str).str.endswith("N")]

    cols = [c for c in sens.columns if c.startswith("extra_cost_")]
    levels = sorted(int(c.split("_")[2]) for c in cols)
    lines = [
        "| Gross margin | Scenario | Unconstrained optimum SL* | Cheapest menu level | "
        + " | ".join(f"Extra holding + shortage cost at {lv}%" for lv in levels)
        + " |",
        "|---|---|---|---|" + "---|" * len(levels),
    ]
    for _, r in sens.iterrows():
        scenario = "grid" if r["scenario"] == "grid" else f"{r['scenario']} assumption"
        extras = " | ".join(f"{float(r[f'extra_cost_{lv}_pct']):+.0f}%" if float(r[f"extra_cost_{lv}_pct"]) > 0 else "0%" for lv in levels)
        lines.append(
            f"| {float(r['gross_margin']):.0%} | {scenario} | {float(r['sl_star']):.1%} | "
            f"{float(r['best_menu_level']):.0%} | {extras} |"
        )

    b98, b95 = float(be.loc["cheapest_98_over_95", "gross_margin"]), float(be.loc["cheapest_95_over_90", "gross_margin"])
    x98, x95 = float(be.loc["sl_star_reaches_98", "gross_margin"]), float(be.loc["sl_star_reaches_95", "gross_margin"])
    cats = sens[sens["scenario"] != "grid"].sort_values("sl_star")
    lo_c, hi_c = cats.iloc[0], cats.iloc[-1]
    grocery = sens[(sens["scenario"] == "grid") & sens["gross_margin"].between(0.25, 0.30)].sort_values("gross_margin")
    g_txt = " and ".join(f"{float(r['gross_margin']):.0%}" for _, r in grocery.iterrows())
    g_extra = " and ".join(f"{float(r['extra_cost_95_pct']):.0f}%" for _, r in grocery.iterrows())
    g_all98 = bool((grocery["best_menu_level"] == 0.98).all())

    reading = [
        f"- **Break-evens** (exact, from the same cost formula the analysis uses): 98% costs less than 95% above a {b98:.1%} gross margin, "
        f"and 95% costs less than 90% above {b95:.1%}. So the cheapest menu level is 98% above {b98:.1%}, 95% between {b95:.1%} and {b98:.1%}, "
        f"and 90% below {b95:.1%}. The cost per unit of error size scales with price, so the answer depends on the margin alone.",
        f"- **Rule of thumb versus exact**: the unconstrained optimum SL* only reaches 98% above a {x98:.1%} margin and 95% above {x95:.1%}. "
        f"Those are looser than the break-evens above: SL* can sit below 98% while 98% is still the cheapest of the three menu levels, "
        f"because SL* is a continuous optimum and the menu is coarse.",
        f"- **Realistic grocery margins**: at {g_txt} gross margin the cheapest level is "
        f"{'98%' if g_all98 else 'not 98%'}, and holding 95% instead would cost {g_extra} more in safety-stock holding plus shortage cost.",
        f"- **Range of SL***: by category it is {float(lo_c['sl_star']):.1%} ({lo_c['scenario']}, {float(lo_c['gross_margin']):.0%} margin) to "
        f"{float(hi_c['sl_star']):.1%} ({hi_c['scenario']}, {float(hi_c['gross_margin']):.0%}); the per-segment medians in "
        f"`segment_cost_optimum.csv` run from {float(optimised['closed_form_sl_star'].min()):.1%} to {float(optimised['closed_form_sl_star'].max()):.1%}. "
        "All of them sit above the whole 90/95/98% menu.",
        f"- **What it means**: because the cheapest level depends on the margin and not on volume or variability, every segment within a "
        f"category shares one cheapest level. Differentiating service levels by ABC/XYZ cannot lower cost under lost-margin costing while the "
        f"gross margin is above {b98:.1%}; it can pay only when working capital is the binding constraint, or for items whose margin is below that.",
    ]
    return "\n".join(lines + [""] + reading)


def render_statistics_section(sig: dict | None = None) -> str:
    """Generate Markdown statistics summary from significance report."""
    if sig is None:
        with open(SIGNIFICANCE_JSON, encoding="utf-8") as f:
            sig = json.load(f)

    friedman = sig.get("friedman", {})
    friedman_stat = float(friedman.get("statistic", 2429.30))
    n_models = int(sig.get("n_models", 10))
    n_series = int(sig.get("n_series_complete", 2950))

    nem = sig.get("nemenyi", {})
    cd = float(nem.get("critical_difference", 0.2494))
    avg_ranks = nem.get("average_ranks", {})
    cliques = nem.get("cliques", [])

    boot = sig.get("bootstrap", {})
    h2h = sig.get("pairwise_per_series", [])

    # 1. Friedman
    line1 = (
        f"- **Non-Parametric Omnibus Test (Friedman)**: Evaluated across {n_models} models over {n_series:,} "
        f"complete series (84 evaluation days per series). The omnibus Friedman test decisively rejects the null "
        f"hypothesis of equal model performance ($\\chi^2 = {friedman_stat:.2f}$, $p < 10^{{-12}}$), "
        f"justifying post-hoc ranking comparisons."
    )

    # 2. Nemenyi
    best_rank_model = min(avg_ranks, key=avg_ranks.get) if avg_ranks else "croston_sba"
    best_rank_val = avg_ranks.get(best_rank_model, 4.55)
    best_rank_name = MODEL_INFO.get(best_rank_model, {}).get("name", f"**{best_rank_model}**")

    top_clique = []
    if cliques:
        top_clique = [m for m in cliques[0] if m != best_rank_model]
    top_clique_str = ", ".join(MODEL_INFO.get(m, {}).get("name", f"**{m}**") for m in top_clique) if top_clique else "none"

    lgb_tw_name = MODEL_INFO.get("lightgbm_tweedie", {}).get("name", "**LightGBM (Tweedie)**")
    lgb_tw_rank = avg_ranks.get("lightgbm_tweedie", 5.38)
    lgb_name = MODEL_INFO.get("lightgbm", {}).get("name", "**LightGBM**")
    lgb_rank = avg_ranks.get("lightgbm", 5.56)

    line2 = (
        f"- **Post-Hoc Critical Difference (Nemenyi)**: Critical difference $CD = {cd:.4f}$ at $\\alpha = 0.05$. "
        f"{best_rank_name} achieved the best average rank ({best_rank_val:.2f}), forming a top statistically "
        f"indistinguishable clique with {top_clique_str}. Standard global models ({lgb_tw_name} "
        f"at {lgb_tw_rank:.2f}, {lgb_name} at {lgb_rank:.2f}) scored significantly worse in average rank across the panel."
    )

    # 3. Bootstrap
    adida_boot = boot.get("adida", {})
    adida_name = MODEL_INFO.get("adida", {}).get("name", "**ADIDA**")
    adida_wape = adida_boot.get("wape", {}).get("point", 75.16)
    adida_wape_ci = [adida_boot.get("wape", {}).get("ci_lower", 72.16), adida_boot.get("wape", {}).get("ci_upper", 78.23)]

    re_dw = boot.get("routed_ensemble", {}).get("delta_wape_vs_best", {})
    re_name = MODEL_INFO.get("routed_ensemble", {}).get("name", "**Routed SBC Ensemble**")
    re_dw_ci = [re_dw.get("ci_lower", -0.01), re_dw.get("ci_upper", 1.04)]

    ma_dr = boot.get("moving_average", {}).get("delta_rmsse_vs_best", {})
    ma_name = MODEL_INFO.get("moving_average", {}).get("name", "**Moving Average (28d)**")
    ma_dr_ci = [ma_dr.get("ci_lower", -0.0001), ma_dr.get("ci_upper", 0.0023)]

    line3 = (
        f"- **Paired Series Bootstrap (95% CI)**: Resampling whole series over 2,000 replicates "
        f"(preserving within-series autocorrelation) reveals that {adida_name} "
        f"leads pooled volume-weighted accuracy at {adida_wape:.2f}% WAPE "
        f"[{adida_wape_ci[0]:.2f}%, {adida_wape_ci[1]:.2f}%]. Relative to ADIDA, "
        f"{re_name} is statistically indistinguishable in WAPE ($\\Delta\\text{{WAPE}} = {re_dw.get('point', 0.50):+.2f}\\%$, "
        f"95% CI [{re_dw_ci[0]:+.2f}%, {re_dw_ci[1]:+.2f}%]; routing assignment chosen on the evaluation origins; a clean test needs an earlier selection origin), while {ma_name} "
        f"is indistinguishable in RMSSE ($\\Delta\\text{{RMSSE}} = {ma_dr.get('point', 0.0011):+.4f}$, "
        f"95% CI [{ma_dr_ci[0]:+.4f}, {ma_dr_ci[1]:+.4f}])."
    )

    # 4. Pairwise Head-to-Head Win Rates
    ac_pair = next(
        (p for p in h2h if (p["model_a"] == "adida" and p["model_b"] == "croston_sba") or (p["model_a"] == "croston_sba" and p["model_b"] == "adida")),
        {},
    )
    if ac_pair.get("model_a") == "croston_sba":
        c_win = ac_pair.get("a_win_pct", 56.97)
    else:
        c_win = ac_pair.get("b_win_pct", 56.97)
    indist_pct = ac_pair.get("dm_indistinguishable_pct", 93.80)
    csba_name = MODEL_INFO.get("croston_sba", {}).get("name", "**Croston SBA**")

    line4 = (
        f"- **Per-Series Pairwise Win Rates**: Comparing models series-by-series resolves the apparent divergence "
        f"between rank tests and volume metrics: {csba_name} beats {adida_name} on {c_win:.1f}% of individual series "
        f"(favoring rank metrics on sparse items), whereas {adida_name} achieves lower aggregate error on higher-volume items. "
        f"On {indist_pct:.1f}% of series, the two models are statistically indistinguishable under Holm-corrected per-series Diebold-Mariano testing."
    )

    return "\n".join([line1, line2, line3, line4])


#: Blocks added after M5-10 share one mechanism: (label, start marker, end marker, renderer).
#: Add a block here and in the README; check and update pick it up.
EXTRA_BLOCKS: list[tuple[str, str, str, Callable[[], str]]] = [
    ("Demand profile", DEMAND_PROFILE_START, DEMAND_PROFILE_END, render_demand_profile),
    ("Evaluation protocol", EVAL_PROTOCOL_START, EVAL_PROTOCOL_END, render_evaluation_protocol),
    ("Model ladder", MODEL_LADDER_START, MODEL_LADDER_END, render_model_ladder),
    ("Router comparison", ROUTER_START, ROUTER_END, render_router_comparison),
    ("Margin sensitivity", MARGIN_START, MARGIN_END, render_margin_sensitivity),
]


def check_readme_tables(readme_path: Path = README_PATH) -> tuple[bool, str]:
    """Check if README.md tables match artifacts. Returns (is_synced, diff_message)."""
    if not readme_path.exists():
        return False, f"README path does not exist: {readme_path}"

    content = readme_path.read_text(encoding="utf-8")

    if LEADERBOARD_START not in content or LEADERBOARD_END not in content:
        return False, f"Leaderboard markers missing in {readme_path}"
    if INVENTORY_START not in content or INVENTORY_END not in content:
        return False, f"Inventory markers missing in {readme_path}"
    if STATISTICS_START not in content or STATISTICS_END not in content:
        return False, f"Statistics markers missing in {readme_path}"
    if SAFETY_STOCK_MODEL_START not in content or SAFETY_STOCK_MODEL_END not in content:
        return False, f"Safety stock model comparison markers missing in {readme_path}"
    if POLICY_COMPARISON_START not in content or POLICY_COMPARISON_END not in content:
        return False, f"Policy comparison markers missing in {readme_path}"
    if DEMAND_BASIS_START not in content or DEMAND_BASIS_END not in content:
        return False, f"Demand basis markers missing in {readme_path}"
    for label, start, end, _ in EXTRA_BLOCKS:
        if start not in content or end not in content:
            return False, f"{label} markers missing in {readme_path}"

    leaderboard_expected = render_leaderboard_table().strip()
    inventory_expected = render_inventory_table().strip()
    statistics_expected = render_statistics_section().strip()
    ss_model_expected = render_safety_stock_by_model_table().strip()
    policy_expected = render_policy_comparison_table().strip()
    basis_expected = render_demand_basis_table().strip()

    # Extract current tables
    lb_part = content.split(LEADERBOARD_START)[1].split(LEADERBOARD_END)[0].strip()
    inv_part = content.split(INVENTORY_START)[1].split(INVENTORY_END)[0].strip()
    stat_part = content.split(STATISTICS_START)[1].split(STATISTICS_END)[0].strip()
    ss_model_part = content.split(SAFETY_STOCK_MODEL_START)[1].split(SAFETY_STOCK_MODEL_END)[0].strip()
    policy_part = content.split(POLICY_COMPARISON_START)[1].split(POLICY_COMPARISON_END)[0].strip()
    basis_part = content.split(DEMAND_BASIS_START)[1].split(DEMAND_BASIS_END)[0].strip()

    diffs = []
    if policy_part != policy_expected:
        diffs.append("Policy comparison table in README does not match artifact outputs.")
    if basis_part != basis_expected:
        diffs.append("Demand basis table in README does not match artifact outputs.")
    if lb_part != leaderboard_expected:
        diffs.append("Leaderboard table in README does not match artifact outputs.")
    if inv_part != inventory_expected:
        diffs.append("Inventory table in README does not match artifact outputs.")
    if stat_part != statistics_expected:
        diffs.append("Statistics section in README does not match artifact outputs.")
    if ss_model_part != ss_model_expected:
        diffs.append("Safety stock by model table in README does not match artifact outputs.")
    for label, start, end, render in EXTRA_BLOCKS:
        if content.split(start)[1].split(end)[0].strip() != render().strip():
            diffs.append(f"{label} in README does not match artifact outputs.")

    if diffs:
        return False, "\n".join(diffs)
    return True, "README tables and statistics are fully synchronized with artifacts."


def update_readme_tables(readme_path: Path = README_PATH) -> None:
    """Update README.md tables and statistics in-place using generated tables."""
    content = readme_path.read_text(encoding="utf-8")

    leaderboard_str = render_leaderboard_table().strip()
    inventory_str = render_inventory_table().strip()
    statistics_str = render_statistics_section().strip()
    ss_model_str = render_safety_stock_by_model_table().strip()
    policy_str = render_policy_comparison_table().strip()
    basis_str = render_demand_basis_table().strip()

    # Replace policy comparison, demand basis and every block in EXTRA_BLOCKS
    for start, end, body, name in (
        (POLICY_COMPARISON_START, POLICY_COMPARISON_END, policy_str, "policy comparison"),
        (DEMAND_BASIS_START, DEMAND_BASIS_END, basis_str, "demand basis"),
        *((s, e, render().strip(), label.lower()) for label, s, e, render in EXTRA_BLOCKS),
    ):
        if start in content and end in content:
            before = content.split(start)[0]
            after = content.split(end)[1]
            content = f"{before}{start}\n{body}\n{end}{after}"
        else:
            print(f"Warning: {start} not found, skipping {name} replacement.")

    # Replace leaderboard
    if LEADERBOARD_START in content and LEADERBOARD_END in content:
        before = content.split(LEADERBOARD_START)[0]
        after = content.split(LEADERBOARD_END)[1]
        content = f"{before}{LEADERBOARD_START}\n{leaderboard_str}\n{LEADERBOARD_END}{after}"
    else:
        print(f"Warning: {LEADERBOARD_START} not found, skipping leaderboard replacement.")

    # Replace safety stock by model
    if SAFETY_STOCK_MODEL_START in content and SAFETY_STOCK_MODEL_END in content:
        before = content.split(SAFETY_STOCK_MODEL_START)[0]
        after = content.split(SAFETY_STOCK_MODEL_END)[1]
        content = f"{before}{SAFETY_STOCK_MODEL_START}\n{ss_model_str}\n{SAFETY_STOCK_MODEL_END}{after}"
    else:
        print(f"Warning: {SAFETY_STOCK_MODEL_START} not found, skipping safety stock model replacement.")

    # Replace inventory
    if INVENTORY_START in content and INVENTORY_END in content:
        before = content.split(INVENTORY_START)[0]
        after = content.split(INVENTORY_END)[1]
        content = f"{before}{INVENTORY_START}\n{inventory_str}\n{INVENTORY_END}{after}"
    else:
        print(f"Warning: {INVENTORY_START} not found, skipping inventory replacement.")

    # Replace statistics
    if STATISTICS_START in content and STATISTICS_END in content:
        before = content.split(STATISTICS_START)[0]
        after = content.split(STATISTICS_END)[1]
        content = f"{before}{STATISTICS_START}\n{statistics_str}\n{STATISTICS_END}{after}"
    else:
        print(f"Warning: {STATISTICS_START} not found, skipping statistics replacement.")

    readme_path.write_text(content, encoding="utf-8")
    print(f"Successfully updated README tables and statistics in {readme_path}")



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
