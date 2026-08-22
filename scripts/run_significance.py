"""
Run the significance battery over an existing backtest's predictions.

Why this must run before generating more results
------------------------------------------------
The current leaderboard spans 75.17 to 77.46 WAPE across the top six models,
with RMSSE clustered at 0.98-1.01. A 2.3-point WAPE spread over 3,000
heterogeneous series is well inside the range that sampling noise produces.
Until a test says otherwise, "ADIDA beats LightGBM" is a ranking, not a
finding, and reporting it as a finding is precisely the failure mode this
project exists to avoid.

Test order is not arbitrary:

  1. Friedman (omnibus) -- is ANY model different? If this does not reject,
     no pairwise claim is licensed and the post-hoc tests are skipped.
  2. Nemenyi (post-hoc) -- which specific pairs differ, and what is the
     critical difference below which models are indistinguishable.
  3. MCB -- how does each model compare against the BEST one, which is the
     question a reader actually asks.
  4. Diebold-Mariano with Holm correction -- pairwise, on the pooled
     per-observation loss differential, accounting for the autocorrelation
     that multi-step forecast errors always carry.

Text output only. Figures 4 and 5 belong to Pillar 5.

Usage:
    python scripts/run_significance.py
    python scripts/run_significance.py --backtest-dir reports/backtest_main --metric rmsse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging import get_logger  # noqa: E402
from src.config.settings import settings  # noqa: E402
from src.evaluation.significance import (  # noqa: E402
    friedman_test,
    mcb_test,
    nemenyi_test,
    pairwise_dm,
    per_series_scores,
)

log = get_logger("significance")


def main() -> int:
    p = argparse.ArgumentParser(description="Significance testing on backtest output")
    p.add_argument("--backtest-dir", default="reports/backtest_main")
    p.add_argument("--metric", default="wape", choices=["wape", "rmsse", "mae", "rmse"])
    p.add_argument("--horizon", type=int, default=28)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    bt_dir = Path(args.backtest_dir)
    if not bt_dir.is_absolute():
        bt_dir = settings.base_dir / bt_dir

    preds_path = bt_dir / "predictions.parquet"
    if not preds_path.exists():
        log.error("predictions_not_found", path=str(preds_path))
        print(f"\nNo predictions at {preds_path}. Run the backtest first.\n")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else bt_dir / "significance"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(preds_path)
    preds = preds.dropna(subset=["y_true", "y_pred"])

    # --- per-series score matrix (series x model) ---
    matrix = per_series_scores(preds, metric=args.metric)
    # Friedman needs complete blocks: a model missing on some series would
    # otherwise be ranked against a different subset than its competitors.
    complete = matrix.dropna(axis=0, how="any")

    models = list(complete.columns)
    pooled = (
        preds.groupby("model", observed=True)
        .apply(
            lambda g: 100.0 * np.abs(g["y_true"] - g["y_pred"]).sum() / np.abs(g["y_true"]).sum(),
            include_groups=False,
        )
        .sort_values()
    )

    print("\n" + "=" * 82)
    print(f"SIGNIFICANCE TESTING  --  metric: {args.metric.upper()}  alpha: {args.alpha}")
    print("=" * 82)
    print(f"  source        : {preds_path.relative_to(settings.base_dir)}")
    print(f"  models        : {len(models)}")
    print(f"  series        : {len(complete):,} with complete scores "
          f"(of {len(matrix):,} total)")
    print(f"  observations  : {len(preds):,}")

    print("\n--- Pooled leaderboard (for reference; NOT a significance claim) " + "-" * 17)
    for name, value in pooled.items():
        print(f"    {name:<24} {value:8.3f}")

    # ---------------------------------------------------------------- 1. Friedman
    print("\n--- 1. FRIEDMAN omnibus test " + "-" * 53)
    friedman = friedman_test(complete, alpha=args.alpha)
    print(f"    chi2 = {friedman.statistic:.3f}   p = {friedman.p_value:.4g}")
    print(f"    {friedman.summary()}")

    results: dict[str, object] = {
        "metric": args.metric,
        "alpha": args.alpha,
        "n_models": len(models),
        "n_series_complete": int(len(complete)),
        "n_observations": int(len(preds)),
        "pooled_wape": {k: round(float(v), 4) for k, v in pooled.items()},
        "friedman": {
            "statistic": friedman.statistic,
            "p_value": friedman.p_value,
            "significant": friedman.significant,
        },
    }

    if not friedman.significant:
        print("\n    Omnibus did not reject. Post-hoc tests are NOT licensed and are")
        print("    skipped. The models are statistically indistinguishable on this data.")
        (out_dir / "significance_report.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        return 0

    # ---------------------------------------------------------------- 2. Nemenyi
    print("\n--- 2. NEMENYI post-hoc " + "-" * 58)
    nem = nemenyi_test(complete, alpha=args.alpha)
    cd = nem["critical_difference"]
    ranks = nem["average_ranks"]

    print(f"    critical difference (CD) = {cd:.4f}")
    print(f"    {'model':<24} {'avg rank':>9}   gap to best")
    best_rank = float(ranks.iloc[0])
    for name, rank in ranks.items():
        gap = float(rank) - best_rank
        marker = "  <-- best" if gap == 0 else ("  (within CD of best)" if gap <= cd else "")
        print(f"    {name:<24} {rank:9.3f}   {gap:+8.3f}{marker}")

    print("\n    Cliques (models joined by a bar are indistinguishable):")
    if nem["cliques"]:
        for clique in nem["cliques"]:
            print(f"      {' = '.join(clique)}")
    else:
        print("      (none -- every model is separable from every other)")

    results["nemenyi"] = {
        "critical_difference": float(cd),
        "average_ranks": {k: round(float(v), 4) for k, v in ranks.items()},
        "cliques": nem["cliques"],
    }

    # ---------------------------------------------------------------- 3. MCB
    print("\n--- 3. MULTIPLE COMPARISONS WITH THE BEST " + "-" * 40)
    mcb = mcb_test(complete, alpha=args.alpha)
    print(f"    {'model':<24} {'rank':>7} {'CI low':>8} {'CI high':>8}   verdict")
    for _, row in mcb.iterrows():
        verdict = (
            "BEST" if row["is_best"]
            else ("indistinguishable from best" if row["indistinguishable_from_best"]
                  else "significantly worse")
        )
        print(f"    {row['model']:<24} {row['average_rank']:7.3f} "
              f"{row['ci_lower']:8.3f} {row['ci_upper']:8.3f}   {verdict}")
    mcb.to_csv(out_dir / "mcb.csv", index=False)
    results["mcb"] = mcb.to_dict(orient="records")

    # ---------------------------------------------------------------- 4. DM
    print("\n--- 4. DIEBOLD-MARIANO (pairwise, Holm-corrected) " + "-" * 32)
    errors = {
        str(model): (g.sort_values(["series_id", "date"])["y_true"].to_numpy(float)
                     - g.sort_values(["series_id", "date"])["y_pred"].to_numpy(float))
        for model, g in preds.groupby("model", observed=True)
    }
    dm = pairwise_dm(errors, horizon=args.horizon, loss="absolute",
                     alpha=args.alpha, correction="holm")
    dm.to_csv(out_dir / "diebold_mariano.csv", index=False)

    n_sep = int(dm["significant_adjusted"].sum())
    print(f"    {n_sep} of {len(dm)} pairs separable at alpha={args.alpha} after Holm correction")

    indistinguishable = dm[~dm["significant_adjusted"]]
    if len(indistinguishable):
        print(f"\n    INDISTINGUISHABLE pairs ({len(indistinguishable)}):")
        for _, r in indistinguishable.iterrows():
            print(f"      {r['model_a']:<22} vs {r['model_b']:<22} "
                  f"p_adj = {r['p_adjusted']:.4f}")

    print("\n    Most significant separations:")
    for _, r in dm.head(8).iterrows():
        print(f"      {r['model_a']:<22} vs {r['model_b']:<22} "
              f"DM={r['dm_statistic']:+7.2f}  p_adj={r['p_adjusted']:.3g}  "
              f"better: {r['better']}")

    results["diebold_mariano"] = {
        "n_pairs": int(len(dm)),
        "n_separable_holm": n_sep,
        "indistinguishable_pairs": indistinguishable[
            ["model_a", "model_b", "p_adjusted"]
        ].to_dict(orient="records"),
    }

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 82)
    print("VERDICT")
    print("=" * 82)

    best = str(ranks.index[0])
    tied_with_best = [
        str(r["model"]) for _, r in mcb.iterrows()
        if r["indistinguishable_from_best"] and not r["is_best"]
    ]

    if tied_with_best:
        print(f"  Best by average rank: {best}")
        print(f"  Statistically indistinguishable from it: {', '.join(tied_with_best)}")
        print(f"  -> The ranking among {{{best}, {', '.join(tied_with_best)}}} is NOT a finding.")
    else:
        print(f"  {best} significantly outperforms every other model tested")
        print(f"  (MCB, alpha={args.alpha}; CD={cd:.4f}).")

    results["verdict"] = {
        "best_by_rank": best,
        "indistinguishable_from_best": tied_with_best,
        "ranking_is_separable": not tied_with_best,
    }

    (out_dir / "significance_report.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n  Artefacts: {out_dir}")
    print("=" * 82 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
