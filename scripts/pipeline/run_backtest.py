"""
CLI entry point for the walk-forward backtest.

The engine itself lives in `src/evaluation/backtest.py` (spec Part 9); this
file is the thin `scripts/` wrapper DVC and Airflow invoke.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.backtest import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
