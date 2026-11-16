"""
Test that README benchmark and inventory tables match their backing artifacts.

Ensures no manually edited numbers or out-of-sync metrics can survive in the public README.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_readme_tables import (
    render_leaderboard_table,
    render_inventory_table,
    check_readme_tables,
)


def test_readme_tables_match_artifacts():
    """Verify that the README tables match the committed artifacts exactly."""
    is_synced, diff_msg = check_readme_tables()
    assert is_synced, f"README tables are out of sync with artifacts:\n{diff_msg}"
