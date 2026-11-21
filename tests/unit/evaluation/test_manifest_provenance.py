"""
G4b: the backtest manifest records where its predictions came from, and a rerun keeps doing so.

``predictions_file``, ``predictions_sha256`` and ``regeneration_command`` were added to
``manifest.json`` by hand after a run, so re-running the backtest silently dropped them.
``manifest_provenance`` now writes them, and these tests pin the helper and tie the committed
manifest to the committed predictions and to the DVC stage command.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.evaluation.backtest import manifest_provenance

ROOT = Path(__file__).resolve().parents[3]
BACKTEST = ROOT / "reports" / "backtest_main"


def test_manifest_provenance_hashes_the_saved_predictions(tmp_path: Path) -> None:
    (tmp_path / "predictions.parquet").write_bytes(b"not really parquet, but bytes are bytes")
    out = manifest_provenance(tmp_path, ["--sample-series", "3000", "--out-dir", "reports/backtest_main"])
    assert out["predictions_file"] == "predictions.parquet"
    assert out["predictions_sha256"] == hashlib.sha256(b"not really parquet, but bytes are bytes").hexdigest()
    assert out["regeneration_command"] == "python scripts/pipeline/run_backtest.py --sample-series 3000 --out-dir reports/backtest_main"


def test_hash_changes_when_the_predictions_change(tmp_path: Path) -> None:
    f = tmp_path / "predictions.parquet"
    f.write_bytes(b"a")
    first = manifest_provenance(tmp_path, [])["predictions_sha256"]
    f.write_bytes(b"b")
    assert manifest_provenance(tmp_path, [])["predictions_sha256"] != first


def test_committed_manifest_matches_the_committed_predictions() -> None:
    manifest = json.loads((BACKTEST / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((BACKTEST / manifest["predictions_file"]).read_bytes()).hexdigest()
    assert manifest["predictions_sha256"] == digest


def test_committed_regeneration_command_is_the_dvc_stage_command() -> None:
    manifest = json.loads((BACKTEST / "manifest.json").read_text(encoding="utf-8"))
    dvc = (ROOT / "dvc.yaml").read_text(encoding="utf-8")
    cmd = re.search(r"^\s*backtest:\s*\n\s*cmd:\s*(.+)$", dvc, flags=re.M)
    assert cmd is not None
    assert manifest["regeneration_command"] == cmd.group(1).strip()
