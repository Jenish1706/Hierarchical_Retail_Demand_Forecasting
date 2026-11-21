# Operations & Execution Runbook

This runbook provides the definitive operational manual for running, reproducing, testing, and troubleshooting the platform on Windows environments.

---

## 1. Environment Setup (Windows 10 / 11)

### Prerequisites
- Python 3.11 or 3.12 (64-bit)
- Git (configured for Windows)
- PowerShell 5.1+ or PowerShell 7 (Core)

### Step-by-Step Installation
Run the following commands in an elevated PowerShell session:

```powershell
# 1. Clone repository (or work from cloned workspace)
git clone <repository_url>
cd "M5 Retail Demand Intelligence Platform"

# 2. Allow PowerShell script execution for current session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Create virtual environment
python -m venv .venv

# 4. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 5. Upgrade pip, setuptools, and wheel
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

# 6. Install dependencies
# Option A: Editable package install (standard)
.\.venv\Scripts\python.exe -m pip install -e .

# Option B: Development dependencies including pytest and ruff
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

---

## 2. End-to-End Execution Pipeline

To run the full research and evaluation pipeline from raw inputs to verified report artifacts, execute the following commands in exact order.

### Pipeline Stage Schedule

| Stage | Command | Expected Runtime | Key Outputs | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **0. Raw Extraction** | `python scripts/extract_workbook.py` | ~10s | `data/raw/` CSVs | *Requires raw Kaggle zip* |
| **1. Catalog Profiling** | `python scripts/run_eda.py` | ~25s | `reports/eda/` | Full 30,490-series analysis |
| **2. Baseline Tier 1** | `python scripts/run_tier1.py` | ~45s | `reports/tier1/` | Evaluates baselines on 3k series |
| **3. Walk-Forward Backtest** | `python scripts/run_backtest.py` | ~450s (7.5 min) | `reports/backtest_main/` | 3,000 series $\times$ 3 origins $\times$ 10 models |
| **4. Hypothesis Battery** | `python scripts/run_significance.py` | ~20s | `reports/backtest_main/significance/` | Bootstrap CIs, DM, Nemenyi plot |
| **5. Inventory Analysis** | `python scripts/run_inventory_analysis.py` | ~15s | `reports/inventory/` | ABC/XYZ classification, policy comparison (capital released next to annual cost) |
| **6. Sync Documentation** | `python scripts/render_readme_tables.py` | ~2s | `README.md` | Injects verified artifact tables |

---

## 3. How to Regenerate and Verify Every README Number

### Complete Reproduction Workflow
```powershell
# Step 1: Execute walk-forward backtest (generates accuracy_overall.csv & predictions.parquet)
& .venv\Scripts\python.exe scripts/run_backtest.py --sample-series 3000 --n-origins 3

# Step 2: Run significance tests (generates critical difference plot, paired bootstrap, per-series DM)
& .venv\Scripts\python.exe scripts/run_significance.py --backtest-dir reports/backtest_main

# Step 3: Run inventory analysis (generates portfolio_summary.csv & service level tradeoffs)
& .venv\Scripts\python.exe scripts/run_inventory_analysis.py

# Step 4: Render dynamic Markdown tables into README.md
& .venv\Scripts\python.exe scripts/render_readme_tables.py

# Step 5: Verify zero drift between README tables and committed artifacts
& .venv\Scripts\python.exe scripts/render_readme_tables.py --check
```

If `render_readme_tables.py --check` exits with status `0`, the README is perfectly synchronized with the backing artifacts.

---

## 4. Test Suite Execution & Linting

### Running Automated Tests
```powershell
# Run the entire test suite
& .venv\Scripts\pytest.exe

# Run unit tests only
& .venv\Scripts\pytest.exe tests/unit

# Run integration tests only
& .venv\Scripts\pytest.exe tests/integration

# Run specific audit regression test suites
& .venv\Scripts\pytest.exe tests/unit/test_m5_01_scale_semantics.py
& .venv\Scripts\pytest.exe tests/unit/test_m5_03_tier1_parity.py
& .venv\Scripts\pytest.exe tests/unit/test_m5_04_source_of_truth.py
& .venv\Scripts\pytest.exe tests/unit/test_m5_06_statistical_defense.py
```

### Static Analysis & Linter
```powershell
# Run Ruff lint check across src and scripts
& .venv\Scripts\ruff.exe check src scripts

# Automatically fix minor formatting/import order issues
& .venv\Scripts\ruff.exe check --fix src scripts
```

### Dynamic Module Import Validation
Verify that all source modules import cleanly without circular references or missing dependencies:
```powershell
& .venv\Scripts\python.exe -c "
import pkgutil, importlib
import src
for mod in pkgutil.walk_packages(src.__path__, prefix='src.'):
    importlib.import_module(mod.name)
print('All modules import cleanly!')
"
```

---

## 5. Troubleshooting Actual Defects & Platform Errors

### Defect 1: Windows Multiprocessing Crash (`BrokenProcessPool` in `statsforecast`)
- **Symptom**: Running baseline models with `n_jobs > 1` on Windows crashes with `concurrent.futures.process.BrokenProcessPool` or hangs indefinitely.
- **Root Cause**: On Windows, Python uses the `spawn` multiprocessing start method rather than Unix `fork`. Each worker re-imports the main module. Without top-level `if __name__ == '__main__':` guards in child subprocesses, child workers spawn infinite recursive processes. Furthermore, `statsforecast` estimators are already Numba-compiled and vectorized across series in C-speed within a single process.
- **Fix**: In `configs/model_config.yml` and all model invocations, strictly enforce `n_jobs=1`.

### Defect 2: Out-Of-Memory (OOM) on Wide-to-Long Sales Table Melt
- **Symptom**: Calling `df.melt()` on the full M5 sales matrix spikes RAM beyond 8 GB and crashes low-memory developer machines.
- **Root Cause**: The wide sales matrix contains 30,490 series $\times$ 1,913 days $= 58,327,370$ records. Melting all columns into Pandas creates massive intermediate arrays.
- **Fix**: Apply `narrow-before-melt`:
  1. Filter or sample the series in wide format first (`pl.DataFrame.slice()` or `filter()`).
  2. Select only the necessary day columns for the required history window.
  3. Execute `melt()` lazily or within Polars streaming mode (`pl.LazyFrame`).

### Defect 3: Missing Price Data and KeyErrors on California Series
- **Symptom**: Joining sales panels with `sell_prices.csv` generates unexpected nulls or missing index keys during inventory calculations.
- **Root Cause**: In the official M5 dataset, 1,507 series in California stores (`CA_1`, `CA_2`) have no recorded prices in `sell_prices.csv` for initial backtest dates because Walmart only records prices for active store-item weeks.
- **Fix**: In `src/inventory/abc_xyz.py` and `src/inventory/working_capital.py`, unpriced series are explicitly classified as `UNPRICED` and partitioned into a separate monitoring segment. Never impute zero prices (which distorts inventory value to zero) or drop active items.

### Defect 4: PowerShell String Quoting and Semicolon Parsing Errors
- **Symptom**: Running commands like `python -c "import a; import b"` in PowerShell fails with syntax errors or command block parsing failures.
- **Root Cause**: PowerShell treats semicolons (`;`) as pipeline command separators outside string blocks and handles double quotes inside single quotes differently from Bash.
- **Fix**: Place complex Python inspection scripts into temporary files in `.audit/repro/` and invoke them via `& .venv\Scripts\python.exe .audit/repro/<script>.py`.

### Defect 5: File Lock Failures on Git Index (`.git/index.lock`)
- **Symptom**: Git operations fail with `fatal: Unable to create '.git/index.lock': Permission denied` or `File exists`.
- **Root Cause**: A previous background git operation was interrupted, or a virus scanner / file indexer temporarily locked `.git/index.lock`.
- **Fix**: Ensure no background git tasks are running (`manage_task action: 'list'`), then delete the stale lock file:
  ```powershell
  Remove-Item -Force .git/index.lock -ErrorAction SilentlyContinue
  ```
