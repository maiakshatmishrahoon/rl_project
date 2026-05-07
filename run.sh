#!/usr/bin/env bash
# =============================================================================
# run.sh — Folsom Reservoir CMDP (SAC-Lagrangian, v5)
# =============================================================================
# Usage:
#   bash run.sh                          # uses default data path
#   bash run.sh path/to/folsom_inflow_hourly.npy   # custom data path
# =============================================================================

set -euo pipefail

# ── Data path (override via first positional arg) ─────────────────────────────
NPY_PATH="${1:-folsom_inflow_hourly.npy}"

# ── 1. Install dependencies ───────────────────────────────────────────────────
echo "==> Installing dependencies..."
pip install --quiet numpy pandas matplotlib gymnasium torch

# ── 2. Convert notebook to a plain Python script ─────────────────────────────
echo "==> Converting notebook to script..."
pip install --quiet nbconvert

# Patch the hard-coded Kaggle path to use whatever NPY_PATH is set to
jupyter nbconvert --to script ReservoirRL_Final.ipynb --output ReservoirRL_Final 2>/dev/null

# Replace the hard-coded Kaggle dataset path with the supplied path
sed -i "s|NPY_PATH = .*|NPY_PATH = \"${NPY_PATH}\"|" ReservoirRL_Final.py

# ── 3. Run ────────────────────────────────────────────────────────────────────
echo "==> Starting training... (this may take a while)"
python ReservoirRL_Final.py

# ── 4. Summarise outputs ──────────────────────────────────────────────────────
echo ""
echo "==> Done. Outputs:"
for f in sac_lag_folsom_v5.pt cmdp_results_v5.png results_summary_v5.txt; do
    [ -f "$f" ] && echo "    ✓ $f" || echo "    ✗ $f  (not found)"
done
