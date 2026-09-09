#!/bin/bash
#SBATCH --job-name=ORBIT_IID_PROD
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/code11_iid_production_%j.out
#SBATCH --error=logs/code11_iid_production_%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON="python"
SCRIPT="03 Code/11_run_gb1_subset_conditioned_mi_iid.py"
OUTDIR="04 Experiments/EXP02_GB1_ESM2/IID_Production_20260910"

cd "$HOME/ORBIT-FMIB"
mkdir -p logs
mkdir -p "$OUTDIR"

echo "=== CODE 11 CORRECTED IID PRODUCTION ==="
echo "Production source:"
sha256sum "$SCRIPT"
echo "Output directory: $OUTDIR"

"$PYTHON" --version
"$PYTHON" - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable.")
print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "=== PREFLIGHT ==="
"$PYTHON" "$SCRIPT" \
    --preflight-only \
    --device cuda \
    --output-dir "$OUTDIR"

echo
echo "=== PRODUCTION ==="
"$PYTHON" "$SCRIPT" \
    --device cuda \
    --output-dir "$OUTDIR" \
    --resume

echo
echo "=== CODE 11 IID PRODUCTION FINISHED ==="
