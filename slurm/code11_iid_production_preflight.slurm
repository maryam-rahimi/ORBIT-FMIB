#!/bin/bash
#SBATCH --job-name=ORBIT_IID_PREFLIGHT
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/code11_iid_production_preflight_%j.out
#SBATCH --error=logs/code11_iid_production_preflight_%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON="python"
SCRIPT="03 Code/11_run_gb1_subset_conditioned_mi_iid.py"

cd "$HOME/ORBIT-FMIB"
mkdir -p logs

echo "=== CODE 11 IID PRODUCTION PREFLIGHT ==="
sha256sum "$SCRIPT"

"$PYTHON" --version
"$PYTHON" - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable.")
PY

"$PYTHON" "$SCRIPT" \
    --preflight-only \
    --device cuda

echo "=== PREFLIGHT FINISHED ==="
