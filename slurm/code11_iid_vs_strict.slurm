#!/bin/bash
#SBATCH --job-name=ORBIT_IID_STRICT
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/code11_iid_vs_strict_%j.out
#SBATCH --error=logs/code11_iid_vs_strict_%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON="python"

cd "$HOME/ORBIT-FMIB"
mkdir -p logs

echo "=== ENVIRONMENT CHECK ==="
"$PYTHON" --version

"$PYTHON" - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on allocated GPU node.")
print("GPU:", torch.cuda.get_device_name(0))
print("Environment check: PASS")
PY

echo "=== STARTING IID-vs-STRICT DIAGNOSTIC ==="

"$PYTHON" \
"03 Code/11_validate_gb1_iid_vs_strict_split.py" \
    --device cuda \
    --resume
