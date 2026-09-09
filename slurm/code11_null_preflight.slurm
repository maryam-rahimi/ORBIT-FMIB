#!/bin/bash
#SBATCH --job-name=ORBIT_NULL_PREF
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/code11_null_preflight_%j.out
#SBATCH --error=logs/code11_null_preflight_%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
eval "$(conda shell.bash hook)"
conda activate orbitfmib

cd "$HOME/ORBIT-FMIB"
mkdir -p logs

which python
python --version

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY

python "03 Code/11_validate_gb1_null_controls.py" \
    --preflight-only \
    --device cuda
