#!/bin/bash

#SBATCH --job-name=ORBIT_C11_CAL
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/code11_cal_%j.out
#SBATCH --error=logs/code11_cal_%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
eval "$(conda shell.bash hook)"
conda activate orbitfmib

cd "$HOME/ORBIT-FMIB"

echo "========================================"
echo "CODE 11 CALIBRATION"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================"

which python
python --version

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
x = torch.tensor([1.,2.,3.], device="cuda")
print("GPU KERNEL TEST:", x * 2)
PY

python "03 Code/11_run_gb1_target_mi.py" \
    --device cuda \
    --calibration-only

echo "========================================"
echo "CODE 11 CALIBRATION JOB FINISHED"
echo "Finished: $(date)"
echo "========================================"
