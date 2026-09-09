#!/bin/bash

#SBATCH --job-name=ORBIT_ESM2_09
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/code09_%j.out
#SBATCH --error=logs/code09_%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
eval "$(conda shell.bash hook)"
conda activate orbitfmib

cd "$HOME/ORBIT-FMIB"

echo "========================================"
echo "ORBIT-FMIB Code 09"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================"

echo "GPU information:"
nvidia-smi

echo "Python:"
which python
python --version

echo "Checking CUDA:"
python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("No usable CUDA GPU was detected.")
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Starting Code 09..."

python "03 Code/09_extract_esm2_representations.py" \
    --device cuda \
    --batch-size 16 \
    --checkpoint-every 10

echo "========================================"
echo "CODE 09 JOB FINISHED"
echo "Finished: $(date)"
echo "========================================"
