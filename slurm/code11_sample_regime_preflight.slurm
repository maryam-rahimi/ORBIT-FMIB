#!/bin/bash
#SBATCH --job-name=ORBIT_SAMPLE_PREF
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/code11_sample_regime_preflight_%j.out
#SBATCH --error=logs/code11_sample_regime_preflight_%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
eval "$(conda shell.bash hook)"
conda activate orbitfmib

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$HOME/ORBIT-FMIB"
mkdir -p logs

python "03 Code/11_validate_gb1_fixed_subset_sample_regime.py" \
    --preflight-only \
    --device cuda
