#!/bin/bash
#SBATCH --job-name=code14_del41L
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=code14_del41L_%j.out
#SBATCH --error=code14_del41L_%j.err

cd "$HOME/ORBIT-FMIB" || exit 1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python \
"03 Code/14_bootstrap_retention_inference.py" \
--delete-one-identity-pilot \
--device auto
