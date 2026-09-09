#!/bin/bash
#SBATCH --job-name=ORBIT_C11_PRE
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/code11_preflight_%j.out
#SBATCH --error=logs/code11_preflight_%j.err

set -euo pipefail
module load anaconda/anaconda-2024.10
eval "$(conda shell.bash hook)"
conda activate orbitfmib
cd "$HOME/ORBIT-FMIB"

python "03 Code/11_run_gb1_target_mi.py" --preflight-only
