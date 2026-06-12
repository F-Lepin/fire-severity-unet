#!/bin/bash
# NLHPC / LEFTRARU — entrenamiento binario leave-one-fire-out (folds 0–43)
#SBATCH -J fire_severity_bin
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-user=felipe.lepin@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00
#SBATCH --array=0-43
#SBATCH -o /home/%u/logs/%x_%A_%a.out
#SBATCH -e /home/%u/logs/%x_%A_%a.err

set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID:-0}

source ~/.bashrc
conda activate mb_fuego

mkdir -p ~/logs
cd ~/X_CONGRESO/fire-severity-unet

echo "Host: $(hostname)  Fold: $FOLD  Date: $(date)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python scripts/train.py --config config/leftraru_binary.yaml --fold "$FOLD"

echo "Done fold $FOLD"
