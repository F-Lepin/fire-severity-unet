#!/bin/bash
#SBATCH --job-name=fire-bin
#SBATCH --output=logs/train_binary_%a.out
#SBATCH --error=logs/train_binary_%a.err
#SBATCH --array=0-43
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# Ajustar partición/GPU según LEFTRARU (ver: sinfo, squeue)
#SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID:-0}

source ~/.bashrc
conda activate mb_fuego

cd ~/X_CONGRESO/fire-severity-unet
mkdir -p logs

echo "Host: $(hostname)  Fold: $FOLD  Date: $(date)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

python scripts/train.py --config config/leftraru_binary.yaml --fold "$FOLD"

echo "Done fold $FOLD"
