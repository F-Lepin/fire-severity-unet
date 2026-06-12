#!/bin/bash
# Piloto: un solo fold (fold 0) — usar si no quieres lanzar el array completo
#SBATCH --job-name=fire-bin-f0
#SBATCH --output=logs/train_binary_fold0.out
#SBATCH --error=logs/train_binary_fold0.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

set -euo pipefail

source ~/.bashrc
conda activate mb_fuego

cd ~/X_CONGRESO/fire-severity-unet
mkdir -p logs

echo "Host: $(hostname)  Fold: 0  Date: $(date)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python scripts/train.py --config config/leftraru_binary.yaml --fold 0

python scripts/analyze_results.py \
  --config config/leftraru_binary.yaml \
  --checkpoint checkpoints_binary/fold_0/best_model.pt \
  --fold 0

echo "Train + analyze fold 0 done."
