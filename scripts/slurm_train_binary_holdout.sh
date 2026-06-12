#!/bin/bash
# NLHPC — entrenamiento binario con holdout 70/30 por cicatriz
#SBATCH -J fire_bin_7030
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-user=felipe.lepin@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

echo "=== START holdout 70/30 $(date) host=$(hostname) ==="

PROJECT_DIR="${SLURM_SUBMIT_DIR:-${HOME}/X_CONGRESO/fire-severity-unet}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/slurm_env.sh"
activate_mb_fuego

set -eo pipefail

python scripts/train.py --config config/leftraru_binary.yaml --fold 0

python scripts/analyze_results.py \
  --config config/leftraru_binary.yaml \
  --checkpoint checkpoints_binary/holdout_7030/best_model.pt \
  --fold 0

echo "=== DONE holdout 70/30 $(date) ==="
