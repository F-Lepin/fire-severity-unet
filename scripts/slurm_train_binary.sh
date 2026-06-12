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

FOLD=${SLURM_ARRAY_TASK_ID:-0}
echo "=== START job=${SLURM_JOB_ID:-?} fold=$FOLD $(date) host=$(hostname) ==="

PROJECT_DIR="${SLURM_SUBMIT_DIR:-${HOME}/X_CONGRESO/fire-severity-unet}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/slurm_env.sh"
activate_mb_fuego

set -eo pipefail

mkdir -p ~/logs

echo "Python: $(which python)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python scripts/train.py --config config/leftraru_binary.yaml --fold "$FOLD"

echo "=== DONE fold $FOLD $(date) ==="
