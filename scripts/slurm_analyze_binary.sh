#!/bin/bash
# NLHPC / LEFTRARU — análisis post-entrenamiento (un fold)
# Uso: sbatch --export=FOLD=0 scripts/slurm_analyze_binary.sh
#SBATCH -J fire_severity_bin_an
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-user=felipe.lepin@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

FOLD=${FOLD:-0}
echo "=== START analyze fold=$FOLD job=${SLURM_JOB_ID:-?} $(date) ==="

PROJECT_DIR="${SLURM_SUBMIT_DIR:-${HOME}/X_CONGRESO/fire-severity-unet}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/slurm_env.sh"
activate_mb_fuego

set -eo pipefail

mkdir -p ~/logs

python scripts/analyze_results.py \
  --config config/leftraru_binary.yaml \
  --checkpoint "checkpoints_binary/fold_${FOLD}/best_model.pt" \
  --fold "$FOLD"

echo "=== DONE analyze fold $FOLD $(date) ==="
