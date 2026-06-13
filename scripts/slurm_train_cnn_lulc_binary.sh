#!/bin/bash
# NLHPC — CNN LULC patch classifier (binary severity)
#SBATCH -J cnn_lulc
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --mail-user=felipe.lepin@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -t 02:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

echo "=== START cnn_lulc $(date) host=$(hostname) ==="

PROJECT_DIR="${SLURM_SUBMIT_DIR:-${HOME}/X_CONGRESO/fire-severity-unet}"
cd "${PROJECT_DIR}"
mkdir -p ~/logs

# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/slurm_env.sh"
activate_mb_fuego

set -eo pipefail

python scripts/train_cnn_lulc_binary.py \
  --config config/cnn_lulc_binary.yaml \
  --generate-patches

echo "=== DONE cnn_lulc $(date) ==="
