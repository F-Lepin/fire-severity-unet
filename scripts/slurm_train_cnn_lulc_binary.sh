#!/bin/bash
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

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p ~/logs

# shellcheck disable=SC1091
source scripts/slurm_env.sh
activate_mb_fuego
set -e

echo "Job started: $(date) on $(hostname)"
python scripts/train_cnn_lulc_binary.py \
  --config config/cnn_lulc_binary.yaml \
  --generate-patches
echo "Job finished: $(date)"
