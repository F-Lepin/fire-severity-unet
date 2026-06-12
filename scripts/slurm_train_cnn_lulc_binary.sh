#!/bin/bash
#SBATCH --job-name=cnn_lulc
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felipe.lepin@ug.uchile.cl
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -f scripts/slurm_env.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/slurm_env.sh
fi

python scripts/train_cnn_lulc_binary.py \
  --config config/cnn_lulc_binary.yaml \
  --generate-patches
