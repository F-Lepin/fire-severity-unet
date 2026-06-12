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

set -euo pipefail

FOLD=${FOLD:-0}

source ~/.bashrc
conda activate mb_fuego

mkdir -p ~/logs
cd ~/X_CONGRESO/fire-severity-unet

python scripts/analyze_results.py \
  --config config/leftraru_binary.yaml \
  --checkpoint "checkpoints_binary/fold_${FOLD}/best_model.pt" \
  --fold "$FOLD"

echo "Analyze fold $FOLD done."
