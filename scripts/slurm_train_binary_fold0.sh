#!/bin/bash
# NLHPC / LEFTRARU — piloto binario: fold 0 + análisis
#SBATCH -J fire_severity_bin_f0
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-user=felipe.lepin@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

echo "=== START job=${SLURM_JOB_ID:-?} $(date) host=$(hostname) ==="

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/slurm_env.sh"
activate_mb_fuego

mkdir -p ~/logs
cd ~/X_CONGRESO/fire-severity-unet

echo "Python: $(which python)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python scripts/train.py --config config/leftraru_binary.yaml --fold 0

python scripts/analyze_results.py \
  --config config/leftraru_binary.yaml \
  --checkpoint checkpoints_binary/fold_0/best_model.pt \
  --fold 0

echo "=== DONE fold 0 $(date) ==="
