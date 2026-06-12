#!/bin/bash
# Pipeline completo — severidad binaria (baja / alta) en LEFTRARU
# Uso: bash scripts/run_leftraru_binary.sh [ingest|patches|train|analyze|all] [fold]

set -eo pipefail

CONFIG="config/leftraru_binary.yaml"
STEP="${1:-all}"
FOLD="${2:-0}"

run_ingest() {
  echo "=== Ingesta (45 cicatrices) ==="
  python scripts/ingest_top45_rasters.py --config "$CONFIG"
}

run_patches() {
  echo "=== Generar patches ==="
  python scripts/generate_patches.py --config "$CONFIG"
}

run_train() {
  echo "=== Entrenar fold $FOLD ==="
  python scripts/train.py --config "$CONFIG" --fold "$FOLD"
}

run_analyze() {
  echo "=== Analizar fold $FOLD ==="
  python scripts/analyze_results.py \
    --config "$CONFIG" \
    --checkpoint "checkpoints_binary/fold_${FOLD}/best_model.pt" \
    --fold "$FOLD"
}

case "$STEP" in
  ingest)  run_ingest ;;
  patches) run_patches ;;
  train)   run_train ;;
  analyze) run_analyze ;;
  all)
    run_ingest
    run_patches
    run_train
    run_analyze
    ;;
  *)
    echo "Uso: $0 [ingest|patches|train|analyze|all] [fold]"
    exit 1
    ;;
esac

echo "Listo."
