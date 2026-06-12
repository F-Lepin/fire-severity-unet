#!/bin/bash
# Subir resultados del experimento binario a GCS (MapBiomas Chile)
# Uso en LEFTRARU:
#   cd ~/X_CONGRESO/fire-severity-unet
#   bash scripts/upload_results_gcs.sh
#   bash scripts/upload_results_gcs.sh binary 0    # solo fold 0
#   bash scripts/upload_results_gcs.sh binary all    # todos los folds con checkpoint

set -e

BUCKET="gs://mapbiomas-chile/image_download/DATA_X_CONGRESO/resultados_v2"
EXPERIMENT="${1:-binary}"
FOLD="${2:-0}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-${HOME}/X_CONGRESO/fire-severity-unet}"
cd "${PROJECT_DIR}"

if ! command -v gsutil &>/dev/null; then
    echo "Error: gsutil no encontrado. Carga el módulo o activa gcloud en LEFTRARU."
    exit 1
fi

DEST="${BUCKET}/fire-severity-unet/${EXPERIMENT}"
echo "Destino: ${DEST}"

upload_fold() {
    local f="$1"
    local dest_fold="${DEST}/fold_${f}"
    echo ""
    echo "=== Subiendo fold ${f} → ${dest_fold} ==="

    if [ -f "checkpoints_${EXPERIMENT}/fold_${f}/best_model.pt" ] || \
       [ -f "checkpoints_binary/fold_${f}/best_model.pt" ]; then
        CKPT_DIR="checkpoints_binary"
        [ "${EXPERIMENT}" != "binary" ] && CKPT_DIR="checkpoints"
        gsutil -m cp -r "${CKPT_DIR}/fold_${f}/" "${dest_fold}/checkpoints/"
    else
        echo "  (sin checkpoint fold_${f}, omitido)"
    fi

    if [ -d "outputs_binary/fold_${f}" ]; then
        gsutil -m cp -r "outputs_binary/fold_${f}/" "${dest_fold}/training_outputs/"
    elif [ -d "outputs/fold_${f}" ]; then
        gsutil -m cp -r "outputs/fold_${f}/" "${dest_fold}/training_outputs/"
    fi

    INTERP="outputs/interpretability_binary/fold_${f}"
    [ "${EXPERIMENT}" != "binary" ] && INTERP="outputs/interpretability/fold_${f}"
    if [ -d "${INTERP}" ]; then
        gsutil -m cp -r "${INTERP}/" "${dest_fold}/interpretability/"
    fi
}

if [ "${EXPERIMENT}" = "binary" ]; then
    CKPT_ROOT="checkpoints_binary"
    INTERP_ROOT="outputs/interpretability_binary"
else
    CKPT_ROOT="checkpoints"
    INTERP_ROOT="outputs/interpretability"
fi

if [ "${FOLD}" = "all" ]; then
    for d in "${CKPT_ROOT}"/fold_*; do
        [ -d "$d" ] || continue
        n="${d##*/fold_}"
        upload_fold "$n"
    done
else
    upload_fold "${FOLD}"
fi

# README con metadatos del experimento
README=$(mktemp)
cat > "${README}" <<EOF
fire-severity-unet resultados
experiment: ${EXPERIMENT}
fold: ${FOLD}
date: $(date -Iseconds)
host: $(hostname)
repo: https://github.com/F-Lepin/fire-severity-unet
config: config/leftraru_${EXPERIMENT}.yaml (binary = 2 clases severidad)
EOF
gsutil cp "${README}" "${DEST}/README_$(date +%Y%m%d_%H%M%S).txt"
rm -f "${README}"

echo ""
echo "Listo. Verifica:"
echo "  gsutil ls ${DEST}/"
