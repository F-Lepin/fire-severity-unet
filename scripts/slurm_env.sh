# Cargar conda en jobs SLURM (sin nounset: evita errores en hooks de geotiff/etc.)
activate_mb_fuego() {
    set +u
    if [ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/.conda/etc/profile.d/conda.sh"
    elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    else
        # shellcheck disable=SC1090
        source "${HOME}/.bashrc"
    fi
    conda activate mb_fuego
    # Prefer conda libstdc++ over older system /lib64/libstdc++.so.6 (scipy/sklearn).
    if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}
