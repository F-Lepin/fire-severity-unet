# Cargar conda en jobs SLURM sin source ~/.bashrc (evita BASHRCSOURCED en NLHPC)
activate_mb_fuego() {
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
        set +u
        # shellcheck disable=SC1090
        source "${HOME}/.bashrc"
        set -u
    fi
    conda activate mb_fuego
}
