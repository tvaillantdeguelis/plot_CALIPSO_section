#!/bin/bash -l

#SBATCH --time=1:00:00
#SBATCH --mem=8G

# Activate Python environment (server or local PC; override with CONDA_SH/CONDA_ENV)
if [[ -z "${CONDA_SH:-}" ]]; then
    if [[ -f /work_users/vaillant/mambaforge/etc/profile.d/conda.sh ]]; then
        CONDA_SH=/work_users/vaillant/mambaforge/etc/profile.d/conda.sh
        CONDA_ENV="${CONDA_ENV:-plot}"
    elif [[ -f "$HOME/python_envs/mambaforge/etc/profile.d/conda.sh" ]]; then
        CONDA_SH="$HOME/python_envs/mambaforge/etc/profile.d/conda.sh"
        CONDA_ENV="${CONDA_ENV:-python3_12_env1}"
    else
        echo "conda.sh not found: set CONDA_SH and CONDA_ENV" >&2
        exit 1
    fi
fi
source "$CONDA_SH"
conda activate "${CONDA_ENV:?CONDA_ENV must name the conda environment}"

# Run program
: "${CONFIG_FILE:?CONFIG_FILE must point to a complete YAML configuration}"
: "${PYTHON_SCRIPT:?PYTHON_SCRIPT must point to plot_calipso_section.py}"

cleanup_config() {
    if [[ "${REMOVE_CONFIG_AFTER_RUN:-0}" == "1" ]]; then
        rm -f -- "$CONFIG_FILE"
    fi
}
trap cleanup_config EXIT

python "$PYTHON_SCRIPT" "$CONFIG_FILE"
