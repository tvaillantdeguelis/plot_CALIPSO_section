#!/bin/bash -l

#SBATCH --time=1:00:00
#SBATCH --mem=4G

# Activer Python virtual environment
source /work_users/vaillant/mambaforge/etc/profile.d/conda.sh
conda activate plot

# Run program
: "${CONFIG_FILE:?CONFIG_FILE must point to a complete YAML configuration}"
: "${PYTHON_SCRIPT:?PYTHON_SCRIPT must point to plot_2D-McDA_masks.py}"

cleanup_config() {
    if [[ "${REMOVE_CONFIG_AFTER_RUN:-0}" == "1" ]]; then
        rm -f -- "$CONFIG_FILE"
    fi
}
trap cleanup_config EXIT

python "$PYTHON_SCRIPT" "$CONFIG_FILE"
