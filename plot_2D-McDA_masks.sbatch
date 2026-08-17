#!/bin/bash -l

#SBATCH --time=1:00:00
#SBATCH --mem=4G

# Activer Python virtual environment
source /work_users/vaillant/mambaforge/etc/profile.d/conda.sh
conda activate plot

# Run program
./plot_2D-McDA_masks.py "$GRANULE_DATE" "$SLICE_START_END_TYPE" "$SLICE_START" "$SLICE_END" "$CASE_STUDY_NAME"