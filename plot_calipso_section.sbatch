#!/bin/bash -l

#SBATCH --time=1:00:00
#SBATCH --mem=4G

# Activer Python virtual environment
source ~/.zshrc
mamba activate plot_CALIPSO_section_v1.0

# Run program
./plot_calipso_section.py "$GRANULE_DATE" "$SLICE_START_END_TYPE" "$SLICE_START" "$SLICE_END" "$CASE_STUDY_NAME"