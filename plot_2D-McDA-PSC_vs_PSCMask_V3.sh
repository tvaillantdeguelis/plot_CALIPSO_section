#!/bin/bash

SCRIPT="plot_2D-McDA-PSC_vs_PSCMask_V3.py"

DATA_ROOT="/home/vaillant/codes/projects/2D_McDA_PSC/out/data/2D_McDA_PSC.v2.5.0"

START="2008-06-01"
END="2008-06-30"

find "$DATA_ROOT" -type f -name "*.nc" | sort | while read file; do

    # extrait date du fichier (format ISO dans le nom)
    date_str=$(echo "$file" | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z' | head -n 1)

    if [ -z "$date_str" ]; then
        continue
    fi

    # conversion en format comparable YYYY-MM-DD
    file_date=$(echo "$date_str" | cut -d'T' -f1)

    # comparaison de dates (lexicographique OK en ISO)
    if [[ "$file_date" > "$START" && "$file_date" < "$END" ]] || \
       [[ "$file_date" == "$START" || "$file_date" == "$END" ]]; then

        echo "Processing: $file_date -> $file"

        python "$SCRIPT" --input-file "$file"
    fi

done