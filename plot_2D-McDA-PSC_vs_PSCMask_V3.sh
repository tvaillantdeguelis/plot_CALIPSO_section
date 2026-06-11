#!/bin/bash

SCRIPT="plot_2D-McDA-PSC_vs_PSCMask_V3.py"
VERSION_2D_McDA="V2.6.1"
DATA_ROOT="/home/vaillant/codes/projects/2D_McDA_PSC/out/data/2D_McDA_PSC.${VERSION_2D_McDA,,}"

START="2009-06-01"
END="2009-09-01"

find "$DATA_ROOT" -type f -name "*.nc" | sort | while read file; do

    # extraction GRANULE_DATE complète (ISO + heure)
    if [[ $file =~ ([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}ZN) ]]; then
        GRANULE_DATE="${BASH_REMATCH[1]}"

        # extrait juste la date YYYY-MM-DD pour filtrer
        file_date="${GRANULE_DATE:0:10}"

        if [[ "$file_date" < "$START" || "$file_date" > "$END" ]]; then
            continue
        fi

        echo "Processing GRANULE_DATE: $GRANULE_DATE"

        python "$SCRIPT" "$GRANULE_DATE" "$VERSION_2D_McDA"
    fi

done