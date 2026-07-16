#!/bin/bash

# Finde alle passenden Dateien
files=$(ls Cube_visuSurf_*.vtu 2>/dev/null)

# Prüfe ob Dateien vorhanden sind
if [ -z "$files" ]; then
    echo "Keine Dateien vom Typ Cube_visuSurf_{Zahl}.vtu gefunden."
    exit 1
fi

# Sortiere Dateien nach eingebetteter Zahl und benenne sie um
i=1
for file in $(ls Cube_visuSurf_*.vtu | sed -E 's/[^0-9]*([0-9]+).*/\1 \0/' | sort -n | cut -d' ' -f2); do
    newname="output${i}.vtu"
    echo "Benenne $file → $newname"
    mv "$file" "$newname"
    ((i++))
done
