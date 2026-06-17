#!/bin/bash

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"

cd "${SCRIPTS_DIR}"

chmod +x get_hardware_info.sh
./get_hardware_info.sh

self=$(basename "$0")
for f in test_*.sh; do
    [ -f "$f" ] || continue
    [ "$f" = "$self" ] && continue
    sed -i 's/\r$//' "$f"
    chmod +x "./$f"
    echo "run: $f"
    "./$f"
    echo ""
done
