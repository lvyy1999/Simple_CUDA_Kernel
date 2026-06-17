#!/bin/bash

ARCH=${1:-sm_75}
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"

# echo "Using architecture: ${ARCH}"

cd "${SCRIPTS_DIR}"

nvcc ../tests/test_transpose.cu ../src/transpose_v*.cu -o "${PROJECT_DIR}/test_transpose" \
    -I"../include" \
    -lcublas \
    -O3 -arch=${ARCH}

"${PROJECT_DIR}/test_transpose"
