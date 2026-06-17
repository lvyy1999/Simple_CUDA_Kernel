#!/bin/bash

ARCH=${1:-sm_75}
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"

# echo "Using architecture: ${ARCH}"

cd "${SCRIPTS_DIR}"

nvcc ../tests/test_vector_add.cu ../src/vector_add_v*.cu -o "${PROJECT_DIR}/test_vector_add" \
    -I"../include" \
    -lcublas \
    -O3 -arch=${ARCH}

"${PROJECT_DIR}/test_vector_add"
