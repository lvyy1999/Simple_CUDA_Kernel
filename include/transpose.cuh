#pragma once

#include "transpose_v1.cuh"
#include "transpose_v2.cuh"

#define MAX_KERNEL_VERSION 2

void (*kernel_funcs[])(const float* input, float* output, int rows, int cols) = {
	transpose_v1, transpose_v2
};


