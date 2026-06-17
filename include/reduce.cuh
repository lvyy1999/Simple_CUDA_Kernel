#pragma once

#include "reduce_v1.cuh"
#include "reduce_v2.cuh"
#include "reduce_v3.cuh"
#include "reduce_v4.cuh"
#include "reduce_v5.cuh"

#define MAX_KERNEL_VERSION 5

void (*kernel_funcs[])(const float* input, float* output, int N) = {
	reduce_v1, reduce_v2, reduce_v3, reduce_v4, reduce_v5
};