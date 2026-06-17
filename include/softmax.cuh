#pragma once

#include "softmax_v1.cuh"
#include "softmax_v2.cuh"
#include "softmax_v3.cuh"
#include "softmax_v4.cuh"

#define MAX_KERNEL_VERSION 4

void (*kernel_funcs[])(const float* input, float* output, int N) = {
	softmax_v1, softmax_v2, softmax_v3, softmax_v4
};


