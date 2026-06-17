#pragma once

#include "vector_add_v1.cuh"
#include "vector_add_v2.cuh"

#define MAX_KERNEL_VERSION 2

void (*kernel_funcs[])(const float* A, const float* B, float* C, int N) = {
	vector_add_v1, vector_add_v2
};


