#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void vector_add_v1(const float* A, const float* B, float* C, int N);

void vector_add_v2(const float* A, const float* B, float* C, int N);

#ifdef __cplusplus
}
#endif

#define MAX_KERNEL_VERSION 2

void (*kernel_funcs[])(const float* A, const float* B, float* C, int N) = {
	vector_add_v1, vector_add_v2
};


