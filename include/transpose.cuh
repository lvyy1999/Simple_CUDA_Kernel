#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void transpose_v1(const float* input, float* output, int rows, int cols);

void transpose_v2(const float* input, float* output, int rows, int cols);

#ifdef __cplusplus
}
#endif
#define MAX_KERNEL_VERSION 2

void (*kernel_funcs[])(const float* input, float* output, int rows, int cols) = {
	transpose_v1, transpose_v2
};


