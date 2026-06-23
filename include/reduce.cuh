#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void reduce_v1(const float* input, float* output, int N);

void reduce_v2(const float* input, float* output, int N);

void reduce_v3(const float* input, float* output, int N);

void reduce_v4(const float* input, float* output, int N);

void reduce_v5(const float* input, float* output, int N);

void reduce_v6(const float* input, float* output, int N);

#ifdef __cplusplus
}
#endif

#define MAX_KERNEL_VERSION 6

void (*kernel_funcs[])(const float* input, float* output, int N) = {
	reduce_v1, reduce_v2, reduce_v3, reduce_v4, reduce_v5, reduce_v6
};