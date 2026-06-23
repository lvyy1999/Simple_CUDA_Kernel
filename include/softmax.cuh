#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void softmax_v1(const float* input, float* output, int N);

void softmax_v2(const float* input, float* output, int N);

void softmax_v3(const float* input, float* output, int N);

void softmax_v4(const float* input, float* output, int N);

#ifdef __cplusplus
}
#endif

#define MAX_KERNEL_VERSION 4

void (*kernel_funcs[])(const float* input, float* output, int N) = {
	softmax_v1, softmax_v2, softmax_v3, softmax_v4
};


