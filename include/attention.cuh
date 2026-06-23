#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void attention_v1(const float* Q, const float* K, const float* V, float* output, int M, int N, int d);

void attention_v2(const float* Q, const float* K, const float* V, float* output, int M, int N, int d);

void attention_v3(const float* Q, const float* K, const float* V, float* output, int M, int N, int d);

#ifdef __cplusplus
}
#endif

#define MAX_KERNEL_VERSION 3

void (*kernel_funcs[])(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) = {
	attention_v1, attention_v2, attention_v3
};
