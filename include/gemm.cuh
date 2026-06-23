#pragma once

#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

void gemm_v1(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta);

void gemm_v2(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta);

void gemm_v3(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta);

void gemm_v4(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta);

void gemm_v5(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta);

void gemm_v6(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta);

#ifdef __cplusplus
}
#endif

#define MAX_KERNEL_VERSION 6

void (*kernel_funcs[])(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) = {
	gemm_v1, gemm_v2, gemm_v3, gemm_v4, gemm_v5, gemm_v6
};
