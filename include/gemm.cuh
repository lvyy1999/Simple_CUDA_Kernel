#pragma once

#include "gemm_v1.cuh"
#include "gemm_v2.cuh"
#include "gemm_v3.cuh"
#include "gemm_v4.cuh"
#include "gemm_v5.cuh"
#include "gemm_v6.cuh"

#define MAX_KERNEL_VERSION 6

void (*kernel_funcs[])(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) = {
	gemm_v1, gemm_v2, gemm_v3, gemm_v4, gemm_v5, gemm_v6
};
