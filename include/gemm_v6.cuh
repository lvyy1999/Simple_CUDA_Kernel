#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

void gemm_v6(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta);

#ifdef __cplusplus
}
#endif
