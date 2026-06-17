#include <cuda_runtime.h>

#include "../include/vector_add_v1.cuh"

__global__ void vector_add_kernel_v1(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void vector_add_v1(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_kernel_v1<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}
