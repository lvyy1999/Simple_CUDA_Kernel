#include <cuda_runtime.h>

#include "../include/vector_add_v2.cuh"

__global__ void vector_add_kernel_v2(const float* A, const float* B, float* C, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid * 4;
    if (idx + 3 < N) {
        float4 a = reinterpret_cast<const float4*>(A)[tid];
        float4 b = reinterpret_cast<const float4*>(B)[tid];
        float4 c = make_float4(
            a.x + b.x, 
            a.y + b.y, 
            a.z + b.z, 
            a.w + b.w
        );
        reinterpret_cast<float4*>(C)[tid] = c;
    } else if (idx < N) {
        for(int i = idx; i < N; i++) {
            C[i] = A[i] + B[i];
        }
    }
}

extern "C" void vector_add_v2(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + 4 * threadsPerBlock - 1) / (4 * threadsPerBlock);
    vector_add_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}
