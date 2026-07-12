#include <cuda_runtime.h>

__global__ void vector_add_kernel_v2(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // 向量化加载
    const float4* vec_A = reinterpret_cast<const float4*>(A);
    const float4* vec_B = reinterpret_cast<const float4*>(B);
    float4* vec_C = reinterpret_cast<float4*>(C);
    int vec_N = N / 4;
    if(idx < vec_N) {
        float4 a = vec_A[idx];
        float4 b = vec_B[idx];
        float4 c = make_float4(
            a.x + b.x, 
            a.y + b.y, 
            a.z + b.z, 
            a.w + b.w
        );
        vec_C[idx] = c;
    }

    // 尾部处理
    if(int i = 4 * vec_N + idx; i < N) {
        C[i] = A[i] + B[i];
    }
}

extern "C" void vector_add_v2(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + 4 * threadsPerBlock - 1) / (4 * threadsPerBlock);
    vector_add_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}
