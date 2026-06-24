#include <cuda_runtime.h>

__global__ void vector_add_kernel_v2(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid Stride Loop + 向量化加载
    const float4* vec_A = reinterpret_cast<const float4*>(A);
    const float4* vec_B = reinterpret_cast<const float4*>(B);
    float4* vec_C = reinterpret_cast<float4*>(C);
    int vec_N = N / 4;
    for(int i = idx; i < vec_N; i += stride) {
        float4 a = vec_A[i];
        float4 b = vec_B[i];
        float4 c = make_float4(
            a.x + b.x, 
            a.y + b.y, 
            a.z + b.z, 
            a.w + b.w
        );
        vec_C[i] = c;
    }

    // 尾部处理
    if(int i = vec_N * 4 + idx; i < N) {
        C[i] = A[i] + B[i];
    }
}

extern "C" void vector_add_v2(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int vec_N = (N + 4 - 1) / 4;
    int blocksPerGrid = fmin((vec_N + threadsPerBlock - 1) / threadsPerBlock, num_sms * 4);
    vector_add_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}
