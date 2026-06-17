#include <cuda_runtime.h>

__global__ void reduction_kernel_v2(const float* input, float* output, int N) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    
    // 反转步长方向，减少 Warp Divergence，消除 Bank Conflict
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) atomicAdd(output, smem[0]);
}

extern "C" void reduce_v2(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    reduction_kernel_v2<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, N);
}
