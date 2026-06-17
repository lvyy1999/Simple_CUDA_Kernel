#include <cuda_runtime.h>

__global__ void reduction_kernel_v1(const float* input, float* output, int N) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // 朴素树形规约
    for(int s = 1; s < blockDim.x; s <<= 1) {
        if(tid % (2 * s) == 0) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) atomicAdd(output, smem[0]);
}

extern "C" void reduce_v1(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    reduction_kernel_v1<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, N);
}
