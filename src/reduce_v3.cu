#include <cuda_runtime.h>

__global__ void reduction_kernel_v3(const float* input, float* output, int N) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    // 每个线程加载 2 个元素，减少空闲线程
    smem[tid] = (idx < N) ? input[idx] : 0.0f;
    smem[tid] += (idx + blockDim.x * gridDim.x < N) ? input[idx + blockDim.x * gridDim.x] : 0.0f;
    __syncthreads();
    
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) atomicAdd(output, smem[0]);
}

extern "C" void reduce_v3(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    // 每个线程处理两个元素，block 数量减半
    int blocksPerGrid = (N + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock);
    reduction_kernel_v3<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, N);
}
