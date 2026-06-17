#include <cuda_runtime.h>

__global__ void reduction_kernel_v4(const float* input, float* output, int N) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < N) ? input[idx] : 0.0f;
    smem[tid] += (idx + blockDim.x * gridDim.x < N) ? input[idx + blockDim.x * gridDim.x] : 0.0f;
    __syncthreads();
    
    // 只规约到 s > 32 的部分
    for(int s = blockDim.x / 2; s > 32; s >>= 1) {
        if(tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // 最后一个 warp 用 __shfl_down_sync 进行规约，更高效且无需 __syncthreads
    if(tid < 32) {
        float val = smem[tid] + smem[tid + 32];
        #pragma unroll
        for(int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if(tid == 0) atomicAdd(output, val);
    }
}

extern "C" void reduce_v4(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock);
    reduction_kernel_v4<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, N);
}