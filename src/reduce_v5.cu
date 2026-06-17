#include <cuda_runtime.h>

__device__ float warp_reduce(float val) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduction_kernel_v5(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float val = (idx < N) ? input[idx] : 0.0f;
    if(idx + blockDim.x * gridDim.x < N) val += input[idx + blockDim.x * gridDim.x];

    // 每个 warp 内进行规约
    val = warp_reduce(val);

    // 每个 warp 内的 lane 0 将结果写入共享内存
    __shared__ float warp_sum[32]; // 最多 32 个 warp
    if(lane_id == 0) warp_sum[warp_id] = val;
    __syncthreads();

    // 最终结果由 warp 0 负责规约
    int num_warps = blockDim.x / 32;
    val = (tid < num_warps) ? warp_sum[tid] : 0.0f;
    if(warp_id == 0) val = warp_reduce(val);

    if(tid == 0) atomicAdd(output, val);
}

extern "C" void reduce_v5(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    reduction_kernel_v5<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}

