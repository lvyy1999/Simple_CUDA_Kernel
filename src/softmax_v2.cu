#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

__device__ float warp_reduce_max(float m) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        m = fmaxf(m, __shfl_down_sync(0xFFFFFFFF, m, offset));
    }
    return m;
}

__device__ float warp_reduce_sum(float s) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xFFFFFFFF, s, offset);
    }
    return s;
}

__global__ void softmax_kernel_v2(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float m = -FLT_MAX;
    for(int i = tid; i < N; i += blockDim.x) {
        m = fmaxf(m, input[i]);
    }

    __shared__ float smem[32];

    m = warp_reduce_max(m);
    if(lane_id == 0) smem[warp_id] = m;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    m = (tid < num_warps) ? smem[tid] : -FLT_MAX;
    if(warp_id == 0) {
        m = warp_reduce_max(m);
        if(lane_id == 0) {
            smem[0] = m;
        }
    }
    __syncthreads();

    m = smem[0];
    __syncthreads();
    float s = 0.0f;
    for(int i = tid; i < N; i += blockDim.x) {
        s += expf(input[i] - m);
    }

    s = warp_reduce_sum(s);
    if(lane_id == 0) smem[warp_id] = s;
    __syncthreads();

    s = (tid < num_warps) ? smem[tid] : 0.0f;
    if(warp_id == 0) {
        s = warp_reduce_sum(s);
        if(lane_id == 0) {
            smem[0] = s;
        }
    }
    __syncthreads();

    s = smem[0];

    for(int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - m) / s;
    }
}

extern "C" void softmax_v2(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    softmax_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
