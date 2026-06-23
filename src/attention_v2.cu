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

__device__ float block_reduce_max(float m) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    m = warp_reduce_max(m);

    __shared__ float smem[32];
    if(lane_id == 0) smem[warp_id] = m;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    m = (tid < num_warps) ? smem[tid] : -FLT_MAX;
    if(warp_id == 0) {
        m = warp_reduce_max(m);
        if(lane_id == 0) smem[0] = m;
    };
    __syncthreads();

    m = smem[0];
    return m;
}

__device__ float block_reduce_sum(float s) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    s = warp_reduce_sum(s);

    __shared__ float smem[32];
    if(lane_id == 0) smem[warp_id] = s;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    s = (tid < num_warps) ? smem[tid] : 0.0f;
    if(warp_id == 0) {
        s = warp_reduce_sum(s);
        if(lane_id == 0) smem[0] = s;
    }
    __syncthreads();

    s = smem[0];
    return s;
}

__global__ void attention_kernel_v2(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    extern __shared__ float scores[];

    int row = blockIdx.x;
    if(row >= M) return;
    int tid = threadIdx.x;

    float m = -FLT_MAX;
    float scale = rsqrtf(d);
    for(int j = tid; j < N; j += blockDim.x) {
        float sum = 0.0f;
        for(int k = 0; k < d; k++) {
            sum += Q[row * d + k] * K[j * d + k];
        }
        sum *= scale;
        scores[j] = sum;
        m = fmaxf(m, sum);
    }
    __syncthreads();

    m = block_reduce_max(m);

    float s = 0.0f;
    for(int j = tid; j < N; j += blockDim.x) {
        s += expf(scores[j] - m);
    }

    s = block_reduce_sum(s);
    
    for(int j = tid; j < d; j += blockDim.x) {
        float sum = 0.0f;
        for(int k = 0; k < N; k++) {
            sum += V[k * d + j] * expf(scores[k] - m) / s;
        }
        output[row * d + j] = sum;
    }
}

extern "C" void attention_v2(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    attention_kernel_v2<<<M, 256, N * sizeof(float)>>>(Q, K, V, output, M, N, d);
}
