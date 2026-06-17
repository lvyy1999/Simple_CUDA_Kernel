#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

__device__ void warp_reduce(float& m, float& s) {
    for(int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xFFFFFFFF, m, offset);
        float s2 = __shfl_down_sync(0xFFFFFFFF, s, offset);
        float m_new = fmaxf(m, m2);
        s = s * expf(m - m_new) + s2 * expf(m2 - m_new);
        m = m_new;
    }
}

__global__ void softmax_kernel_v4(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float s = 0.0f;
    float m = -FLT_MAX;
    for(int i = tid; i < N; i += blockDim.x) {
        float x = input[i];
        float m_new = fmaxf(m, x);
        s = s * expf(m - m_new) + expf(x - m_new);
        m = m_new;
    }
    __syncthreads();

    warp_reduce(m, s);

    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];
    if(lane_id == 0) {
        warp_max[warp_id] = m;
        warp_sum[warp_id] = s;
    }
    __syncthreads();
    
    int num_warps = blockDim.x / 32;
    m = (tid < num_warps) ? warp_max[tid] : -FLT_MAX;
    s = (tid < num_warps) ? warp_sum[tid] : 0.0f;
    if(warp_id == 0) {
        warp_reduce(m, s);
        if(lane_id == 0) {
            warp_max[0] = m;
            warp_sum[0] = s;
        }
    };
    __syncthreads();

    m = warp_max[0];
    s = warp_sum[0];
    for(int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - m) / s;
    }
}

extern "C" void softmax_v4(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    softmax_kernel_v4<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
