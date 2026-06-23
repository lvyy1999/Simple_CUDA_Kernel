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

__global__ void attention_kernel_v2(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    extern __shared__ float scores[];

    int row = blockIdx.x;
    if(row >= M) return;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float sumExp = 0.0f;
    float maxVal = -FLT_MAX;
    float scale = rsqrtf(d);
    for(int j = tid; j < N; j += blockDim.x) {
        float sum = 0.0f;
        for(int k = 0; k < d; k++) {
            sum += Q[row * d + k] * K[j * d + k];
        }
        float score = sum * scale;
        scores[j] = score;
        float m_new = fmaxf(maxVal, score);
        sumExp = sumExp * expf(maxVal - m_new) + expf(score - m_new);
        maxVal = m_new;
    }
    __syncthreads();

    warp_reduce(maxVal, sumExp);

    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];
    if(lane_id == 0) {
        warp_max[warp_id] = maxVal;
        warp_sum[warp_id] = sumExp;
    }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    maxVal = (tid < num_warps) ? warp_max[tid] : -FLT_MAX;
    sumExp = (tid < num_warps) ? warp_sum[tid] : 0.0f;
    if(warp_id == 0) {
        warp_reduce(maxVal, sumExp);
        if(lane_id == 0) {
            warp_max[0] = maxVal;
            warp_sum[0] = sumExp;
        }
    };
    __syncthreads();

    maxVal = warp_max[0];
    sumExp = warp_sum[0];
    for(int j = tid; j < d; j += blockDim.x) {
        float sum = 0.0f;
        for(int k = 0; k < N; k++) {
            sum += V[k * d + j] * expf(scores[k] - maxVal) / sumExp;
        }
        output[row * d + j] = sum;
    }
}

extern "C" void attention_v2(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    attention_kernel_v2<<<M, 256, N * sizeof(float)>>>(Q, K, V, output, M, N, d);
}
