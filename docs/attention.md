## Attention

### v1 : naive版

朴素实现，分三步完成计算，第一步计算 S = QK^T/d ，第二步计算 P = Softmax(S) ， 第三步计算 O = PV ，分别实现三步中的算子，依次调用
源码：[attention_v1.cu](../src/attention_v1.cu)

```cpp
#include <cuda_runtime.h>
#include <float.h>

__global__ void qkt_kernel(const float* Q, const float* K, float* S, int M, int N, int d) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(row >= M || col >= N) return;

    float sum = 0.0f;
    for(int k = 0; k < d; k++) {
        sum += Q[row * d + k] * K[col * d + k];
    }
    S[row * N + col] = sum / sqrtf(d);
}

__global__ void softmax_kernel(const float* S, float* P, int M, int N) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    if(row >= M) return;
    
    int tid = threadIdx.x;

    float maxVal = -FLT_MAX;
    for(int j = tid; j < N; j += blockDim.x) {
        maxVal = fmaxf(maxVal, S[row * N + j]);
    }
    smem[tid] = maxVal;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    maxVal = smem[0];
    __syncthreads();

    float sumExp = 0.0f;
    for(int j = tid; j < N; j += blockDim.x) {
        sumExp += expf(S[row * N + j] - maxVal);
    }
    smem[tid] = sumExp;
    __syncthreads();
    
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    sumExp = smem[0];
    __syncthreads();

    for(int j = tid; j < N; j += blockDim.x) {
        P[row * N + j] = expf(S[row * N + j] - maxVal) / sumExp;
    }
}

__global__ void pv_kernel(const float* P, const float* V, float* output, int M, int N, int d) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(row >= M || col >= d) return;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += P[row * N + k] * V[k * d + col];
    }
    output[row * d + col] = sum;
}

extern "C" void attention_v1(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *S, *P;
    cudaMalloc(&S, M * N * sizeof(float));
    cudaMalloc(&P, M * N * sizeof(float));

    dim3 block_qkt(16, 16);
    dim3 grid_qkt((N + block_qkt.x - 1) / block_qkt.x,
                  (M + block_qkt.y - 1) / block_qkt.y);
    qkt_kernel<<<grid_qkt, block_qkt>>>(Q, K, S, M, N, d);
    cudaDeviceSynchronize();

    softmax_kernel<<<M, 256, 256 * sizeof(float)>>>(S, P, M, N);
    cudaDeviceSynchronize();

    dim3 block_pv(16, 16);
    dim3 grid_pv((d + block_pv.x - 1) / block_pv.x,
                 (M + block_pv.y - 1) / block_pv.y);
    pv_kernel<<<grid_pv, block_pv>>>(P, V, output, M, N, d);
    cudaDeviceSynchronize();

    cudaFree(S);
    cudaFree(P);
}

```

## v2 : kernel 融合

将v1中的三个kernel融合为一个kernel，每个block负责Q的一行，中间数据放在共享内存，减少对HBM显存的N^2占用；softmax部分采用online softmax + warp规约
源码：[attention_v2.cu](../src/attention_v2.cu)

```cpp
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
    attention_kernel_v2<<<M, 256, N * sizeof(float)>>>(S, P, M, N);
}

```

## v3 : 矩阵分块 （未实现）
