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

将v1中的三个kernel融合为一个kernel，每个block负责一行，中间数据放在共享内存，减少对HBM显存的占用
源码：[attention_v2.cu](../src/attention_v2.cu)

```cpp
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

```

## v3 : 矩阵分块

利用矩阵分块和共享内存增加数据复用，减少全局内存读取
源码：[attention_v3.cu](../src/attention_v3.cu)

```cpp
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define BM 4
#define BN 32
#define MAX_D 128
#define THREADS 256

__global__ void attention_kernel_v3(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    __shared__ float Qs[BM * MAX_D];
    __shared__ float Ks[BN * MAX_D];
    __shared__ float Vs[BN * MAX_D];
    __shared__ float S[BM * BN];
    __shared__ float o[BM * MAX_D];
    __shared__ float m[BM];
    __shared__ float l[BM];

    int rq = blockIdx.x * BM;
    int tid = threadIdx.x;

    // load Q
    for(int i = tid; i < BM * d; i += blockDim.x) {
        int rb = i / d;
        int cb = i % d;
        int r = rq + rb;
        Qs[rb * MAX_D + cb] = r < M ? Q[r * d + cb] : 0.0f;
        o[rb * MAX_D + cb] = 0.0f;
    }

    if (tid < BM) {
        int r = rq + tid;
        m[tid] = r < M ? -FLT_MAX : 0.0f;
        l[tid] = r < M ? 0.0f : 1.0f;
    }
    __syncthreads();

    float scale = rsqrtf(d);
    for(int rkv = 0; rkv < N; rkv += BN) {
        // load K / V
        for(int i = tid; i < BN * d; i += blockDim.x) {
            int rb = i / d;
            int cb = i % d;
            int r = rkv + rb;
            Ks[rb * MAX_D + cb] = r < N ? K[r * d + cb] : 0.0f;
            Vs[rb * MAX_D + cb] = r < N ? V[r * d + cb] : 0.0f;
        }
        __syncthreads();

        // compute S = Q * K^T
        for(int i = tid; i < BM * BN; i += blockDim.x) {
            int rb = i / BN;
            int cb = i % BN;
            int r = rq + rb;
            int c = rkv + cb;
            float sum = 0.0f;
            if(r < M && c < N) {
                for(int j = 0; j < d; j++) {
                    sum += Qs[rb * MAX_D + j] * Ks[cb * MAX_D + j];
                }
            }
            S[rb * BN + cb] = sum * scale;
        }
        __syncthreads();

        int rb = tid / BN;
        int cb = tid % BN;
        if(rb < BM) {
            int r = rq + rb;
            int c = rkv + cb;
            if(r < M && c < N) {
                float m_old = m[rb];
                float l_old = l[rb];

                float m_cur = S[rb * BN + cb];
                #pragma unroll
                for(int offset = 16; offset > 0; offset >>= 1) {
                    m_cur = fmaxf(m_cur, __shfl_down_sync(0xFFFFFFFF, m_cur, offset));
                }
                m_cur = __shfl_sync(0xFFFFFFFF, m_cur, 0);
                float m_new = fmaxf(m_old, m_cur);
                float old_scale = expf(m_old - m_new);
                float l_cur = expf(S[rb * BN + cb] - m_new);
                #pragma unroll
                for(int offset = 16; offset > 0; offset >>= 1) {
                    l_cur += __shfl_down_sync(0xFFFFFFFF, l_cur, offset);
                }
                l_cur = __shfl_sync(0xFFFFFFFF, l_cur, 0);
                float l_new = l_old * old_scale + l_cur;

                for(int j = cb; j < d; j += BN) {
                    float o_old = o[rb * MAX_D + j];
                    float o_cur = 0.0f;
                    for(int k = 0; k < BN && rkv + k < N; k++) {
                        o_cur += expf(S[rb * BN + k] - m_new) * Vs[k * MAX_D + j];
                    }
                    o[rb * MAX_D + j] = o_old * old_scale + o_cur;
                }

                if(cb == 0) {
                    m[rb] = m_new;
                    l[rb] = l_new;
                }
            }
        }
        __syncthreads();
    }

    for(int i = tid; i < BM * d; i += blockDim.x) {
        int rb = i / d;
        int cb = i % d;
        int r = rq + rb;
        if(r < M) {
            output[r * d + cb] = o[rb * MAX_D + cb] / l[rb];
        }
    }
}

extern "C" void attention_v3(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    if(d > MAX_D) {
        printf("only supports d <= %d\n", MAX_D);
        return;
    }
    int threadsPerBlock = THREADS;
    int blocksPerGrid = ((M + BM - 1) / BM);
    attention_kernel_v3<<<blocksPerGrid, threadsPerBlock>>>(Q, K, V, output, M, N, d);
}


```
