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
