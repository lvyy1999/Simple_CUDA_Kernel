#include <math.h>
#include <float.h>
#include <stdio.h>
#include <cuda_runtime.h>

template <int BLOCK_SIZE, int BM, int BN, int BD>
__global__ void attention_kernel_v3(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    __shared__ float Qs[BM * BD];
    __shared__ float Ks[BN * BD];
    __shared__ float Vs[BN * BD];
    __shared__ float S[BM * BN];
    __shared__ float o[BM * BD];
    __shared__ float m[BM];
    __shared__ float l[BM];

    int rq = blockIdx.x * BM;
    int tid = threadIdx.x;

    // load Q
    for(int i = tid; i < BM * d; i += blockDim.x) {
        int rb = i / d;
        int cb = i % d;
        int r = rq + rb;
        Qs[rb * BD + cb] = r < M ? Q[r * d + cb] : 0.0f;
        o[rb * BD + cb] = 0.0f;
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
            Ks[rb * BD + cb] = r < N ? K[r * d + cb] : 0.0f;
            Vs[rb * BD + cb] = r < N ? V[r * d + cb] : 0.0f;
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
                    sum += Qs[rb * BD + j] * Ks[cb * BD + j];
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
                float alpha = cb == 0 ? expf(m_i - m_new) : 0.0f;
                alpha = __shfl_sync(0xFFFFFFFF, alpha, 0);
                float l_cur = expf(S[rb * BN + cb] - m_new);
                #pragma unroll
                for(int offset = 16; offset > 0; offset >>= 1) {
                    l_cur += __shfl_down_sync(0xFFFFFFFF, l_cur, offset);
                }
                l_cur = __shfl_sync(0xFFFFFFFF, l_cur, 0);
                float l_new = l_old * alpha + l_cur;

                for(int j = cb; j < d; j += BN) {
                    float o_old = o[rb * BD + j];
                    float o_cur = 0.0f;
                    for(int k = 0; k < BN && rkv + k < N; k++) {
                        o_cur += expf(S[rb * BN + k] - m_new) * Vs[k * BD + j];
                    }
                    o[rb * BD + j] = o_old * alpha + o_cur;
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
            output[r * d + cb] = o[rb * BD + cb] / l[rb];
        }
    }
}

extern "C" void attention_v3(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    constexpr int BM = 8, BN = 32, BD = 128;
    if(d > BD) {
        printf("only supports d <= %d\n", BD);
        return;
    }
    int threadsPerBlock = 256;
    int blocksPerGrid = ((M + BM - 1) / BM);
    attention_kernel_v3<256, BM, BN, BD><<<blocksPerGrid, threadsPerBlock>>>(Q, K, V, output, M, N, d);
}
