#include <math.h>
#include <float.h>
#include <stdio.h>
#include <cuda_runtime.h>

template <int BLOCK_SIZE, int BM, int BN, int BD>
__global__ void attention_kernel_v4(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    __shared__ float Qs[BM][BD];
    __shared__ float Ks[BN][BD + 1];
    __shared__ float Vs[BN][BD];
    __shared__ float P[BM][BN];

    int tid = threadIdx.x;
    int warp_id = tid / BN;
    int lane_id = tid % BN;
    int rq = blockIdx.x * BM + warp_id;

    // 每个 warp 负责一行 BD=128 个输出，每个 thread 负责 4 个输出
    float l_i = 0.0f;
    float m_i = -FLT_MAX;
    float acc[4] = {0.0f};

    // 每个 warp 负责读取 Q 的一行 BD=128 个元素，每个 thread 恰好读取 4 个元素，用 float4 加速
    int cq = 4 * lane_id;
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if(rq < M) {
        value = *reinterpret_cast<const float4*>(&Q[rq * d + cq]);
    }
    Qs[warp_id][cq] = value.x;
    Qs[warp_id][cq + 1] = value.y;
    Qs[warp_id][cq + 2] = value.z;
    Qs[warp_id][cq + 3] = value.w;
    __syncthreads();

    float scale = rsqrtf(d);
    for(int rkv = 0; rkv < N; rkv += BN) {
        // 每次加载 BN 行 K/V 并计算 attention，每个 warp 负责加载 BN / 8 = 4 行
        int col = 4 * lane_id;
        #pragma unroll
        for(int row = warp_id; row < BN; row += 8) {
            int global_row = rkv + row;
            float4 k_value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 v_value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if(global_row < N) {
                k_value = *reinterpret_cast<const float4*>(&K[global_row * d + col]);
                v_value = *reinterpret_cast<const float4*>(&V[global_row * d + col]);
            }

            Ks[row][col] = k_value.x;
            Ks[row][col + 1] = k_value.y;
            Ks[row][col + 2] = k_value.z;
            Ks[row][col + 3] = k_value.w;
            Vs[row][col] = v_value.x;
            Vs[row][col + 1] = v_value.y;
            Vs[row][col + 2] = v_value.z;
            Vs[row][col + 3] = v_value.w;
        }
        __syncthreads();

        if(rq < M) {
            // 每个 block 负责 S=QK^T/sqrt(d) 中的 BM * BN 个元素，恰好每个 thread 负责一个
            int cs = rkv + lane_id;
            float score = 0.0f;
            if(cs < N) {
                #pragma unroll 4
                for(int col = 0; col < BD; col++) {
                    score += Qs[warp_id][col] * Ks[lane_id][col];
                }
                score *= scale;
            } else {
                score = -FLT_MAX;
            }

            // online softmax
            float m = score;
            #pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1) {
                m = fmaxf(m, __shfl_down_sync(0xFFFFFFFF, m, offset));
            }
            m = __shfl_sync(0xFFFFFFFF, m, 0);

            float m_new = fmaxf(m_i, m);
            float alpha = lane_id == 0 ? expf(m_i - m_new) : 0.0f;
            alpha = __shfl_sync(0xFFFFFFFF, alpha, 0);
            float p = cs < N ? expf(score - m_new) : 0.0f;

            float l = p;
            #pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1) {
                l += __shfl_down_sync(0xFFFFFFFF, l, offset);
            }
            l = __shfl_sync(0xFFFFFFFF, l, 0);
            float l_new = l_i * alpha + l;

            P[warp_id][lane_id] = p;
            __syncwarp(0xFFFFFFFF);

            // 计算 O = P * V
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                int col = i * 32 + lane_id; 
                float sum = 0.0f;
                #pragma unroll
                for(int k = 0; k < BN; k++) {
                    sum += P[warp_id][k] * Vs[k][col];
                }
                acc[i] = acc[i] * alpha + sum;
            }

            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    if(rq < M) {
        #pragma unroll
        for(int col = lane_id; col < d; col += 32) {
            output[rq * d + col] = acc[col / 32] / l_i;
        }
    }
}

extern "C" void attention_v4(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    // 每个 block 负责 BM 行，每个 warp 负责一行
    constexpr int BM = 8, BN = 32, BD = 128, BLOCK_SIZE = BM * BN;
    if(d != BD) {
        printf("only supports d == %d\n", BD);
        return;
    }

    int blocksPerGrid = (M + BM - 1) / BM;
    attention_kernel_v4<BLOCK_SIZE, BM, BN, BD><<<blocksPerGrid, BLOCK_SIZE>>>(Q, K, V, output, M, N, d);
}
