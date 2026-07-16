#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef FLOAT4
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#endif

template <int BLOCK_SIZE, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_kernel_v5(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[BK][BM + 4]; // 转置存储As，并添加padding
    __shared__ float Bs[BK][BN];
    
    int tid = threadIdx.x;
    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;

    // 处理 A 时，block 内布局
    constexpr int A_BLOCK_COLS = BK;
    constexpr int A_BLOCK_ROWS = BLOCK_SIZE / A_BLOCK_COLS;
    int ra = tid / A_BLOCK_COLS, ca = tid % A_BLOCK_COLS;

    // 处理 B 时，block 内布局
    constexpr int B_BLOCK_ROWS = BK;
    constexpr int B_BLOCK_COLS = BLOCK_SIZE / B_BLOCK_ROWS;
    int rb = tid / B_BLOCK_COLS, cb = tid % B_BLOCK_COLS;

    // 处理 C 时，block 内布局
    constexpr int C_BLOCK_ROWS = 16;
    constexpr int C_BLOCK_COLS = BLOCK_SIZE / C_BLOCK_ROWS;
    int rc = tid / 16, cc = tid % 16;
    
    // 每个 thread 负责 C 中的 TM * TN 个元素
    float At[TM];
    float Bt[TN];
    float Ct[TM][TN] = {0.0f};

    // 沿着 K 维度遍历
    for(int k0 = 0; k0 < K; k0 += BK) {
        // 读取 A 的一块，并转置存储
        #pragma unroll
        for(int i = ra; i < BM; i += A_BLOCK_ROWS) {
            int r = r0 + i, c = k0 + ca;
            As[ca][i] = (r < M && c < K) ? __half2float(A[r * K + c]) : 0.0f;
        }

        // 读取 B 的一块
        #pragma unroll
        for(int j = cb; j < BN; j += B_BLOCK_COLS) {
            int r = k0 + rb, c = c0 + j;
            Bs[rb][j] = (r < K && c < N) ? __half2float(B[r * N + c]) : 0.0f;
        }

        __syncthreads();

        // 用外积的方式计算，并提前取数据到寄存器，减少共享内存访问
        for(int k = 0; k < BK; k++) {
            #pragma unroll
            for(int i = 0; i < TM; i += 4) { 
                FLOAT4(At[i]) = FLOAT4(As[k][4 * rc + i * C_BLOCK_ROWS]); // 向量化加载共享内存
            }
            #pragma unroll
            for(int j = 0; j < TN; j += 4) {
                FLOAT4(Bt[j]) = FLOAT4(Bs[k][4 * cc + j  * C_BLOCK_COLS]); // 向量化加载共享内存
            }
            // 在寄存器上做外积
            for(int i = 0; i < TM; i++) {
                #pragma unroll
                for(int j = 0; j < TN; j++) {
                    Ct[i][j] += At[i] * Bt[j];
                }
            }
        }

        __syncthreads();
    }

    // write C
    for(int i = 0; i < TM; i++) {
        int r = r0 + 4 * rc + (i / 4) * 4 * C_BLOCK_ROWS + i % 4;
        #pragma unroll
        for(int j = 0; j < TN; j++) {
            int c = c0 + 4 * cc + (j / 4) * 4 * C_BLOCK_COLS + j % 4;
            if(r < M && c < N) {
                float c_val = (beta != 0.0f) ? __half2float(C[r * N + c]) * beta : 0.0f;
                C[r * N + c] = __float2half_rn(Ct[i][j] * alpha + c_val);
            }
        } 
    }
}

extern "C" void gemm_v5(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int threadsPerBlock = 256;
    constexpr int BM = 128, BN = 128, BK = 16, TM = BM / 16, TN = BN / 16;
    dim3 blocksPerGrid((N + BN - 1) / BN,
                       (M + BM - 1) / BM);
    gemm_kernel_v5<256, BM, BN, BK, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
