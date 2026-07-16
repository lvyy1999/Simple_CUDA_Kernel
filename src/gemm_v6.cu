#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

union __align__(8) Half4 {
    uint2 packed;
    half values[4];
};
static_assert(sizeof(Half4) == 4 * sizeof(half), "Half4 must be 8 bytes");

__device__ __forceinline__ Half4 load_half4(const half* ptr) {
    Half4 value;
    value.packed = *reinterpret_cast<const uint2*>(ptr);
    return value;
}

template <int BLOCK_SIZE, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_kernel_v6(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ half As[BK][BM + 4]; // 转置存储As，并添加padding
    __shared__ half Bs[BK][BN];

    int tid = threadIdx.x;
    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;

    constexpr int HALF4_WIDTH = 4;
    static_assert(BK % HALF4_WIDTH == 0, "BK must be divisible by 4");
    static_assert(BN % HALF4_WIDTH == 0, "BN must be divisible by 4");

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
        // 每次从 global memory 读取连续 4 个 half，A 转置写入 shared memory
        #pragma unroll
        for(int vec = tid; vec < BM * BK / HALF4_WIDTH; vec += BLOCK_SIZE) {
            int linear = vec * HALF4_WIDTH;
            int row = linear / BK;
            int col = linear % BK;
            Half4 value = load_half4(&A[(r0 + row) * K + k0 + col]);

            As[col][row] = value.values[0];
            As[col + 1][row] = value.values[1];
            As[col + 2][row] = value.values[2];
            As[col + 3][row] = value.values[3];
        }

        // 每次从 global memory 读取连续 4 个 half，B 按行写入 shared memory
        #pragma unroll
        for(int vec = tid; vec < BK * BN / HALF4_WIDTH; vec += BLOCK_SIZE) {
            int linear = vec * HALF4_WIDTH;
            int row = linear / BN;
            int col = linear % BN;
            Half4 value = load_half4(&B[(k0 + row) * N + c0 + col]);

            Bs[row][col] = value.values[0];
            Bs[row][col + 1] = value.values[1];
            Bs[row][col + 2] = value.values[2];
            Bs[row][col + 3] = value.values[3];
        }

        __syncthreads();

        // 用外积的方式计算，并提前取数据到寄存器，减少共享内存访问
        for(int k = 0; k < BK; k++) {
            #pragma unroll
            for(int i = 0; i < TM; i++) {
                At[i] = __half2float(As[k][rc + i * C_BLOCK_ROWS]);
            }
            #pragma unroll
            for(int j = 0; j < TN; j++) {
                Bt[j] = __half2float(Bs[k][cc + j * C_BLOCK_COLS]);
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
        int r = r0 + i * C_BLOCK_ROWS + rc;
        #pragma unroll
        for(int j = 0; j < TN; j++) {
            int c = c0 + j * C_BLOCK_COLS + cc;
            // if(r < M && c < N) {
                float c_val = (beta != 0.0f) ? __half2float(C[r * N + c]) * beta : 0.0f;
                C[r * N + c] = __float2half_rn(Ct[i][j] * alpha + c_val);
            // }
        }
    }
}

extern "C" void gemm_v6(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int BM = 128, BN = 128, BK = 32, TM = BM / 16, TN = BN / 16;
    if(M % BM || N % BN || K % BK) {
        printf("only supports M %% %d == 0 and N %% %d == 0 and K %% %d == 0\n", BM, BN, BK);
        return;
    }

    int threadsPerBlock = 256;
    dim3 blocksPerGrid((N + BN - 1) / BN,
                       (M + BM - 1) / BM);
    gemm_kernel_v6<256, BM, BN, BK, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
