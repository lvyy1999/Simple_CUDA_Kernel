#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

#define BM 128
#define BN 128
#define BK 8

__global__ void gemm_kernel_v3(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    int tid = threadIdx.x;
    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;

    // 处理 A 时，block 内重排为 32 * 8，分四次读 A(128 * 8)
    constexpr int A_COLS = BK; // 8
    constexpr int A_ROWS = BLOCK_SIZE / A_COLS; // 32
    int ra = tid / A_COLS, ca = tid % A_COLS;

    // 处理 B 时，block 内重排为 8 * 32，分四次读 B(8 * 128)
    constexpr int B_ROWS = BK; // 8
    constexpr int B_COLS = BLOCK_SIZE / B_ROWS; // 32
    int rb = tid / B_COLS, cb = tid % B_COLS;

    // 处理 C 时，block 内重排为 16 * 16
    constexpr int C_COLS = 16;
    constexpr int C_ROWS = 16;
    int rc = tid / C_COLS, cc = tid % C_COLS;
    
    // 每个 thread 负责 C 中的 TM * TN 个元素
    constexpr int TM = BM / C_ROWS; // 8
    constexpr int TN = BN / C_COLS; // 8
    
    float At[TM];
    float Bt[TN];
    float Ct[TM][TN] = {0.0f};

    // 沿着 K 维度遍历
    for(int k0 = 0; k0 < K; k0 += BK) {
        // 读取 A 的一块
        for(int i = ra; i < BM; i += A_ROWS) {
            int r = r0 + i, c = k0 + ca;
            As[i][ca] = (r < M && c < K) ? __half2float(A[r * K + c]) : 0.0f;
        }

        // 读取 B 的一块
        for(int j = cb; j < BN; j += B_COLS) {
            int r = k0 + rb, c = c0 + j;
            Bs[rb][j] = (r < K && c < N) ? __half2float(B[r * N + c]) : 0.0f;
        }

        __syncthreads();

        // 用外积的写法替代内积，并提前取数据到寄存器，减少共享内存访问
        for(int k = 0; k < BK; k++) {
            for(int i = 0; i < TM; i++) {
                At[i] = As[rc + i * C_ROWS][k];
            }
            for(int j = 0; j < TN; j++) {
                Bt[j] = Bs[k][cc + j * C_COLS];
            }
            // 在寄存器上做外积
            for(int i = 0; i < TM; i++) {
                for(int j = 0; j < TN; j++) {
                    Ct[i][j] += At[i] * Bt[j];
                }
            }
        }

        __syncthreads();
    }

    // write C
    for(int i = 0; i < TM; i++) {
        int r = r0 + i * C_ROWS + rc;
        for(int j = 0; j < TN; j++) {
            int c = c0 + j * C_COLS + cc;
            if(r < M && c < N) {
                float c_val = (beta != 0.0f) ? __half2float(C[r * N + c]) * beta : 0.0f;
                C[r * N + c] = __float2half_rn(Ct[i][j] * alpha + c_val);
            }
        } 
    }
}

extern "C" void gemm_v3(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int threadsPerBlock = BLOCK_SIZE;
    dim3 blocksPerGrid((N + BN - 1) / BN,
                       (M + BM - 1) / BM);
    gemm_kernel_v3<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
