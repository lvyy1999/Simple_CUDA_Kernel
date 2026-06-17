#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void gemm_kernel_v2(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for(int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? __half2float(A[row * K + aCol]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? __half2float(B[bRow * N + col]) : 0.0f;
        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < M && col < N) {
        float c_val = (beta != 0.0f) ? __half2float(C[row * N + col]) * beta : 0.0f;
        C[row * N + col] = __float2half_rn(sum * alpha + c_val);
    }
}

extern "C" void gemm_v2(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
