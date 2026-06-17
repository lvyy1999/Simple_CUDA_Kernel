#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matrix_transpose_kernel_v2(const float* input, float* output, int rows, int cols) {
    __shared__ float smem[TILE_SIZE][TILE_SIZE + 1];

    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    if (col < cols && row < rows) {
        smem[threadIdx.y][threadIdx.x] = input[row * cols + col];
    }
    __syncthreads();

    row = blockIdx.x * blockDim.x + threadIdx.y;
    col = blockIdx.y * blockDim.y + threadIdx.x;
    if(row < cols && col < rows) {
        output[row * rows + col] = smem[threadIdx.x][threadIdx.y];
    }
}

extern "C" void transpose_v2(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_transpose_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
}
