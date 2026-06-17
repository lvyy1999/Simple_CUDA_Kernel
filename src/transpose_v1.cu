#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel_v1(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    if (col < cols && row < rows) {
        output[col * rows + row] = input[row * cols + col];
    }
}

extern "C" void transpose_v1(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_transpose_kernel_v1<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
}
