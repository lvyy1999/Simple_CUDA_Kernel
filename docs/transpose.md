## Transpose

### v1 : naive版

源码：[transpose_v1.cu](../src/transpose_v1.cu)

```cpp
__global__ void matrix_transpose_kernel_v1(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    if (col < cols && row < rows) {
        output[col * rows + row] = input[row * cols + col];
    }
}
```

### v2 : 共享内存 + Padding

使用共享内存 + Padding 的方式存储数据，消除按列访问时的 Bank Conflict
源码：[transpose_v1.cu](../src/transpose_v1.cu)

```cpp
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
```
