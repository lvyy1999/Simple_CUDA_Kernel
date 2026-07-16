# Matrix Transpose

Matrix Transpose 将输入矩阵 `input[rows, cols]` 转置为 `output[cols, rows]`。该算子重点观察全局内存读写是否合并，以及 shared memory padding 对 bank conflict 的影响。

- 测试入口：[`tests/test_transpose.cu`](../tests/test_transpose.cu)
- 头文件：[`include/transpose.cuh`](../include/transpose.cuh)
- 当前版本数：2

## v1: 朴素转置

源码：[`src/transpose_v1.cu`](../src/transpose_v1.cu)

每个线程处理一个元素，从输入矩阵读取后写到输出矩阵的转置位置。

```cpp
__global__ void matrix_transpose_kernel_v1(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        output[col * rows + row] = input[row * cols + col];
    }
}
```

特点：

- 输入读取是连续的，容易合并访存。
- 输出写入按列跳跃，访存不连续，带宽利用率受限。

## v2: Shared Memory Tiling + Padding

源码：[`src/transpose_v2.cu`](../src/transpose_v2.cu)

先将一个 `32 x 32` tile 连续读入 shared memory，再交换块坐标后写回输出。shared memory 使用 `TILE_SIZE + 1` 的列数 padding，减少按列访问时的 bank conflict。

```cpp
#define TILE_SIZE 32

__shared__ float smem[TILE_SIZE][TILE_SIZE + 1];

smem[threadIdx.y][threadIdx.x] = input[row * cols + col];
__syncthreads();

output[row * rows + col] = smem[threadIdx.x][threadIdx.y];
```

特点：

- 全局内存读和写都更接近连续访问。
- padding 避免 shared memory 转置读写时多个线程落到同一个 bank。
- 适合说明“用 shared memory 改变访存形态”的经典 CUDA 优化。

## 当前测试

`tests/test_transpose.cu` 中默认：

- 矩阵大小 `1024 x 1024`
- GPU warmup 10 次，计时重复 10 次
- 使用 CPU 转置结果做 correctness check

历史 T4 结果中，v2 明显快于 v1，说明 shared memory tiling 对转置这类访存主导算子很有效。
