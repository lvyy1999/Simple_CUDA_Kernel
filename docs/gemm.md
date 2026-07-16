# GEMM

GEMM 计算：

```text
C = alpha * A * B + beta * C
```

当前实现使用 `half` 输入/输出，内部用 `float` 累加，没有使用 Tensor Core。测试中的 cuBLAS 对比也通过 `CUBLAS_PEDANTIC_MATH` 禁用了 Tensor Core 路径，因此这个文档关注 CUDA Core 路径下的分块、shared memory、寄存器复用、向量化加载和 bank conflict 优化。

- 测试入口：[`tests/test_gemm.cu`](../tests/test_gemm.cu)
- 头文件：[`include/gemm.cuh`](../include/gemm.cuh)
- 当前版本数：6
- 参考库：cuBLAS GEMM

## v1: 朴素 GEMM

源码：[`src/gemm_v1.cu`](../src/gemm_v1.cu)

每个线程计算 `C` 的一个元素，沿 `K` 维循环累加。

```cpp
float sum = 0.0f;
for (int k = 0; k < K; ++k) {
    float a = __half2float(A[row * K + k]);
    float b = __half2float(B[k * N + col]);
    sum += a * b;
}
C[row * N + col] = __float2half_rn(sum * alpha + c_val);
```

特点：

- 实现简单，作为 correctness baseline。
- A/B 的数据复用完全依赖 cache，性能较低。

## v2: Block 级 Shared Memory 分块

源码：[`src/gemm_v2.cu`](../src/gemm_v2.cu)

将 A 和 B 的 tile 搬到 shared memory，再在 tile 内计算。

```cpp
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];
```

收益：

- 减少对全局内存的重复访问。
- 让一个 tile 中的数据被多个线程复用。

## v3: 线程级分块

源码：[`src/gemm_v3.cu`](../src/gemm_v3.cu)

每个线程不再只计算一个 `C` 元素，而是计算一个 `TM x TN` 的小块，并把中间结果保存在寄存器中。

```cpp
float Ct[TM][TN] = {0.0f};
```

收益：

- 增加每个线程的计算工作量。
- 提高 A/B tile 数据在寄存器和 shared memory 中的复用率。
- 明显提升算术强度。

## v4: 外积分解 + 寄存器复用

源码：[`src/gemm_v4.cu`](../src/gemm_v4.cu)

将 shared memory 中的 A/B 片段先加载到寄存器 `At`、`Bt`，再做外积更新 `Ct`。

```cpp
float At[TM];
float Bt[TN];

for (int k = 0; k < BK; k++) {
    for (int i = 0; i < TM; i++) At[i] = As[rc + i * C_BLOCK_ROWS][k];
    for (int j = 0; j < TN; j++) Bt[j] = Bs[k][cc + j * C_BLOCK_COLS];

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            Ct[i][j] += At[i] * Bt[j];
        }
    }
}
```

收益：

- 减少 repeated shared memory load。
- 用寄存器承接更细粒度的数据复用。

## v5: A 转置存储 + `float4` + Padding

源码：[`src/gemm_v5.cu`](../src/gemm_v5.cu)

当前 v5 使用 `BM = 128`、`BN = 128`、`BK = 16`，每个 block 启动 256 个线程，每个线程计算 `8 x 8` 个输出。A tile 以转置布局写入 shared memory，并增加 4 列 padding；shared memory 到寄存器的搬运使用 `float4`。

```cpp
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

__shared__ float As[BK][BM + 4];

FLOAT4(At[i]) = FLOAT4(As[k][4 * rc + i * C_BLOCK_ROWS]);
FLOAT4(Bt[j]) = FLOAT4(Bs[k][4 * cc + j * C_BLOCK_COLS]);
```

收益：

- 降低访存指令数量。
- 改善 shared memory 到寄存器的数据搬运效率。
- 通过 `BM + 4` 的行跨度缓解 A tile 的 shared memory bank conflict。
- v5 仍保留 M/N/K 的边界判断，可以处理非整 tile 的矩阵尺寸。

## v6: Shared Half + Global `Half4` 加载

源码：[`src/gemm_v6.cu`](../src/gemm_v6.cu)

当前 v6 使用 `BM = 128`、`BN = 128`、`BK = 32`。A/B 在 shared memory 中改为 `half`，同时通过一个 8 字节对齐的 `Half4` 联合体从 global memory 一次加载 4 个连续 `half`。

```cpp
union __align__(8) Half4 {
    uint2 packed;
    half values[4];
};

__shared__ half As[BK][BM + 4];
__shared__ half Bs[BK][BN];

value.packed = *reinterpret_cast<const uint2*>(ptr);
```

加载 A 时，4 个连续的 K 维元素在 shared memory 中被转置写入；加载 B 时则保持按行布局。计算阶段再通过 `__half2float` 转换为 `float` 并做 FP32 累加。

特点：

- global memory 加载宽度从单个 `half` 提升到 4 个 `half`。
- shared memory 使用 `half`，在 `BK` 从 16 增加到 32 后，总 shared memory 仍约为 16.25 KiB。
- `BK = 32` 减少沿 K 维迭代时的 block 级同步次数。
- 计算仍然走 CUDA Core 的 FP32 FMA，没有使用 Tensor Core。

v6 是对齐尺寸专用的快速路径。launcher 明确要求：

```text
M % 128 == 0
N % 128 == 0
K % 32 == 0
```

kernel 内的 global load 和 C 写回没有边界判断；不满足约束时 launcher 会打印提示并直接返回。`Half4` 加载还依赖 A/B 起始地址满足 8 字节对齐，当前测试中的 `cudaMalloc` 能满足该要求。

## 当前测试

`tests/test_gemm.cu` 中默认：

- `M = 1024`
- `N = 1024`
- `K = 1024`
- GPU warmup 10 次，计时重复 10 次
- 使用 CPU GEMM 结果做 correctness check
- 额外运行禁用 Tensor Core 后的 cuBLAS GEMM 作为参考

注意：当前自定义 GEMM 没有使用 Tensor Core，测试代码也通过 `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)` 禁用了 cuBLAS Tensor Core 路径；性能对比主要用于观察手写 CUDA Core kernel 和 cuBLAS CUDA Core 路径之间的差距。

README 中保存的 T4 历史结果早于当前 v5/v6 的参数和存储布局调整，不能直接视为这两个最新实现的性能。重新测试时应单独记录 GPU 型号、编译架构和 `ptxas` 的寄存器/spill 信息；Tesla T4 与 RTX 2080 Ti 都使用 `sm_75`，但两张卡的 SM 数量、频率和显存带宽不同，结果不应混在同一组表格中。
