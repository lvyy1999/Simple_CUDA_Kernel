# Vector Add

Vector Add 计算 `C[i] = A[i] + B[i]`。这个算子计算量很小，主要受全局内存带宽限制，适合作为 CUDA 线程映射、访存合并和向量化加载的入门实验。

- 测试入口：[`tests/test_vector_add.cu`](../tests/test_vector_add.cu)
- 头文件：[`include/vector_add.cuh`](../include/vector_add.cuh)
- 当前版本数：2

## v1: 朴素逐元素实现

源码：[`src/vector_add_v1.cu`](../src/vector_add_v1.cu)

每个线程负责一个元素，使用一维 grid 覆盖整个数组。

```cpp
__global__ void vector_add_kernel_v1(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

特点：

- 实现最直接，便于验证正确性。
- 相邻线程访问相邻 `float`，能够形成合并访存。
- 性能主要取决于内存带宽和 launch 配置。

## v2: `float4` 向量化加载

源码：[`src/vector_add_v2.cu`](../src/vector_add_v2.cu)

每次以 `float4` 为单位读取 A/B 并写回 C，同时使用 grid-stride loop 覆盖更大的输入。

```cpp
const float4* vec_A = reinterpret_cast<const float4*>(A);
const float4* vec_B = reinterpret_cast<const float4*>(B);
float4* vec_C = reinterpret_cast<float4*>(C);
int vec_N = N / 4;

for (int i = idx; i < vec_N; i += stride) {
    float4 a = vec_A[i];
    float4 b = vec_B[i];
    vec_C[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
```

特点：

- 减少单个线程的访存指令数量。
- 对齐良好且问题规模足够大时通常更有利。
- 尾部元素通过标量路径处理。

## 当前测试

`tests/test_vector_add.cu` 中默认：

- `N = 1 << 26`
- GPU warmup 10 次，计时重复 10 次
- 使用 CPU 结果做 correctness check

历史 T4 结果中，v1 略快于 v2。这说明向量化加载并不一定总能带来收益，实际效果会受到对齐、指令调度、访存吞吐和编译器生成代码影响。
