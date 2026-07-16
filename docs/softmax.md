# Softmax

Softmax 计算：

```text
output[i] = exp(input[i] - max(input)) / sum_j exp(input[j] - max(input))
```

实现中使用减最大值的形式保证数值稳定。该算子同时包含规约和逐元素写回，适合学习 block 级归约、warp shuffle 和 online softmax。

- 测试入口：[`tests/test_softmax.cu`](../tests/test_softmax.cu)
- 头文件：[`include/softmax.cuh`](../include/softmax.cuh)
- 当前版本数：4

## v1: 三遍 Softmax + Shared Memory 归约

源码：[`src/softmax_v1.cu`](../src/softmax_v1.cu)

计算分三步：

1. 遍历输入求最大值 `maxVal`
2. 遍历输入求 `sumExp`
3. 再遍历输入写回归一化结果

```cpp
for (int i = tid; i < N; i += blockDim.x) {
    maxVal = fmaxf(maxVal, input[i]);
}

for (int i = tid; i < N; i += blockDim.x) {
    sumExp += expf(input[i] - maxVal);
}

for (int i = tid; i < N; i += blockDim.x) {
    output[i] = expf(input[i] - maxVal) / sumExp;
}
```

特点：

- 思路清晰，适合作为 baseline。
- max 和 sum 都通过 shared memory 树形归约完成。
- 对输入数组需要多次遍历。

## v2: Warp 归约

源码：[`src/softmax_v2.cu`](../src/softmax_v2.cu)

用 `__shfl_down_sync` 先在 warp 内归约，再用 shared memory 汇总各 warp 的结果。

```cpp
for (int offset = 16; offset > 0; offset >>= 1) {
    m = fmaxf(m, __shfl_down_sync(0xFFFFFFFF, m, offset));
}
```

收益：

- 减少 shared memory 使用和 block 级同步压力。
- 归约路径更接近 Reduce v5。

## v3: Online Softmax

源码：[`src/softmax_v3.cu`](../src/softmax_v3.cu)

online softmax 在扫描过程中同时维护当前最大值 `m` 和归一化分母 `s`。当遇到更大的最大值时，用缩放项修正旧的累积和。

```cpp
float m = -FLT_MAX, s = 0.0f;
for (int i = tid; i < N; i += blockDim.x) {
    float x = input[i];
    float m_new = fmaxf(m, x);
    s = s * expf(m - m_new) + expf(x - m_new);
    m = m_new;
}
```

收益：

- 将求 max 和求 sum 的两次读合并为一次扫描。
- 保持数值稳定。
- 是 FlashAttention 中在线归一化思想的基础。

## v4: Online Softmax + Warp 归约

源码：[`src/softmax_v4.cu`](../src/softmax_v4.cu)

在 v3 的 online softmax 基础上，用 warp shuffle 合并 `(m, s)` 二元组。

```cpp
float m2 = __shfl_down_sync(0xFFFFFFFF, m, offset);
float s2 = __shfl_down_sync(0xFFFFFFFF, s, offset);
float m_new = fmaxf(m, m2);
s = s * expf(m - m_new) + s2 * expf(m2 - m_new);
m = m_new;
```

特点：

- 归约逻辑更复杂，但减少 shared memory 树形归约。
- 历史 T4 结果中 v3 和 v4 接近，v3 略快，说明优化收益需要结合具体硬件和 kernel 资源占用判断。

## 当前测试

`tests/test_softmax.cu` 中默认：

- `N = 1 << 24`
- GPU warmup 10 次，计时重复 10 次
- 使用 CPU softmax 结果做 correctness check
