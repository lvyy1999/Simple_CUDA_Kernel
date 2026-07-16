# Attention

Attention 计算过程为：

```text
S = Q * K^T / sqrt(d)
P = softmax(S)
O = P * V
```

当前实现为单头 dense attention，输入输出布局如下：

- `Q: [M, d]`
- `K: [N, d]`
- `V: [N, d]`
- `output: [M, d]`

- 测试入口：[`tests/test_attention.cu`](../tests/test_attention.cu)
- 头文件：[`include/attention.cuh`](../include/attention.cuh)
- 当前注册版本：v1-v4，共 4 个版本

## v1：三个 Kernel 的朴素实现

源码：[`src/attention_v1.cu`](../src/attention_v1.cu)

v1 将 attention 拆成三个 CUDA kernel：

1. `qkt_kernel`：计算 `S = QK^T / sqrt(d)`
2. `softmax_kernel`：逐行计算 `P = softmax(S)`
3. `pv_kernel`：计算 `O = PV`

```cpp
qkt_kernel<<<grid_qkt, block_qkt>>>(Q, K, S, M, N, d);
softmax_kernel<<<M, 256, 256 * sizeof(float)>>>(S, P, M, N);
pv_kernel<<<grid_pv, block_pv>>>(P, V, output, M, N, d);
```

特点：

- 结构清晰，最容易和数学公式对应。
- 需要额外分配 `S` 和 `P` 两个 `M x N` 中间矩阵。
- `S`、`P` 都会写入并重新读取全局内存，显存流量较大。
- 适合作为 correctness baseline，不是面向高性能的实现。

## v2：单 Kernel 融合

源码：[`src/attention_v2.cu`](../src/attention_v2.cu)

v2 将 `QK^T -> softmax -> PV` 融合到一个 kernel 中。每个 block 负责一个 query 行，先将该行的 `N` 个 score 放入动态 shared memory，再完成 softmax 和 V 的加权求和。

```cpp
extern __shared__ float scores[];

for (int j = tid; j < N; j += blockDim.x) {
    float sum = 0.0f;
    for (int k = 0; k < d; k++) {
        sum += Q[row * d + k] * K[j * d + k];
    }
    scores[j] = sum * rsqrtf(d);
}
```

特点：

- 不再将完整的 `S` 和 `P` 写回全局内存。
- 减少了 kernel launch 次数。
- 每个 block 只处理一行 query，block 间并行度由 `M` 决定。
- 每个 block 需要约 `N * sizeof(float)` 的动态 shared memory，因此可支持的 `N` 受设备 shared memory 上限约束。

简单地融合三个阶段并不等价于 FlashAttention。v2 减少了中间结果的全局内存流量，但没有对 K/V 进行分块复用，shared memory 需求还会随 `N` 线性增长。

## v3：分块 Attention + Online Softmax

源码：[`src/attention_v3.cu`](../src/attention_v3.cu)

v3 按 query block 和 key/value block 分块。每个 block 处理 8 行 query，每个 warp 对应一行 query；K/V 以 32 行为一块进行扫描，并使用 online softmax 累积结果。

当前启动参数：

```cpp
constexpr int BM = 8;
constexpr int BN = 32;
constexpr int BD = 128;
constexpr int BLOCK_SIZE = 256;
```

shared memory 中保存：

- `Qs[BM * BD]`：当前 8 行 Q
- `Ks[BN * BD]`、`Vs[BN * BD]`：当前 32 行 K/V
- `S[BM * BN]`：当前 score tile
- `o[BM * BD]`：未归一化的输出累积
- `m[BM]`、`l[BM]`：每一行的最大值和 softmax 分母

online softmax 的目标更新公式为：

```cpp
float m_new = fmaxf(m_old, m_cur);
float alpha = expf(m_old - m_new);
float l_new = l_old * alpha + l_cur;

o_new = o_old * alpha + o_cur;
```

特点：

- 不再显式保存完整的 `M x N` score/probability 矩阵。
- K/V 按 tile 载入 shared memory，避免 v2 中 shared memory 随 `N` 增长。
- online softmax 允许在逐块扫描 K/V 时保持全局归一化语义。
- 启动器支持 `d <= 128`，但 tile 存储按 `BD = 128` 固定分配。

### v3 当前已知问题

当前源码中的 `alpha` 使用了未定义变量 `m_i`：

```cpp
float alpha = cb == 0 ? expf(m_i - m_new) : 0.0f;
```

这里应使用该行已经读取的旧最大值 `m_old`。在修正前，v3 会阻止完整测试程序通过编译。

此外，v3 只在 `c < N` 的线程中执行使用 `0xFFFFFFFF` mask 的 warp shuffle，并由这些线程分别更新输出列。当 `N` 不是 32 的整数倍时，最后一个 K/V tile 的部分 lane 不参与 shuffle，也不会更新其负责的输出列。因此当前 v3 只能在 `N % 32 == 0` 时保证这条路径完整，默认测试的 `N = 4096` 会掩盖该问题。

## v4：一行一个 Warp + 向量化加载

源码：[`src/attention_v4.cu`](../src/attention_v4.cu)

v4 保留分块和 online softmax，但重新组织了线程职责：

- 一个 block 包含 8 个 warp，处理 8 行 query。
- 一个 warp 的 32 个 lane 共同处理一行 query。
- K/V tile 大小为 `BN = 32`，head dimension 固定为 `BD = 128`。
- 每个 lane 最终在寄存器数组 `acc[4]` 中保存 4 个输出元素。

当前启动参数：

```cpp
constexpr int BM = 8;
constexpr int BN = 32;
constexpr int BD = 128;
constexpr int BLOCK_SIZE = BM * BN;  // 256 threads
```

### v4 的主要优化

1. **Q/K/V 全局内存向量化加载**

   每个 lane 使用一次 `float4` 读取连续 4 个 FP32 元素。32 个 lane 正好覆盖一行 128 个元素，减少 load 指令数量，并形成连续合并访存。

2. **K 的 shared memory Padding**

   ```cpp
   __shared__ float Ks[BN][BD + 1];
   ```

   K 在计算 `QK^T` 时会被 warp 按列方向读取。末维增加一个元素可以改变相邻行的起始 bank，缓解访问固定列时的 shared memory bank conflict。

3. **Online Softmax 状态保存在寄存器**

   每个 warp 使用寄存器变量 `m_i`、`l_i` 维护对应 query 行的 softmax 状态，输出部分累积在 `acc[4]` 中，减少 v3 对 shared memory 状态的访问。

4. **每个概率只计算一次**

   当前 tile 的未归一化概率先写入 `P[BM][BN]`，同一 warp 再复用这些值完成 `P * V`，避免为不同输出列重复调用 `expf`。

5. **正确屏蔽 N 的尾块**

   超出 `N` 的 lane 将 `score` 设为 `-FLT_MAX`、将 `p` 设为 0，但整个 warp 仍参与 shuffle 和输出累积，因此 v4 可以处理 `N` 不是 32 整数倍的情况。

### v4 使用约束

- 启动器明确要求 `d == 128`，其他 head dimension 会直接返回。
- `float4` 读取要求地址满足 16 字节对齐。测试中的 `cudaMalloc` 地址满足基础对齐要求，且 `d = 128` 使每一行起始地址继续保持 16 字节对齐。
- 当前实现使用 38,016 字节（约 37.1 KiB）静态 shared memory，适用于默认每 block 至少支持 48 KiB shared memory 的目标设备。
- 仍然使用 CUDA Core 完成 FP32 乘加，没有使用 Tensor Core。

## 当前测试

[`tests/test_attention.cu`](../tests/test_attention.cu) 当前固定使用：

- `M = 4096`
- `N = 4096`
- `d = 128`
- GPU warmup 10 次，计时重复 10 次
- 使用 CPU attention 结果逐版本检查正确性
- 估算计算量为 `4MNd + 6MN`

头文件会依次注册并测试 v1-v4。由于 v3 仍有未定义变量，当前直接编译全部 `src/attention_v*.cu` 时会先遇到 v3 编译错误；文档中的 v4 结构说明来自静态代码核对，本地未进行 CUDA 编译与 2080 Ti 运行验证。

## 后续优化方向

- 使用 `half2`、Tensor Core 或 WMMA 加速 `QK^T` 和 `PV`，同时保留 FP32 softmax 与累积以控制误差。
- 用 warp-level MMA 重新设计 tile，避免当前逐元素 FP32 点积吞吐不足。
- 对 `expf`、寄存器数量、shared memory bank conflict 和 occupancy 使用 Nsight Compute 做定量分析。
- 为 `M`、`N` 和 `d` 增加参数化测试，覆盖 `M % 8 != 0`、`N % 32 != 0`、小尺寸以及 v4 不支持的 head dimension。
- 在 2080 Ti 上单独记录最新代码的结果，不与 README 中其他 GPU 或旧实现的历史数据直接混用。

v3 解决了 v2 的中间矩阵和 shared memory 随 `N` 增长问题；v4 则进一步将计算映射到 warp、向量化全局内存访问，并把 softmax 与输出状态移入寄存器，是当前更适合作为后续优化基础的版本。
