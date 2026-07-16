# Reduce

Reduce 将长度为 `N` 的数组规约为一个标量和。这个算子适合观察线程同步、warp divergence、shared memory bank conflict、warp shuffle 和 grid-stride loop 对性能的影响。

- 测试入口：[`tests/test_reduce.cu`](../tests/test_reduce.cu)
- 头文件：[`include/reduce.cuh`](../include/reduce.cuh)
- 当前版本数：6
- 参考库：CUB reduce

## v1: 朴素树形归约

源码：[`src/reduce_v1.cu`](../src/reduce_v1.cu)

每个 block 在 shared memory 中做树形归约，最后由 `thread 0` 通过 `atomicAdd` 写入全局输出。

```cpp
for (int s = 1; s < blockDim.x; s <<= 1) {
    if (tid % (2 * s) == 0) {
        smem[tid] += smem[tid + s];
    }
    __syncthreads();
}
```

问题：

- `tid % (2 * s)` 会造成明显 warp divergence。
- 早期步长下 shared memory 访问模式不够友好。
- 每一轮都需要 block 级同步。

## v2: 反向步长归约

源码：[`src/reduce_v2.cu`](../src/reduce_v2.cu)

将归约步长从 `blockDim.x / 2` 逐步减半，让活跃线程集中在前半段。

```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        smem[tid] += smem[tid + s];
    }
    __syncthreads();
}
```

收益：

- 分支条件更简单。
- 活跃线程更集中，减少 warp divergence。
- shared memory 访问更规整。

## v3: 每线程读取两个元素

源码：[`src/reduce_v3.cu`](../src/reduce_v3.cu)

每个线程先从全局内存读取两个元素并相加，再进入 block 内归约。

```cpp
smem[tid] = (idx < N) ? input[idx] : 0.0f;
smem[tid] += (idx + blockDim.x * gridDim.x < N)
    ? input[idx + blockDim.x * gridDim.x]
    : 0.0f;
```

收益：

- block 数量约减半。
- 减少空闲线程和原子写回次数。
- 提高每个线程的工作量。

## v4: 展开最后一个 warp

源码：[`src/reduce_v4.cu`](../src/reduce_v4.cu)

block 级归约只做到 `s > 32`，最后一个 warp 内使用 shuffle 完成归约。

```cpp
if (tid < 32) {
    float val = smem[tid] + smem[tid + 32];
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (tid == 0) atomicAdd(output, val);
}
```

收益：

- 最后一个 warp 内不再需要 `__syncthreads()`。
- 减少 shared memory 往返。

## v5: Warp Shuffle 分层归约

源码：[`src/reduce_v5.cu`](../src/reduce_v5.cu)

每个 warp 先用 shuffle 得到局部和，再把每个 warp 的结果写入 shared memory，最后由第一个 warp 汇总。

```cpp
val = warp_reduce(val);
if (lane_id == 0) warp_sum[warp_id] = val;
__syncthreads();

val = (tid < num_warps) ? warp_sum[tid] : 0.0f;
if (warp_id == 0) val = warp_reduce(val);
```

收益：

- 大部分规约在寄存器和 warp 内通信中完成。
- shared memory 只保存每个 warp 的部分和。

## v6: Grid-Stride Loop

源码：[`src/reduce_v6.cu`](../src/reduce_v6.cu)

固定 grid 大小，每个线程用 grid-stride loop 覆盖多个输入元素。

```cpp
float val = 0.0f;
int stride = blockDim.x * gridDim.x;
for (int i = idx; i < N; i += stride) {
    val += input[i];
}
```

收益：

- 减少过多 block 带来的调度和原子写压力。
- 每个线程累加更多元素，提升访存和指令效率。
- 历史 T4 测试中 v6 已略快于 CUB 结果。

## 当前测试

`tests/test_reduce.cu` 中默认：

- `N = 1 << 24`
- GPU warmup 10 次，计时重复 10 次
- 使用 CPU 结果做 correctness check
- 额外运行 CUB reduce 作为参考
