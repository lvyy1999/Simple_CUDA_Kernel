# Simple_CUDA_Kernel

一个用于学习 CUDA Kernel 优化的算子实验仓库。当前代码覆盖 Vector Add、Matrix Transpose、Reduce、Softmax、GEMM 和 Attention，每个算子都保留了从朴素实现到逐步优化版本的演进，并配套 CPU 基准、正确性检查和简单性能测试。

## 目录

- [项目结构](#项目结构)
- [项目亮点](#项目亮点)
- [已实现算子](#已实现算子)
- [运行方式](#运行方式)
- [测试环境](#测试环境)
- [测试结果](#测试结果)
- [文档索引](#文档索引)
- [说明](#说明)

## 项目结构

```text
.
├── include/        # 头文件，声明各版本 kernel 的 host 入口和函数表
├── src/            # CUDA kernel 实现
├── tests/          # correctness + benchmark 测试程序
├── scripts/        # nvcc 编译和运行脚本
└── docs/           # 各算子的优化思路说明
```

## 项目亮点

- 覆盖 Vector Add、Transpose、Reduce、Softmax、GEMM 和 Attention 6 类常见算子，共实现 24 个渐进式 CUDA Kernel 版本。
- 系统实践合并访存、`float4`/`Half4` 向量化加载、shared memory tiling、padding、寄存器分块、warp shuffle 和 online softmax 等优化方法。
- 为每类算子提供 CPU reference、GPU warmup、CUDA Event 计时和正确性检查；GEMM 使用 `cublasGemmEx`、Reduce 使用 `cub::DeviceReduce::Sum` 作为性能参考。
- 最新 Tesla T4 测试中，GEMM v5 达到禁用 Tensor Core 后 `cublasGemmEx` 的 67.61%，Reduce v6 达到 `cub::DeviceReduce::Sum` 的 102.30%，Attention v4 达到 841.04 GFLOPS。

## 已实现算子

| 算子 | 当前版本 | 主要优化路线 | 文档 |
| --- | ---: | --- | --- |
| Vector Add | v1-v2 | 朴素逐元素计算、`float4` 向量化加载 | [docs/vector_add.md](docs/vector_add.md) |
| Transpose | v1-v2 | 朴素转置、shared memory tiling、padding 消除 bank conflict | [docs/transpose.md](docs/transpose.md) |
| Reduce | v1-v6 | 树形归约、反向步长、双元素加载、warp shuffle、grid-stride loop | [docs/reduce.md](docs/reduce.md) |
| Softmax | v1-v4 | 三遍 softmax、warp 归约、online softmax、online + warp 归约 | [docs/softmax.md](docs/softmax.md) |
| GEMM | v1-v6 | shared memory/寄存器分块、`float4`/`Half4` 加载、shared half、padding | [docs/gemm.md](docs/gemm.md) |
| Attention | v1-v4 | kernel 融合、分块计算、online softmax、一行一 warp、向量化加载 | [docs/attention.md](docs/attention.md) |

各算子的 `include/*.cuh` 中通过 `MAX_KERNEL_VERSION` 和 `kernel_funcs[]` 注册当前可测试版本。新增版本时，需要同时补齐：

1. `src/<op>_v*.cu`
2. `include/<op>.cuh` 中的声明、版本数和函数表
3. `tests/test_<op>.cu` 的测试覆盖
4. `docs/<op>.md` 的说明

## 运行方式

脚本默认使用 `sm_75`，适合 Tesla T4 和 RTX 2080 Ti。也可以给脚本传入其他目标架构，例如 `sm_86`、`sm_89`。

```bash
git clone https://github.com/lvyy1999/Simple_CUDA_Kernel
cd Simple_CUDA_Kernel/scripts
chmod +x test_all.sh test_*.sh

# 跑全部算子
./test_all.sh

# 单独测试某个算子
./test_gemm.sh
./test_attention.sh

# 指定 GPU 架构
./test_gemm.sh sm_86
```

每个测试程序会先运行 CPU baseline，再依次运行 `kernel_funcs[]` 中注册的 CUDA kernel，并将 GPU 输出与 CPU 输出做正确性检查。GEMM 额外对比 cuBLAS 的 `cublasGemmEx` 接口，Reduce 额外对比 CUB 的设备级求和接口 `cub::DeviceReduce::Sum`。

当前测试规模：

| 算子 | 测试规模 |
| --- | --- |
| Vector Add | `N = 1 << 26` |
| Transpose | `1024 x 1024` |
| Reduce | `N = 1 << 24` |
| Softmax | `N = 1 << 24` |
| GEMM | `A: 1024 x 1024, B: 1024 x 1024, C: 1024 x 1024`，`half` 输入/输出，`float` 累加 |
| Attention | `Q: 4096 x 128, K/V: 4096 x 128, O: 4096 x 128` |

## 测试环境

本轮测试结果来自 Google Colab 的 Tesla T4 环境：

```text
Platform: Google Colab
GPU: Tesla T4, 16 GB
Compile target: sm_75
Optimization: -O3
```

## 测试结果

以下结果来自上述 Colab/T4 环境。本轮所有版本均执行了 CPU correctness check，输出中未报告结果不一致。不同 GPU、驱动、CUDA 版本、时钟状态和编译架构下数值会有差异。

### 性能摘要

| 算子 | 最佳版本 | 核心结果 | 对比结果 |
| --- | --- | --- | --- |
| Attention | v4 | 10.3332 ms，841.04 GFLOPS | 相比 v1 提升 4.62x |
| GEMM | v5 | 0.4636 ms，4.63 TFLOPS | `cublasGemmEx`（禁用 Tensor Core）的 67.61% |
| Reduce | v6 | 0.2538 ms，264.46 GB/s | `cub::DeviceReduce::Sum` 的 102.30% |
| Softmax | v3 | 50.4927 ms，5.32 GB/s | 当前自定义最优 |
| Transpose | v2 | 0.0537 ms，156.16 GB/s | 当前自定义最优 |
| Vector Add | v1 | 3.0502 ms，264.02 GB/s | 当前自定义最优 |

### Attention

| 实现 | Time (ms) | GFLOPS | Speedup vs CPU | 备注 |
| --- | ---: | ---: | ---: | --- |
| CPU baseline | 9231.8701 | 0.94 | 1.0x | CPU 参考实现 |
| v1 | 47.7114 | 182.15 | 193.5x | 三 kernel 朴素实现 |
| v2 | 87.0373 | 99.85 | 106.1x | 单 kernel 融合 |
| v3 | 53.6481 | 161.99 | 172.1x | 分块 + online softmax |
| v4 | 10.3332 | 841.04 | 893.4x | 当前最优，一行一 warp + 向量化加载 |

### GEMM

| 实现 | Time (ms) | GFLOPS | Speedup vs CPU | 备注 |
| --- | ---: | ---: | ---: | --- |
| CPU baseline | 10712.9902 | 0.20 | 1.0x | CPU 参考实现 |
| v1 | 5.8108 | 369.57 | 1843.6x | 朴素 CUDA GEMM |
| v2 | 3.7058 | 579.49 | 2890.9x | shared memory 分块 |
| v3 | 0.5698 | 3768.79 | 18801.1x | 线程级分块 |
| v4 | 0.5510 | 3897.15 | 19441.4x | 寄存器复用 |
| v5 | 0.4636 | 4631.81 | 23106.3x | 当前自定义最优，转置存储 + `float4` + padding |
| v6 | 0.4946 | 4341.51 | 21658.2x | shared half + global `Half4` 加载 |
| cuBLAS `cublasGemmEx` | 0.3135 | 6850.36 | 34173.9x | FP16 输入/输出、FP32 累加，禁用 Tensor Core |

自定义最佳版本为 v5，达到 `cublasGemmEx` 的 67.61%。参考测试的 A/B/C 数据类型均为 `CUDA_R_16F`，计算类型为 `CUDA_R_32F`，并通过 `CUBLAS_PEDANTIC_MATH` 限制数学模式；handle 创建和 warmup 不计入 CUDA Event 计时。

### Reduce

| 实现 | Time (ms) | GFLOPS | Bandwidth (GB/s) | Speedup vs CPU | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU baseline | 30.4657 | 0.55 | 2.20 | 1.0x | CPU 参考实现 |
| v1 | 1.5781 | 10.63 | 42.53 | 19.3x | 朴素树形归约 |
| v2 | 0.9501 | 17.66 | 70.63 | 32.1x | 反向步长 |
| v3 | 0.5064 | 33.13 | 132.53 | 60.2x | 每线程读取两个元素 |
| v4 | 0.3186 | 52.67 | 210.67 | 95.6x | 展开最后一个 warp |
| v5 | 0.2867 | 58.51 | 234.05 | 106.3x | warp shuffle |
| v6 | 0.2538 | 66.11 | 264.46 | 120.1x | 当前自定义最优，grid-stride loop |
| CUB `DeviceReduce::Sum` | 0.2596 | 64.63 | 258.51 | 117.4x | CUB 官方设备级求和接口 |

自定义最佳版本为 v6，达到 `cub::DeviceReduce::Sum` 的 102.30%。测试程序先以空临时存储调用该接口查询所需空间，完成分配后再执行 warmup 和 CUDA Event 计时。

### Softmax

| 实现 | Time (ms) | GFLOPS | Bandwidth (GB/s) | Speedup vs CPU | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU baseline | 248.8587 | 0.34 | 1.08 | 1.0x | CPU 参考实现 |
| v1 | 71.3790 | 1.18 | 3.76 | 3.5x | 三遍 softmax |
| v2 | 71.3787 | 1.18 | 3.76 | 3.5x | warp 归约 |
| v3 | 50.4927 | 1.66 | 5.32 | 4.9x | 当前最优，online softmax |
| v4 | 50.5954 | 1.66 | 5.31 | 4.9x | online softmax + warp 归约 |

### Transpose

| 实现 | Time (ms) | Bandwidth (GB/s) | Speedup vs CPU | 备注 |
| --- | ---: | ---: | ---: | --- |
| CPU baseline | 6.2181 | 1.35 | 1.0x | CPU 参考实现 |
| v1 | 0.0827 | 101.41 | 75.2x | 朴素 CUDA 转置 |
| v2 | 0.0537 | 156.16 | 115.8x | 当前最优，shared memory + padding |

### Vector Add

| 实现 | Time (ms) | GFLOPS | Bandwidth (GB/s) | Speedup vs CPU | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU baseline | 68.1312 | 0.98 | 11.82 | 1.0x | CPU 参考实现 |
| v1 | 3.0502 | 22.00 | 264.02 | 22.3x | 当前最优，朴素逐元素 CUDA |
| v2 | 3.2110 | 20.90 | 250.79 | 21.2x | `float4` 向量化加载 |

## 文档索引

- [Vector Add](docs/vector_add.md)
- [Transpose](docs/transpose.md)
- [Reduce](docs/reduce.md)
- [Softmax](docs/softmax.md)
- [GEMM](docs/gemm.md)
- [Attention](docs/attention.md)

## 说明

- 本仓库以学习 CUDA 优化路径为主，代码优先展示不同优化技巧的影响，而不是追求生产级通用算子库。
- GEMM 的自定义 kernel 和 `cublasGemmEx` 对比均未使用 Tensor Core；测试代码通过 `CUBLAS_PEDANTIC_MATH` 禁用 cuBLAS Tensor Core 路径。
- Attention v3 支持 `d <= 128`；v4 针对 `d == 128` 的固定 head dimension 进行专门优化。
