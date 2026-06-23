# Simple_CUDA_Kernel

## 项目说明

本项目使用 CUDA 实现 VectorAdd、Matrix Transpose、Reduce、Softmax、GEMM、Attention 等算子并进行简单的测试

## 目录

- [文件结构](#文件结构)
- [算子实现](#算子实现)
- [测试命令](#测试命令)
- [测试环境](#测试环境)
- [测试结果](#测试结果)

## 文件结构

```text
.
├── include(头文件) 
│   ├── gemm.cuh
│   └── ......
├── src(算子实现) 
│   ├── gemm_v1.cu
│   └── ......
├── docs(算子实现说明文档) 
│   ├── gemm.md
│   └── ......
├── tests(测试代码) 
│   ├── test_gemm.cu
│   └── ......
└── scripts(脚本) 
    ├── test_gemm.sh
    └── ......
```

## 算子实现

[Vector_Add](./docs/vector_add.md)

[Transpose](./docs/transpose.md)

[Reduce](./docs/reduce.md)

[GEMM](./docs/gemm.md)

[Softmax](./docs/softmax.md)

[Attention](./docs/attention.md)

## 测试命令

本项目提供了简单的测试代码和脚本，运行命令如下：

```bash
git clone https://github.com/lvyy1999/Simple_CUDA_Kernel

cd Simple_CUDA_Kernel/scripts
chmod +x test_all.sh
./test_all.sh
```

## 测试环境

本项目测试使用google的Colab平台，环境信息如下：

```text
CPU Information
----------------------------------------
CPU Model           : Intel(R) Xeon(R) CPU @ 2.00GHz
Total CPU Threads   : 2
Threads per Core    : 2
Cores per Socket    : 1
Sockets             : 1

GPU Information
----------------------------------------
0, Tesla T4, 15360 MiB, 580.82.07

Detailed GPU Info
----------------------------------------
Wed Jun 17 15:40:54 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   36C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

CUDA Information
----------------------------------------
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

## 测试结果

每种算子测试均以 CPU 运行的结果作为基准，进行正确性校验和简单的性能比较，其中 GEMM 算子额外增加了和 cuBLAS 官方库的对比（禁用了Tensor Core），测试结果如下：

```text
==================== GEMM ====================
Cpu baseline: 9882.5986 ms, 0.22 GFLOPS
My kernel (v1): 6.4332 ms, 333.81 GFLOPS, 1536.2x Speedup
My kernel (v2): 4.0888 ms, 525.21 GFLOPS, 2417.0x Speedup
My kernel (v3): 0.5649 ms, 3801.30 GFLOPS, 17493.4x Speedup
My kernel (v4): 0.5506 ms, 3899.98 GFLOPS, 17947.5x Speedup
My kernel (v5): 0.5365 ms, 4002.63 GFLOPS, 18419.9x Speedup
My kernel (v6): 0.4808 ms, 4466.72 GFLOPS, 20555.6x Speedup
Nvidia's cuBLAS GEMM: 0.3093 ms, 6942.06 GFLOPS, 31947.0x Speedup
My best performance at v6, reach 64.34% of cuBLAS (No Tensor Core)


==================== Reduce test start ====================
Cpu baseline: 31.3234 ms, 0.54 GFLOPS, 2.14 GB/s
My kernel (v1): 2.4929 ms, 6.73 GFLOPS, 26.92 GB/s, 12.6x Speedup
My kernel (v2): 1.4588 ms, 11.50 GFLOPS, 46.00 GB/s, 21.5x Speedup
My kernel (v3): 0.7722 ms, 21.73 GFLOPS, 86.91 GB/s, 40.6x Speedup
My kernel (v4): 0.4588 ms, 36.57 GFLOPS, 146.27 GB/s, 68.3x Speedup
My kernel (v5): 0.4109 ms, 40.83 GFLOPS, 163.31 GB/s, 76.2x Speedup
My kernel (v6): 0.2568 ms, 65.32 GFLOPS, 261.30 GB/s, 122.0x Speedup
Nvidia's CUB reduce: 0.2732 ms, 61.42 GFLOPS, 245.67 GB/s, 114.7x Speedup
My best performance at v6, reach 106.36% of CUB


==================== Softmax test start ====================
Cpu baseline: 235.4198 ms, 0.36 GFLOPS, 1.14 GB/s
My kernel (v1): 71.4434 ms, 1.17 GFLOPS, 3.76 GB/s, 3.3x Speedup
My kernel (v2): 71.4478 ms, 1.17 GFLOPS, 3.76 GB/s, 3.3x Speedup
My kernel (v3): 50.5252 ms, 1.66 GFLOPS, 5.31 GB/s, 4.7x Speedup
My kernel (v4): 50.6299 ms, 1.66 GFLOPS, 5.30 GB/s, 4.6x Speedup
My best performance at v3


==================== Transpose test start ====================
Cpu baseline: 6.4289 ms, 1.30 GB/s
My kernel (v1): 0.0630 ms, 133.05 GB/s, 102.0x Speedup
My kernel (v2): 0.0453 ms, 185.16 GB/s, 141.9x Speedup
My best performance at v2


==================== Vector add test start ====================
Cpu baseline: 16.0923 ms, 1.04 GFLOPS, 12.51 GB/s
My kernel (v1): 0.7676 ms, 21.86 GFLOPS, 262.28 GB/s, 21.0x Speedup
My kernel (v2): 0.8301 ms, 20.21 GFLOPS, 242.52 GB/s, 19.4x Speedup
My best performance at v1

```
