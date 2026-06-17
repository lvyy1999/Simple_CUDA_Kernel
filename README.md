# Simple_CUDA_Kernel

## 项目说明

本项目使用 CUDA 实现 VectorAdd、Matrix Transpose、Reduce、Softmax、GEMM 等基础算子并进行简单的测试

## 运行方法

本项目提供了简单的测试代码和脚本，运行命令如下：

```bash
git clone https://github.com/lvyy1999/Simple_CUDA_Kernel

cd Simple_CUDA_Kernel/scripts
chmod +x test_all.sh
./test_all.sh
```

## 目录结构

```text
.
├── include(头文件) 
│   ├── gemm.cuh
│   └── ......
├── src(算子实现) 
│   ├── gemm_v1.cu
│   └── ......
├── tests(测试代码) 
│   ├── test_gemm.cu
│   └── ......
└── scripts(脚本) 
    ├── test_gemm.sh
    └── ......
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
==================== GEMM test start ====================
Data type: A/B/C = half, accumulate = float
Matrix size: A = 1024 x 1024, B = 1024 x 1024, C = 1024 x 1024
Running CPU baseline...
Running my kernel (v1)...
Correctness checking vs CPU...
Running my kernel (v2)...
Correctness checking vs CPU...
Running my kernel (v3)...
Correctness checking vs CPU...
Running cuBLAS GEMM...

Benchmark analyzing...
Cpu baseline: 9911.2158 ms, 0.22 GFLOPS
My kernel (v1): 5.8062 ms, 369.86 GFLOPS, 1707.0x Speedup
My kernel (v2): 3.7176 ms, 577.66 GFLOPS, 2666.1x Speedup
My kernel (v3): 0.9398 ms, 2285.07 GFLOPS, 10546.2x Speedup
Nvidia's cuBLAS GEMM: 0.4009 ms, 5356.97 GFLOPS, 24723.8x Speedup
My best performance at v3, reach 42.66% of cuBLAS (No Tensor Core)

run: test_reduce.sh

==================== Reduce test start ====================
Data type: float, Vector size: N = 16777216, Bytes: 64 MB
Running CPU baseline...
Running my kernel (v1)...
Correctness checking vs CPU...
Running my kernel (v2)...
Correctness checking vs CPU...
Running my kernel (v3)...
Correctness checking vs CPU...
Running my kernel (v4)...
Correctness checking vs CPU...
Running my kernel (v5)...
Correctness checking vs CPU...

Benchmark analyzing...
Cpu baseline: 31.8616 ms, 0.53 GFLOPS, 2.11 GB/s
My kernel (v1): 1.9418 ms, 8.64 GFLOPS, 34.56 GB/s, 16.4x Speedup
My kernel (v2): 1.1644 ms, 14.41 GFLOPS, 57.64 GB/s, 27.4x Speedup
My kernel (v3): 0.6142 ms, 27.32 GFLOPS, 109.26 GB/s, 51.9x Speedup
My kernel (v4): 0.3772 ms, 44.48 GFLOPS, 177.92 GB/s, 84.5x Speedup
My kernel (v5): 0.3386 ms, 49.54 GFLOPS, 198.18 GB/s, 94.1x Speedup
My best performance at v5

run: test_softmax.sh

==================== Softmax test start ====================
Data type: float, Vector size: N = 16777216, Bytes: 64 MB
Running CPU baseline...
Running my kernel (v1)...
Correctness checking vs CPU...
Running my kernel (v2)...
Correctness checking vs CPU...
Running my kernel (v3)...
Correctness checking vs CPU...
Running my kernel (v4)...
Correctness checking vs CPU...

Benchmark analyzing...
Cpu baseline: 227.1248 ms, 0.37 GFLOPS, 1.18 GB/s
My kernel (v1): 71.1882 ms, 1.18 GFLOPS, 3.77 GB/s, 3.2x Speedup
My kernel (v2): 71.2048 ms, 1.18 GFLOPS, 3.77 GB/s, 3.2x Speedup
My kernel (v3): 50.3404 ms, 1.67 GFLOPS, 5.33 GB/s, 4.5x Speedup
My kernel (v4): 50.4458 ms, 1.66 GFLOPS, 5.32 GB/s, 4.5x Speedup
My best performance at v3

run: test_transpose.sh

==================== Transpose test start ====================
Data type: float, Matrix size: A = 1024 × 1024, Bytes: 4 MB
Running CPU baseline...
Running my kernel (v1)...
Correctness checking vs CPU...
Running my kernel (v2)...
Correctness checking vs CPU...

Benchmark analyzing...
Cpu baseline: 5.4201 ms, 1.55 GB/s
My kernel (v1): 0.0722 ms, 116.22 GB/s, 75.1x Speedup
My kernel (v2): 0.0494 ms, 169.66 GB/s, 109.6x Speedup
My best performance at v2

run: test_vector_add.sh

==================== Vector add test start ====================
Data type: float, Vector size: N = 16777216, Bytes: 64 MB
Running CPU baseline...
Running my kernel (v1)...
Correctness checking vs CPU...
Running my kernel (v2)...
Correctness checking vs CPU...

Benchmark analyzing...
Cpu baseline: 18.7105 ms, 0.90 GFLOPS, 10.76 GB/s
My kernel (v1): 0.7747 ms, 21.66 GFLOPS, 259.89 GB/s, 24.2x Speedup
My kernel (v2): 0.8076 ms, 20.77 GFLOPS, 249.29 GB/s, 23.2x Speedup
My best performance at v1
```
