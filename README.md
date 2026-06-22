# Simple_CUDA_Kernel

## 项目说明

本项目使用 CUDA 实现 VectorAdd、Matrix Transpose、Reduce、Softmax、GEMM 等基础算子并进行简单的测试

## 目录

- [文件结构](#文件结构)
- [算子实现](#算子实现)
  - [Vector_Add](#Vector_Add)
  - [Transpose](#Transpose)
  - [Reduce](#Reduce)
  - [GEMM](#GEMM)
  - [Softmax](#Softmax)
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
├── tests(测试代码) 
│   ├── test_gemm.cu
│   └── ......
└── scripts(脚本) 
    ├── test_gemm.sh
    └── ......
```

## 算子实现

### Vector_Add

#### v1 : naive版
源码：[./src/vector_add_v1.cu](./src/vector_add_v1.cu)
```cpp
__global__ void vector_add_kernel_v1(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

### v2 : 向量化加载
源码：[./src/vector_add_v2.cu](./src/vector_add_v2.cu)
```cpp
__global__ void vector_add_kernel_v2(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid Stride Loop + 向量化加载
    const float4* vec_A = reinterpret_cast<const float4*>(A);
    const float4* vec_B = reinterpret_cast<const float4*>(B);
    float4* vec_C = reinterpret_cast<float4*>(C);
    int vec_N = N / 4;
    for(int i = idx; i < vec_N; i += stride) {
        float4 a = vec_A[i];
        float4 b = vec_B[i];
        float4 c = make_float4(
            a.x + b.x, 
            a.y + b.y, 
            a.z + b.z, 
            a.w + b.w
        );
        vec_C[i] = c;
    }

    // 尾部处理
    if(int i = vec_N * 4 + idx; i < N) {
        C[i] = A[i] + B[i];
    }
}
```

### Transpose

#### v1 : naive版
源码：[./src/transpose_v1.cu](./src/transpose_v1.cu)
```cpp
__global__ void matrix_transpose_kernel_v1(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    if (col < cols && row < rows) {
        output[col * rows + row] = input[row * cols + col];
    }
}
```

#### v2 : 共享内存 + Padding
使用共享内存 + Padding 的方式存储数据，消除按列访问时的 Bank Conflict
源码：[./src/transpose_v1.cu](./src/transpose_v1.cu)
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

### Reduce

#### v1 : naive版
采用树形规约，最后每个 block 内的 thread 0 将 block 内的规约结果原子加到输出
源码：[./src/reduce_v1.cu](./src/reduce_v1.cu)
```cpp
__global__ void reduction_kernel_v1(const float* input, float* output, int N) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // 朴素树形规约
    for(int s = 1; s < blockDim.x; s <<= 1) {
        if(tid % (2 * s) == 0) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) atomicAdd(output, smem[0]);
}
```

#### v2 : 反转步长
反转规约时的步长方向，减少 Warp Divergence 和 Bank Conflict
源码：[./src/reduce_v2.cu](./src/reduce_v2.cu)
```cpp
for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) {
        smem[tid] += smem[tid + s];
    }
    __syncthreads();
}
```

#### v3 : 减少空闲线程
每个线程负责读取两个数据，block 数量减半，大幅减少空闲线程
源码：[./src/reduce_v3.cu](./src/reduce_v3.cu)
```cpp
smem[tid] = (idx < N) ? input[idx] : 0.0f;
smem[tid] += (idx + blockDim.x * gridDim.x < N) ? input[idx + blockDim.x * gridDim.x] : 0.0f;

int blocksPerGrid = (N + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock);
```

#### v4 : 展开最后一个 warp
树形规约只处理到步长大于32的部分，当步长小于等于32时，活跃线程只剩第一个 warp 内的线程，改用 warp内的同步原语进行规约，更高效且无需全局线程等待
源码：[./src/reduce_v4.cu](./src/reduce_v4.cu)
```cpp
for(int s = blockDim.x / 2; s > 32; s >>= 1) {
    if(tid < s) {
        smem[tid] += smem[tid + s];
    }
    __syncthreads();
}

if(tid < 32) {
    float val = smem[tid] + smem[tid + 32];
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if(tid == 0) atomicAdd(output, val);
}
```

#### v5 : warp shuffle
全部使用 warp shuffle 替代朴素树形规约，由于 warp 数量最大为32(1024 / 32 = 32)，开辟大小为32的共享内存 smem，第 i 个 warp 将 warp 内的规约结果写到 smem[i]，再由第一个 warp 对 smem 进行规约
源码：[./src/reduce_v5.cu](./src/reduce_v5.cu)
```cpp 
int tid = threadIdx.x;
int warp_id = tid / 32;
int lane_id = tid % 32;

// 每个 warp 内进行规约
val = warp_reduce(val);

// 每个 warp 内的 lane 0 将结果写入共享内存
__shared__ float warp_sum[32]; // 最多 32 个 warp
if(lane_id == 0) warp_sum[warp_id] = val;
__syncthreads();

// 最终结果由 warp 0 负责规约
int num_warps = blockDim.x / 32;
val = (tid < num_warps) ? warp_sum[tid] : 0.0f;
if(warp_id == 0) val = warp_reduce(val);

if(tid == 0) atomicAdd(output, val);
```

#### v6 : Grid Stride Loop
采用固定 grid 大小，每个线程采用跳步循环覆盖整个数组
源码：[./src/reduce_v6.cu](./src/reduce_v6.cu)
```cpp 
float val = 0.0f;
int stride = blockDim.x * gridDim.x;
for(int i = idx; i < N; i += stride) {
    val += input[i];
}
```

### GEMM

#### v1 : naive版
源码：[./src/gemm_v1.cu](./src/gemm_v1.cu)
```cpp
__global__ void gemm_kernel_v1(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[k * N + col]);
            sum += a * b;
        }
        float c_val = (beta != 0.0f) ? __half2float(C[row * N + col]) * beta : 0.0f;
        C[row * N + col] = __float2half_rn(sum * alpha + c_val);
    }
}
```

#### v2 : block 级分块
将矩阵A和B分块读入共享内存，然后在共享内存中计算矩阵乘，增加数据复用，减少全局内存访问
源码：[./src/gemm_v2.cu](./src/gemm_v2.cu)
```cpp
#define TILE_SIZE 16

__global__ void gemm_kernel_v2(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for(int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? __half2float(A[row * K + aCol]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? __half2float(B[bRow * N + col]) : 0.0f;
        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < M && col < N) {
        float c_val = (beta != 0.0f) ? __half2float(C[row * N + col]) * beta : 0.0f;
        C[row * N + col] = __float2half_rn(sum * alpha + c_val);
    }
}
```

#### v3 : thread 级分块
每个线程负责更多数据，在寄存器上存储中间结果
源码：[./src/gemm_v3.cu](./src/gemm_v3.cu)
```cpp
template <int BLOCK_SIZE, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_kernel_v3(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    int tid = threadIdx.x;
    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;

    // 处理 A 时，block 内重排为 32 * 8，分四次读 A 的一块(128 * 8)
    constexpr int A_BLOCK_COLS = BK; // 8
    constexpr int A_BLOCK_ROWS = BLOCK_SIZE / A_BLOCK_COLS; // 32
    int ra = tid / A_BLOCK_COLS, ca = tid % A_BLOCK_COLS;

    // 处理 B 时，block 内重排为 8 * 32，分四次读 B 的一块(8 * 128)
    constexpr int B_BLOCK_ROWS = BK; // 8
    constexpr int B_BLOCK_COLS = BLOCK_SIZE / B_BLOCK_ROWS; // 32
    int rb = tid / B_BLOCK_COLS, cb = tid % B_BLOCK_COLS;

    // 处理 C 时，block 内重排为 16 * 16
    constexpr int C_BLOCK_ROWS = 16;
    constexpr int C_BLOCK_COLS = BLOCK_SIZE / C_BLOCK_ROWS;
    int rc = tid / 16, cc = tid % 16;
    
    // 每个 thread 负责 C 中的 TM * TN 个元素
    // constexpr int TM = BM / C_BLOCK_ROWS; // 8
    // constexpr int TN = BN / C_BLOCK_COLS; // 8
    float Ct[TM][TN] = {0.0f};

    // 沿着 K 维度遍历
    for(int k0 = 0; k0 < K; k0 += BK) {
        // 读取 A 的一块
        for(int i = ra; i < BM; i += A_BLOCK_ROWS) {
            int r = r0 + i, c = k0 + ca;
            As[i][ca] = (r < M && c < K) ? __half2float(A[r * K + c]) : 0.0f;
        }

        // 读取 B 的一块
        for(int j = cb; j < BN; j += B_BLOCK_COLS) {
            int r = k0 + rb, c = c0 + j;
            Bs[rb][j] = (r < K && c < N) ? __half2float(B[r * N + c]) : 0.0f;
        }

        __syncthreads();

        // 计算 As * Bs
        for(int k = 0; k < BK; k++) {
            for(int i = 0; i < TM; i++) {
                int r = rc + i * C_BLOCK_ROWS;
                for(int j = 0; j < TN; j++) {
                    int c = cc + j * C_BLOCK_COLS;
                    Ct[i][j] += As[r][k] * Bs[k][c];
                }
            }
        }

        __syncthreads();
    }

    // write C
    for(int i = 0; i < TM; i++) {
        int r = r0 + i * C_BLOCK_ROWS + rc;
        for(int j = 0; j < TN; j++) {
            int c = c0 + j * C_BLOCK_COLS + cc;
            if(r < M && c < N) {
                float c_val = (beta != 0.0f) ? __half2float(C[r * N + c]) * beta : 0.0f;
                C[r * N + c] = __float2half_rn(Ct[i][j] * alpha + c_val);
            }
        } 
    }
}
```

#### v4 : 外积分解 + 寄存器级复用
先将共享内存数据搬运到寄存器，再在寄存器上做外积，增加数据复用
源码：[./src/gemm_v4.cu](./src/gemm_v4.cu)
```cpp
float At[TM];
float Bt[TN];
float Ct[TM][TN] = {0.0f};    

for(int k = 0; k < BK; k++) {
    #pragma unroll
    for(int i = 0; i < TM; i++) {
        At[i] = As[rc + i * C_BLOCK_ROWS][k];
    }
    #pragma unroll
    for(int j = 0; j < TN; j++) {
        Bt[j] = Bs[k][cc + j * C_BLOCK_COLS];
    }
    // 在寄存器上做外积
    for(int i = 0; i < TM; i++) {
        for(int j = 0; j < TN; j++) {
            Ct[i][j] += At[i] * Bt[j];
        }
    }
}
```

#### v5 : 转置存储 + 向量化加载
将 A 转置存储到 As，Bs 保持不变，然后在加载共享内存到寄存器时，使用 float4 向量化加载，减少访存指令数量
源码：[./src/gemm_v5.cu](./src/gemm_v5.cu)
```cpp
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

__shared__ float As[BK][BM]; // 转置存储As

As[ca][i] = (r < M && c < K) ? __half2float(A[r * K + c]) : 0.0f; // As 行列互换

// 向量化加载共享内存
for(int i = 0; i < TM; i += 4) { 
    FLOAT4(At[i]) = FLOAT4(As[k][4 * rc + (i / 4) * 4 * C_BLOCK_ROWS]); 
}
for(int j = 0; j < TN; j += 4) {
    FLOAT4(Bt[j]) = FLOAT4(Bs[k][4 * cc + (j / 4) * 4  * C_BLOCK_COLS]);
}

// 写回时行列计算方式相应改变
for(int i = 0; i < TM; i++) {
    int r = r0 + 4 * rc + (i / 4) * 4 * C_BLOCK_ROWS + i % 4;
    for(int j = 0; j < TN; j++) {
        int c = c0 + 4 * cc + (j / 4) * 4 * C_BLOCK_COLS + j % 4;
        if(r < M && c < N) {
            float c_val = (beta != 0.0f) ? __half2float(C[r * N + c]) * beta : 0.0f;
            C[r * N + c] = __float2half_rn(Ct[i][j] * alpha + c_val);
        }
    } 
}
```

#### v6 : Padding
对 As 添加padding，减少 Bank Conflict
源码：[./src/gemm_v6.cu](./src/gemm_v6.cu)
```cpp

__shared__ float As[BK][BM + 4]; // 转置存储As，并添加padding

```

### Softmax

#### v1 : naive版
朴素 softmax，三次遍历 + 树形规约
源码：[./src/softmax_v1.cu](./src/softmax_v1.cu)
```cpp
__global__ void softmax_kernel_v1(const float* input, float* output, int N) {
    extern __shared__ float smem[];
    
    int tid = threadIdx.x;

    float maxVal = -FLT_MAX;
    for(int i = tid; i < N; i += blockDim.x) {
        maxVal = fmaxf(maxVal, input[i]);
    }
    smem[tid] = maxVal;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    maxVal = smem[0];
    __syncthreads();

    float sumExp = 0.0f;
    for(int i = tid; i < N; i += blockDim.x) {
        sumExp += expf(input[i] - maxVal);
    }
    smem[tid] = sumExp;
    __syncthreads();
    
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    sumExp = smem[0];
    __syncthreads();

    for(int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - maxVal) / sumExp;
    }
}
```

#### v2 : warp 规约
在朴素 softmax 的基础上，用 warp 规约替代树形规约
源码：[./src/softmax_v2.cu](./src/softmax_v2.cu)
```cpp
__device__ float warp_reduce_max(float m) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        m = fmaxf(m, __shfl_down_sync(0xFFFFFFFF, m, offset));
    }
    return m;
}

__device__ float warp_reduce_sum(float s) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xFFFFFFFF, s, offset);
    }
    return s;
}

__global__ void softmax_kernel_v2(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float m = -FLT_MAX;
    for(int i = tid; i < N; i += blockDim.x) {
        m = fmaxf(m, input[i]);
    }

    __shared__ float smem[32];

    m = warp_reduce_max(m);
    if(lane_id == 0) smem[warp_id] = m;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    m = (tid < num_warps) ? smem[tid] : -FLT_MAX;
    if(warp_id == 0) {
        m = warp_reduce_max(m);
        if(lane_id == 0) {
            smem[0] = m;
        }
    }
    __syncthreads();

    m = smem[0];
    __syncthreads();
    float s = 0.0f;
    for(int i = tid; i < N; i += blockDim.x) {
        s += expf(input[i] - m);
    }

    s = warp_reduce_sum(s);
    if(lane_id == 0) smem[warp_id] = s;
    __syncthreads();

    s = (tid < num_warps) ? smem[tid] : 0.0f;
    if(warp_id == 0) {
        s = warp_reduce_sum(s);
        if(lane_id == 0) {
            smem[0] = s;
        }
    }
    __syncthreads();

    s = smem[0];

    for(int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - m) / s;
    }
}
```

#### v3 : online softmax
采用 online softmax 算法，从三次遍历减少为两次遍历
源码：[./src/softmax_v3.cu](./src/softmax_v3.cu)
```cpp
__global__ void softmax_kernel_v3(const float* input, float* output, int N) {
    extern __shared__ float smem[];
    float* smem_max = smem;
    float* smem_sum = smem + blockDim.x;

    int tid = threadIdx.x;

    float m = -FLT_MAX, s = 0.0f;
    for(int i = tid; i < N; i += blockDim.x) {
        float x = input[i];
        float m_new = fmaxf(m, x);
        s = s * expf(m - m_new) + expf(x - m_new);
        m = m_new;
    }
    smem_max[tid] = m;
    smem_sum[tid] = s;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            float m1 = smem_max[tid], m2 = smem_max[tid + s];
            float s1 = smem_sum[tid], s2 = smem_sum[tid + s];
            m = fmaxf(m1, m2);
            smem_max[tid] = m;
            smem_sum[tid] = s1 * expf(m1 - m) + s2 *expf(m2 - m);
        }
        __syncthreads();
    }

    m = smem_max[0];
    s = smem_sum[0];
    for(int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - m) / s;
    }
}
```

#### v4 : online softmax + warp 规约
在 v3 的基础上，用 warp 规约替代树形规约
源码：[./src/softmax_v4.cu](./src/softmax_v4.cu)
```cpp
__device__ void warp_reduce(float& m, float& s) {
    for(int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xFFFFFFFF, m, offset);
        float s2 = __shfl_down_sync(0xFFFFFFFF, s, offset);
        float m_new = fmaxf(m, m2);
        s = s * expf(m - m_new) + s2 * expf(m2 - m_new);
        m = m_new;
    }
}

__global__ void softmax_kernel_v4(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float s = 0.0f;
    float m = -FLT_MAX;
    for(int i = tid; i < N; i += blockDim.x) {
        float x = input[i];
        float m_new = fmaxf(m, x);
        s = s * expf(m - m_new) + expf(x - m_new);
        m = m_new;
    }
    __syncthreads();

    warp_reduce(m, s);

    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];
    if(lane_id == 0) {
        warp_max[warp_id] = m;
        warp_sum[warp_id] = s;
    }
    __syncthreads();
    
    int num_warps = blockDim.x / 32;
    m = (tid < num_warps) ? warp_max[tid] : -FLT_MAX;
    s = (tid < num_warps) ? warp_sum[tid] : 0.0f;
    if(warp_id == 0) {
        warp_reduce(m, s);
        if(lane_id == 0) {
            warp_max[0] = m;
            warp_sum[0] = s;
        }
    };
    __syncthreads();

    m = warp_max[0];
    s = warp_sum[0];
    for(int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - m) / s;
    }
}

```

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
run: test_gemm.sh

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
Running my kernel (v4)...
Correctness checking vs CPU...
Running my kernel (v5)...
Correctness checking vs CPU...
Running my kernel (v6)...
Correctness checking vs CPU...
Running cuBLAS GEMM...

Benchmark analyzing...
Cpu baseline: 9882.5986 ms, 0.22 GFLOPS
My kernel (v1): 6.4332 ms, 333.81 GFLOPS, 1536.2x Speedup
My kernel (v2): 4.0888 ms, 525.21 GFLOPS, 2417.0x Speedup
My kernel (v3): 0.5649 ms, 3801.30 GFLOPS, 17493.4x Speedup
My kernel (v4): 0.5506 ms, 3899.98 GFLOPS, 17947.5x Speedup
My kernel (v5): 0.5365 ms, 4002.63 GFLOPS, 18419.9x Speedup
My kernel (v6): 0.4808 ms, 4466.72 GFLOPS, 20555.6x Speedup
Nvidia's cuBLAS GEMM: 0.3093 ms, 6942.06 GFLOPS, 31947.0x Speedup
My best performance at v6, reach 64.34% of cuBLAS (No Tensor Core)

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
Running my kernel (v6)...
Correctness checking vs CPU...
Running CUB reduce...

Benchmark analyzing...
Cpu baseline: 31.3234 ms, 0.54 GFLOPS, 2.14 GB/s
My kernel (v1): 2.4929 ms, 6.73 GFLOPS, 26.92 GB/s, 12.6x Speedup
My kernel (v2): 1.4588 ms, 11.50 GFLOPS, 46.00 GB/s, 21.5x Speedup
My kernel (v3): 0.7722 ms, 21.73 GFLOPS, 86.91 GB/s, 40.6x Speedup
My kernel (v4): 0.4588 ms, 36.57 GFLOPS, 146.27 GB/s, 68.3x Speedup
My kernel (v5): 0.4109 ms, 40.83 GFLOPS, 163.31 GB/s, 76.2x Speedup
My kernel (v6): 0.2568 ms, 65.32 GFLOPS, 261.30 GB/s, 122.0x Speedup
Nvidia's CUB reduce: 0.2732 ms, 61.42 GFLOPS, 245.67 GB/s, 114.7x Speedup
My best performance at v6, reach 106.36% of CUB

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
Cpu baseline: 235.4198 ms, 0.36 GFLOPS, 1.14 GB/s
My kernel (v1): 71.4434 ms, 1.17 GFLOPS, 3.76 GB/s, 3.3x Speedup
My kernel (v2): 71.4478 ms, 1.17 GFLOPS, 3.76 GB/s, 3.3x Speedup
My kernel (v3): 50.5252 ms, 1.66 GFLOPS, 5.31 GB/s, 4.7x Speedup
My kernel (v4): 50.6299 ms, 1.66 GFLOPS, 5.30 GB/s, 4.6x Speedup
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
Cpu baseline: 6.4289 ms, 1.30 GB/s
My kernel (v1): 0.0630 ms, 133.05 GB/s, 102.0x Speedup
My kernel (v2): 0.0453 ms, 185.16 GB/s, 141.9x Speedup
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
Cpu baseline: 16.0923 ms, 1.04 GFLOPS, 12.51 GB/s
My kernel (v1): 0.7676 ms, 21.86 GFLOPS, 262.28 GB/s, 21.0x Speedup
My kernel (v2): 0.8301 ms, 20.21 GFLOPS, 242.52 GB/s, 19.4x Speedup
My best performance at v1

```
