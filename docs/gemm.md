## GEMM

### v1 : naive版

源码：[gemm_v1.cu](../src/gemm_v1.cu)

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

### v2 : block 级分块

将矩阵A和B分块读入共享内存，然后在共享内存中计算矩阵乘，增加数据复用，减少全局内存访问
源码：[gemm_v2.cu](../src/gemm_v2.cu)

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

### v3 : thread 级分块

每个线程负责更多数据，在寄存器上存储中间结果
源码：[gemm_v3.cu](../src/gemm_v3.cu)

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

### v4 : 外积分解 + 寄存器级复用

先将共享内存数据搬运到寄存器，再在寄存器上做外积，增加数据复用
源码：[gemm_v4.cu](../src/gemm_v4.cu)

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

### v5 : 转置存储 + 向量化加载

将 A 转置存储到 As，Bs 保持不变，然后在加载共享内存到寄存器时，使用 float4 向量化加载，减少访存指令数量
源码：[gemm_v5.cu](../src/gemm_v5.cu)

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

### v6 : Padding

对 As 添加padding，减少 Bank Conflict
源码：[gemm_v6.cu](../src/gemm_v6.cu)

```cpp

__shared__ float As[BK][BM + 4]; // 转置存储As，并添加padding

```
