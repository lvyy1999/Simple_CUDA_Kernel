## Reduce

### v1 : naive版

采用树形规约，最后每个 block 内的 thread 0 将 block 内的规约结果原子加到输出
源码：[reduce_v1.cu](../src/reduce_v1.cu)

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

### v2 : 反转步长

反转规约时的步长方向，减少 Warp Divergence 和 Bank Conflict
源码：[reduce_v2.cu](../src/reduce_v2.cu)

```cpp
for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) {
        smem[tid] += smem[tid + s];
    }
    __syncthreads();
}
```

### v3 : 减少空闲线程

每个线程负责读取两个数据，block 数量减半，大幅减少空闲线程
源码：[reduce_v3.cu](../src/reduce_v3.cu)

```cpp
smem[tid] = (idx < N) ? input[idx] : 0.0f;
smem[tid] += (idx + blockDim.x * gridDim.x < N) ? input[idx + blockDim.x * gridDim.x] : 0.0f;

int blocksPerGrid = (N + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock);
```

### v4 : 展开最后一个 warp

树形规约只处理到步长大于32的部分，当步长小于等于32时，活跃线程只剩第一个 warp 内的线程，改用 warp内的同步原语进行规约，更高效且无需全局线程等待
源码：[reduce_v4.cu](../src/reduce_v4.cu)

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

### v5 : warp shuffle

全部使用 warp shuffle 替代朴素树形规约，由于 warp 数量最大为32(1024 / 32 = 32)，开辟大小为32的共享内存 smem，第 i 个 warp 将 warp 内的规约结果写到 smem[i]，再由第一个 warp 对 smem 进行规约
源码：[reduce_v5.cu](../src/reduce_v5.cu)

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

### v6 : Grid Stride Loop

采用固定 grid 大小，每个线程采用跳步循环覆盖整个数组
源码：[reduce_v6.cu](../src/reduce_v6.cu)

```cpp 
float val = 0.0f;
int stride = blockDim.x * gridDim.x;
for(int i = idx; i < N; i += stride) {
    val += input[i];
}
```
