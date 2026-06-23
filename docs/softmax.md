## Softmax

### v1 : naive版

朴素 softmax，三次遍历 + 树形规约
源码：[softmax_v1.cu](../src/softmax_v1.cu)

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

### v2 : warp 规约

在朴素 softmax 的基础上，用 warp 规约替代树形规约
源码：[softmax_v2.cu](../src/softmax_v2.cu)

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

### v3 : online softmax

采用 online softmax 算法，从三次遍历减少为两次遍历
源码：[softmax_v3.cu](../src/softmax_v3.cu)

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

### v4 : online softmax + warp 规约

在 v3 的基础上，用 warp 规约替代树形规约
源码：[softmax_v4.cu](../src/softmax_v4.cu)

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
