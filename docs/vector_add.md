## Vector_Add

### v1 : naive版

源码：[vector_add_v1.cu](../src/vector_add_v1.cu)

```cpp
__global__ void vector_add_kernel_v1(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

## v2 : 向量化加载

源码：[vector_add_v2.cu](../src/vector_add_v2.cu)

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