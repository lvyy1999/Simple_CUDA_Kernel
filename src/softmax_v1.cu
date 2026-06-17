#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

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

extern "C" void softmax_v1(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    softmax_kernel_v1<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, N);
}
