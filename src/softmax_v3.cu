#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

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

extern "C" void softmax_v3(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    softmax_kernel_v3<<<blocksPerGrid, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(input, output, N);
}
