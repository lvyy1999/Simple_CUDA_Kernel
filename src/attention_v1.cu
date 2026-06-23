#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

__global__ void qkt_kernel(const float* Q, const float* K, float* S, int M, int N, int d) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(row >= M || col >= N) return;

    float sum = 0.0f;
    for(int k = 0; k < d; k++) {
        sum += Q[row * d + k] * K[col * d + k];
    }
    S[row * N + col] = sum / sqrtf(d);
}

__global__ void softmax_kernel(const float* S, float* P, int M, int N) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    if(row >= M) return;
    
    int tid = threadIdx.x;

    float maxVal = -FLT_MAX;
    for(int j = tid; j < N; j += blockDim.x) {
        maxVal = fmaxf(maxVal, S[row * N + j]);
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
    for(int j = tid; j < N; j += blockDim.x) {
        sumExp += expf(S[row * N + j] - maxVal);
    }
    smem[tid] = sumExp;
    __syncthreads();
    
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    sumExp = smem[0];
    __syncthreads();

    for(int j = tid; j < N; j += blockDim.x) {
        P[row * N + j] = expf(S[row * N + j] - maxVal) / sumExp;
    }
}

__global__ void pv_kernel(const float* P, const float* V, float* output, int M, int N, int d) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(row >= M || col >= d) return;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += P[row * N + k] * V[k * d + col];
    }
    output[row * d + col] = sum;
}

extern "C" void attention_v1(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *S, *P;
    cudaMalloc(&S, M * N * sizeof(float));
    cudaMalloc(&P, M * N * sizeof(float));

    dim3 block_qkt(16, 16);
    dim3 grid_qkt((N + block_qkt.x - 1) / block_qkt.x,
                  (M + block_qkt.y - 1) / block_qkt.y);
    qkt_kernel<<<grid_qkt, block_qkt>>>(Q, K, S, M, N, d);
    cudaDeviceSynchronize();

    softmax_kernel<<<M, 256, 256 * sizeof(float)>>>(S, P, M, N);
    cudaDeviceSynchronize();

    dim3 block_pv(16, 16);
    dim3 grid_pv((d + block_pv.x - 1) / block_pv.x,
                 (M + block_pv.y - 1) / block_pv.y);
    pv_kernel<<<grid_pv, block_pv>>>(P, V, output, M, N, d);
    cudaDeviceSynchronize();

    cudaFree(S);
    cudaFree(P);
}
