#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <float.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/attention.cuh"
#include "../include/cuda_check.cuh"
#include "../include/cuda_timer.cuh"
#include "../include/cuda_utils.cuh"

void run_cpu(
    const float* Q, 
    const float* K, 
    const float* V, 
    float* output, 
    int M, 
    int N, 
    int d
) {
    float* S = (float*)malloc(M * N * sizeof(float));
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for(int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum / sqrtf(d);
        }
    }

    float* P = (float*)malloc(M * N * sizeof(float));
    for(int i = 0; i < M; i++) {
        float maxVal = -FLT_MAX;
        for(int j = 0; j < N; j++) {
            maxVal = fmaxf(maxVal, S[i * N + j]);
        }

        float sumExp = 0.0f;
        for(int j = 0; j < N; j++) {
            sumExp += expf(S[i * N + j] - maxVal);
        }

        for(int j = 0; j < N; j++) {
            P[i * N + j] = expf(S[i * N + j] - maxVal) / sumExp;
        }
    }

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < d; j++) {
            float sum = 0.0f;
            for(int k = 0; k < N; k++) {
                sum += P[i * N + k] * V[k * d + j];
            }
            output[i * d + j] = sum;
        }
    }  

    free(S);
    free(P);
}

// ============================================================
// main
// ============================================================

int main() {
    int M = 4096;
    int N = 4096;
    int d = 128;
    int warmup = 10; // gpu
    int repeat = 10; // gpu

    printf("\n==================== Attention test start ====================\n");
    printf("Matrix size: Q = %d x %d, K = %d x %d, V = %d x %d, output = %d × %d\n", M, d, N, d, N, d, M, d);

    size_t size_Q = static_cast<size_t>(M) * d * sizeof(float);
    size_t size_K = static_cast<size_t>(N) * d * sizeof(float);
    size_t size_V = static_cast<size_t>(N) * d * sizeof(float);
    size_t size_output = static_cast<size_t>(M) * d * sizeof(float);

    float* h_Q = (float*)malloc(size_Q);
    float* h_K = (float*)malloc(size_K);
    float* h_V = (float*)malloc(size_V);
    float* h_output_cpu = (float*)malloc(size_output);
    float* h_output_gpu = (float*)malloc(size_output);

    fill_random_float(h_Q, M * d);
    fill_random_float(h_K, N * d);
    fill_random_float(h_V, N * d);

    // -------------- CPU ------------------
    printf("Running CPU baseline...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    run_cpu(h_Q, h_K, h_V, h_output_cpu, M, N, d);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // -------------- GPU ------------------
    float *d_Q, *d_K, *d_V, *d_output;
    CUDA_CHECK(cudaMalloc(&d_Q, size_Q));
    CUDA_CHECK(cudaMalloc(&d_K, size_K));
    CUDA_CHECK(cudaMalloc(&d_V, size_V));
    CUDA_CHECK(cudaMalloc(&d_output, size_output));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, size_Q, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, size_K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, size_V, cudaMemcpyHostToDevice));

    float gpu_ms[MAX_KERNEL_VERSION];
    for(int i = 0; i < MAX_KERNEL_VERSION; i++) {
        printf("Running my kernel (v%d)...\n", i + 1);
        // warm up
        for(int j = 0; j < warmup; j++) {
            kernel_funcs[i](d_Q, d_K, d_V, d_output, M, N, d);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms = 0.0;
        GpuTimer timer;
        for(int j = 0; j < repeat; j++) {
            timer.start();
            kernel_funcs[i](d_Q, d_K, d_V, d_output, M, N, d);
            ms += timer.stop();
        }
        CUDA_CHECK_KERNEL();
        gpu_ms[i] = ms / repeat;

        printf("Correctness checking vs CPU...\n");
        CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, size_output, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        check_result(h_output_cpu, h_output_gpu, M * d);
    }

    printf("\nBenchmark analyzing...\n");
    double flops = static_cast<double>(4 * M * N * d + 6 * M * N);
    double cpu_gflops = compute_gflops(flops, cpu_ms);
    printf("Cpu baseline: %.4f ms, %.2f GFLOPS\n", 
        cpu_ms, cpu_gflops);
    int best = 0;
    for(int i = 0; i < MAX_KERNEL_VERSION; i++) {
        if(gpu_ms[i] < gpu_ms[best]) best = i;
        double gpu_gflops = compute_gflops(flops, gpu_ms[i]);
        printf("My kernel (v%d): %.4f ms, %.2f GFLOPS, %.1fx Speedup\n", 
            i + 1, gpu_ms[i], gpu_gflops, cpu_ms / gpu_ms[i]);
    }
    printf("My best performance at v%d\n", best + 1);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_output));

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_output_cpu);
    free(h_output_gpu);

    return 0;
}
