#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/transpose.cuh"
#include "../include/cuda_check.cuh"
#include "../include/cuda_timer.cuh"
#include "../include/cuda_utils.cuh"

void run_cpu(
    const float* A, 
    float* C, 
    int M,
    int N
) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            C[j * M + i] = A[i * N + j];
        }
    }
}

// ============================================================
// main
// ============================================================

int main() {
    int M = 1024;
    int N = 1024;
    int warmup = 10; // gpu
    int repeat = 10; // gpu

    printf("\n==================== Transpose test start ====================\n");
    printf("Data type: float, Matrix size: A = %d × %d, Bytes: %d MB\n", 
        M, N, (M * N) << 2 >> 20);

    size_t size = static_cast<size_t>(M * N) * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    float* h_C_cpu = (float*)malloc(size);
    float* h_C_gpu = (float*)malloc(size);

    fill_random_float(h_A, M * N);
    memset(h_C, 0, size);

    // -------------- CPU ------------------
    printf("Running CPU baseline...\n");
    memcpy(h_C_cpu, h_C, size);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    run_cpu(h_A, h_C_cpu, M, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // -------------- GPU ------------------
    float *d_A, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));
    
    float gpu_ms[MAX_KERNEL_VERSION];
    for(int i = 0; i < MAX_KERNEL_VERSION; i++) {
        printf("Running my kernel (v%d)...\n", i + 1);
        // warm up
        for(int j = 0; j < warmup; j++) {
            kernel_funcs[i](d_A, d_C, M, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

        float ms = 0.0;
        GpuTimer timer;
        for(int j = 0; j < repeat; j++) {
            timer.start();
            kernel_funcs[i](d_A, d_C, M, N);
            ms += timer.stop();
        }
        CUDA_CHECK_KERNEL();
        gpu_ms[i] = ms / repeat;

        printf("Correctness checking vs CPU...\n");
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        check_result(h_C_cpu, h_C_gpu, M * N);
    }

    printf("\nBenchmark analyzing...\n");
    double bytes = static_cast<double>(size) * 2.0; // read one, write one
    double cpu_bandwidth = compute_gbandwidth(bytes, cpu_ms);
    printf("Cpu baseline: %.4f ms, %.2f GB/s\n", 
        cpu_ms, cpu_bandwidth);
    int best = 0;
    for(int i = 0; i < MAX_KERNEL_VERSION; i++) {
        if(gpu_ms[i] < gpu_ms[best]) best = i;
        double gpu_bandwidth = compute_gbandwidth(bytes, gpu_ms[i]);;
        printf("My kernel (v%d): %.4f ms, %.2f GB/s, %.1fx Speedup\n", 
            i + 1, gpu_ms[i], gpu_bandwidth, cpu_ms / gpu_ms[i]);
    }
    printf("My best performance at v%d\n", best + 1);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_C);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
