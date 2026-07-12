#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/vector_add.cuh"
#include "../include/cuda_check.cuh"
#include "../include/cuda_timer.cuh"
#include "../include/cuda_utils.cuh"

void run_cpu(
    const float* A, 
    const float* B, 
    float* C, 
    int N
) {
    for(int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================
// main
// ============================================================

int main() {
    int N = 1 << 26;
    int warmup = 10; // gpu
    int repeat = 10; // gpu

    printf("\n==================== Vector add test start ====================\n");
    printf("Data type: float, Vector size: N = %d, Bytes: %d MB\n", 
        N, N >> 20 << 2);

    size_t size = static_cast<size_t>(N) * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    float* h_C_cpu = (float*)malloc(size);
    float* h_C_gpu = (float*)malloc(size);

    fill_random_float(h_A, N);
    fill_random_float(h_B, N);
    memset(h_C, 0, size);

    // -------------- CPU ------------------
    printf("Running CPU baseline...\n");
    memcpy(h_C_cpu, h_C, size);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    run_cpu(h_A, h_B, h_C_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // -------------- GPU ------------------
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));
    
    float gpu_ms[MAX_KERNEL_VERSION];
    for(int i = 0; i < MAX_KERNEL_VERSION; i++) {
        printf("Running my kernel (v%d)...\n", i + 1);
        // warm up
        for(int j = 0; j < warmup; j++) {
            kernel_funcs[i](d_A, d_B, d_C, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

        float ms = 0.0;
        GpuTimer timer;
        for(int j = 0; j < repeat; j++) {
            timer.start();
            kernel_funcs[i](d_A, d_B, d_C, N);
            ms += timer.stop();
        }
        CUDA_CHECK_KERNEL();
        gpu_ms[i] = ms / repeat;

        printf("Correctness checking vs CPU...\n");
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        check_result(h_C_cpu, h_C_gpu, N);
    }

    printf("\nBenchmark analyzing...\n");
    double flops = static_cast<double>(N);
    double bytes = static_cast<double>(size) * 3.0; // read two, write one
    double cpu_gflops = compute_gflops(flops, cpu_ms);
    double cpu_bandwidth = compute_gbandwidth(bytes, cpu_ms);
    printf("Cpu baseline: %.4f ms, %.2f GFLOPS, %.2f GB/s\n", 
        cpu_ms, cpu_gflops, cpu_bandwidth);
    int best = 0;
    for(int i = 0; i < MAX_KERNEL_VERSION; i++) {
        if(gpu_ms[i] < gpu_ms[best]) best = i;
        double gpu_gflops = compute_gflops(flops, gpu_ms[i]);
        double gpu_bandwidth = compute_gbandwidth(bytes, gpu_ms[i]);;
        printf("My kernel (v%d): %.4f ms, %.2f GFLOPS, %.2f GB/s, %.1fx Speedup\n", 
            i + 1, gpu_ms[i], gpu_gflops, gpu_bandwidth, cpu_ms / gpu_ms[i]);
    }
    printf("My best performance at v%d\n", best + 1);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}