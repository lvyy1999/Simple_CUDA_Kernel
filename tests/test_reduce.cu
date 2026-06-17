#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/reduce.cuh"
#include "../include/cuda_check.cuh"
#include "../include/cuda_timer.cuh"
#include "../include/cuda_utils.cuh"

void run_cpu(
    const float* input,  
    float* output, 
    int N
) {
    float sum = 0.0f;
    for(int i = 0; i < N; i++) {
        sum += input[i];
    }
    *output = sum;
}

void check_reduce_result(float cpu, float gpu) {
    float abs_err = fabs(cpu - gpu);
    float rel_err = abs_err / (fabs(cpu) + 1e-6f);
    if(abs_err >= 1e-2f && rel_err >= 1e-3f) {
        printf("Result mismatch: CPU=%.6f, GPU=%.6f\n", cpu, gpu);
    }
}

// ============================================================
// main
// ============================================================

int main() {
    int N = 1 << 24;
    int warmup = 10; // gpu
    int repeat = 10; // gpu

    printf("\n==================== Reduce test start ====================\n");
    printf("Data type: float, Vector size: N = %d, Bytes: %d MB\n", 
        N, N << 2 >> 20);

    size_t size = static_cast<size_t>(N) * sizeof(float);

    float* h_input = (float*)malloc(size);
    float h_output_cpu = 0.0f, h_output_gpu = 0.0f;

    fill_random_float(h_input, N);

    // -------------- CPU ------------------
    printf("Running CPU baseline...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    run_cpu(h_input, &h_output_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // -------------- GPU ------------------
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    float gpu_ms[MAX_KERNEL_VERSION];
    for(int i = 0; i < MAX_KERNEL_VERSION; i++) {
        printf("Running my kernel (v%d)...\n", i + 1);
        // warm up
        for(int j = 0; j < warmup; j++) {
            CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
            kernel_funcs[i](d_input, d_output, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms = 0.0;
        GpuTimer timer;
        for(int j = 0; j < repeat; j++) {
            CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float))); // reset output
            timer.start();
            kernel_funcs[i](d_input, d_output, N);
            ms += timer.stop();
        }
        CUDA_CHECK_KERNEL();
        gpu_ms[i] = ms / repeat;

        printf("Correctness checking vs CPU...\n");
        CUDA_CHECK(cudaMemcpy(&h_output_gpu, d_output, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        check_reduce_result(h_output_cpu, h_output_gpu);
    }

    printf("\nBenchmark analyzing...\n");
    double flops = static_cast<double>(N);
    double bytes = static_cast<double>(size);
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

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    free(h_input);

    return 0;
}