#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/gemm.cuh"
#include "../include/cuda_check.cuh"
#include "../include/cuda_timer.cuh"
#include "../include/cuda_utils.cuh"

void run_cpu(
    const half* A, 
    const half* B, 
    half* C, 
    int M, 
    int N, 
    int K, 
    float alpha, 
    float beta
) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = __half2float(A[i * K + k]);
                float b = __half2float(B[k * N + j]);
                sum += a * b;
            }
            float c_val = (beta != 0.0f) ? __half2float(C[i * N + j]) * beta : 0.0f;
            C[i * N + j] = __float2half_rn(sum * alpha + c_val);
        }
    }
}

void run_cublas(
    cublasHandle_t handle,
    const half* A,
    const half* B,
    half* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,                  // rows of C^T
        M,                  // cols of C^T
        K,
        &alpha,
        B,
        CUDA_R_16F,
        N,                  // lda for B as column-major B^T
        A,
        CUDA_R_16F,
        K,                  // ldb for A as column-major A^T
        &beta,
        C,
        CUDA_R_16F,
        N,                  // ldc
        CUDA_R_32F,         // accumulate type
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// ============================================================
// main
// ============================================================

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float alpha = 1.0f;
    float beta = 0.0f;
    int warmup = 10; // gpu
    int repeat = 10; // gpu

    printf("\n==================== GEMM test start ====================\n");
    printf("Data type: A/B/C = half, accumulate = float\n");
    printf("Matrix size: A = %d x %d, B = %d x %d, C = %d x %d\n", M, K, K, N, M, N);

    size_t size_A = static_cast<size_t>(M) * K * sizeof(half);
    size_t size_B = static_cast<size_t>(K) * N * sizeof(half);
    size_t size_C = static_cast<size_t>(M) * N * sizeof(half);

    half* h_A = (half*)malloc(size_A);
    half* h_B = (half*)malloc(size_B);
    half* h_C = (half*)malloc(size_C);
    half* h_C_cpu = (half*)malloc(size_C);
    half* h_C_gpu = (half*)malloc(size_C);

    fill_random_half(h_A, M * K);
    fill_random_half(h_B, K * N);
    fill_random_half(h_C, M * N);

    // -------------- CPU ------------------
    printf("Running CPU baseline...\n");
    memcpy(h_C_cpu, h_C, size_C);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    run_cpu(h_A, h_B, h_C_cpu, M, N, K, alpha, beta);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // -------------- GPU ------------------
    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    float gpu_ms[MAX_KERNEL_VERSION];
    for(int i = 0; i < MAX_KERNEL_VERSION; i++) {
        printf("Running my kernel (v%d)...\n", i + 1);
        // warm up
        for(int j = 0; j < warmup; j++) {
            kernel_funcs[i](d_A, d_B, d_C, M, N, K, alpha, beta);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms = 0.0;
        GpuTimer timer;
        for(int j = 0; j < repeat; j++) {
            CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
            timer.start();
            kernel_funcs[i](d_A, d_B, d_C, M, N, K, alpha, beta);
            ms += timer.stop();
        }
        CUDA_CHECK_KERNEL();
        gpu_ms[i] = ms / repeat;

        printf("Correctness checking vs CPU...\n");
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        check_result(h_C_cpu, h_C_gpu, M * N);
    }

    // -------------- cuBLAS ------------------
    printf("Running cuBLAS GEMM...\n");
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    // CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    for(int i = 0; i < warmup; i++) {
        run_cublas(handle, d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    float cublas_ms = 0.0f;
    GpuTimer timer;
    for(int i = 0; i < repeat; i++) {
        CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
        timer.start();
        run_cublas(handle, d_A, d_B, d_C, M, N, K, alpha, beta);
        cublas_ms += timer.stop();
    }
    cublas_ms /= repeat;

    printf("\nBenchmark analyzing...\n");
    double flops = static_cast<double>(2.0) * M * N * K;
    double cpu_gflops = compute_gflops(flops, cpu_ms);
    printf("Cpu baseline: %.4f ms, %.2f GFLOPS\n", 
        cpu_ms, cpu_gflops);
    double cublas_gflops = compute_gflops(flops, cublas_ms);
    printf("Nvidia cuBLAS: %.4f ms, %.2f GFLOPS, %.1fx Speedup vs CPU\n", 
        cublas_ms, cublas_gflops, cpu_ms / cublas_ms);
    int best = 0;
    for(int i = 0; i < MAX_KERNEL_VERSION; i++) {
        if(gpu_ms[i] < gpu_ms[best]) best = i;
        double gpu_gflops = compute_gflops(flops, gpu_ms[i]);
        printf("My kernel (v%d): %.4f ms, %.2f GFLOPS, %.1fx Speedup vs CPU, %.2f%% of cuBLAS (No Tensor Core)\n", 
            i + 1, gpu_ms[i], gpu_gflops, cpu_ms / gpu_ms[i], gpu_gflops / cublas_gflops * 100.0);
    }
    
    double my_best_gflops = compute_gflops(flops, gpu_ms[best]);
    printf("My best performance at v%d, reach %.2f%% of cuBLAS (No Tensor Core)\n", 
        best + 1, my_best_gflops / cublas_gflops * 100.0);

    CUBLAS_CHECK(cublasDestroy(handle));

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
