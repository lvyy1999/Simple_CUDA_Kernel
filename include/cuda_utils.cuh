#pragma once
#include <cmath>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

double compute_gflops(double flops, float ms) {
	double seconds = ms / 1000.0;
	return flops / seconds / 1e9;
}

double compute_gbandwidth(double bytes, float ms) {
	double seconds = ms / 1000.0;
	return bytes / seconds / 1e9;
}

void fill_random_float(float* data, int n, float low = -1.0f, float high = 1.0f) {
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < n; i++) {
        data[i] = dist(gen);
    }
}

void fill_random_half(half* data, int n, float low = -1.0f, float high = 1.0f) {
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < n; i++) {
        data[i] = __float2half_rn(dist(gen));
    }
}

template<typename T>
void check_result(const T* h_C_cpu, const T* h_C_gpu, int N, float eps = 1e-3f) {
	int errors = 0;
    for(int i = 0; i < N; i++) {
        if(fabsf(h_C_cpu[i] - h_C_gpu[i]) > eps) {
            if (errors++ < 5) {
                printf("Mismatch at %d: CPU=%.6f, GPU=%.6f\n", i, (float)h_C_cpu[i], (float)h_C_gpu[i]);
            }
        }
    }
}

void CudaDeviceInfo() {
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",

       deviceId,
       props.multiProcessorCount,
       props.major,
       props.minor,
       props.memoryBusWidth,
       props.maxThreadsPerBlock,
       props.maxThreadsPerMultiProcessor,
       props.totalGlobalMem / 1024 / 1024,
       props.sharedMemPerBlock / 1024,
       props.sharedMemPerMultiprocessor / 1024,
       props.totalConstMem / 1024,
       props.multiProcessorCount,
       props.warpSize);
};
