#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"

class GpuTimer {
private:
    cudaStream_t stream;
    cudaEvent_t start_event, stop_event;

public:
    explicit GpuTimer(cudaStream_t stream_ = 0) : stream(stream_) {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }

    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float elapsed_time = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
        return elapsed_time;
    }
};
