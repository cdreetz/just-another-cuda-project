#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

// Error checking macro
#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                               \
        printf("CUDA error at %s:%d: %s\n",                 \
               __FILE__, __LINE__,                          \
               cudaGetErrorString(err));                    \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

// Timer class for kernel profiling
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }

    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }

private:
    cudaEvent_t start_, stop_;
}; 