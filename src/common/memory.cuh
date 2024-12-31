#pragma once
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

template<typename T>
class DeviceMemory {
public:
    static T* allocate(size_t size) {
        T* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
        return ptr;
    }

    static void free(T* ptr) {
        if (ptr) CUDA_CHECK(cudaFree(ptr));
    }

    static void memset(T* ptr, int value, size_t count) {
        CUDA_CHECK(cudaMemset(ptr, value, count * sizeof(T)));
    }
};

template<typename T>
class PinnedMemory {
public:
    static T* allocate(size_t size) {
        T* ptr;
        CUDA_CHECK(cudaMallocHost(&ptr, size * sizeof(T)));
        return ptr;
    }

    static void free(T* ptr) {
        if (ptr) CUDA_CHECK(cudaFreeHost(ptr));
    }
}; 