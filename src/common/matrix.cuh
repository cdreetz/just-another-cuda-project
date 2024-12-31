#pragma once
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

template<typename T>
class Matrix {
public:
    Matrix(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols), stride_(cols) {
        CUDA_CHECK(cudaMallocHost(&host_data_, sizeof(T) * rows * cols));
        CUDA_CHECK(cudaMalloc(&device_data_, sizeof(T) * rows * cols));
    }

    ~Matrix() {
        if (host_data_) CUDA_CHECK(cudaFreeHost(host_data_));
        if (device_data_) CUDA_CHECK(cudaFree(device_data_));
    }

    void to_device() {
        CUDA_CHECK(cudaMemcpy(device_data_, host_data_, 
                             sizeof(T) * rows_ * cols_, 
                             cudaMemcpyHostToDevice));
    }

    void to_host() {
        CUDA_CHECK(cudaMemcpy(host_data_, device_data_,
                             sizeof(T) * rows_ * cols_,
                             cudaMemcpyDeviceToHost));
    }

    T* host_ptr() { return host_data_; }
    T* device_ptr() { return device_data_; }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t stride() const { return stride_; }

private:
    size_t rows_, cols_, stride_;
    T* host_data_ = nullptr;
    T* device_data_ = nullptr;
}; 