#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "../../src/common/cuda_utils.cuh"
#include "../../src/common/matrix.cuh"

template<typename T>
bool verify_matrix(const Matrix<T>& actual, const Matrix<T>& expected, T tolerance = 1e-5) {
    actual.to_host();
    
    for (size_t i = 0; i < actual.rows(); ++i) {
        for (size_t j = 0; j < actual.cols(); ++j) {
            T diff = std::abs(actual.host_ptr()[i * actual.cols() + j] - 
                            expected.host_ptr()[i * expected.cols() + j]);
            if (diff > tolerance) {
                std::cout << "Mismatch at (" << i << "," << j << "): "
                         << actual.host_ptr()[i * actual.cols() + j] << " vs "
                         << expected.host_ptr()[i * expected.cols() + j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

template<typename T>
void print_matrix_stats(const Matrix<T>& mat, const char* name) {
    T min_val = mat.host_ptr()[0];
    T max_val = mat.host_ptr()[0];
    T sum = 0;
    
    for (size_t i = 0; i < mat.rows() * mat.cols(); ++i) {
        T val = mat.host_ptr()[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
    }
    
    std::cout << name << " stats:\n"
              << "  Min: " << min_val << "\n"
              << "  Max: " << max_val << "\n"
              << "  Avg: " << sum / (mat.rows() * mat.cols()) << std::endl;
} 