#pragma once
#include <random>
#include "../../src/common/matrix.cuh"

template<typename T>
class MatrixGenerator {
public:
    static void random_uniform(Matrix<T>& mat, T min = -1.0, T max = 1.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min, max);
        
        for (size_t i = 0; i < mat.rows(); ++i) {
            for (size_t j = 0; j < mat.cols(); ++j) {
                mat.host_ptr()[i * mat.cols() + j] = dis(gen);
            }
        }
    }
    
    static void identity(Matrix<T>& mat) {
        std::fill(mat.host_ptr(), mat.host_ptr() + mat.rows() * mat.cols(), 0);
        for (size_t i = 0; i < std::min(mat.rows(), mat.cols()); ++i) {
            mat.host_ptr()[i * mat.cols() + i] = 1;
        }
    }
    
    static void constant(Matrix<T>& mat, T value) {
        std::fill(mat.host_ptr(), mat.host_ptr() + mat.rows() * mat.cols(), value);
    }
}; 