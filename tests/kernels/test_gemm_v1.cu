#include <gtest/gtest.h>
#include "../../src/kernels/gemm_v1.cu"
#include "../test_utils/test_helpers.cuh"
#include "../test_utils/matrix_generators.cuh"

TEST(GemmV1Test, SmallMatrixMultiplication) {
    const int M = 32;
    const int N = 32;
    const int K = 32;
    
    Matrix<float> A(M, K);
    Matrix<float> B(K, N);
    Matrix<float> C(M, N);
    Matrix<float> C_expected(M, N);
    
    // Initialize matrices
    MatrixGenerator<float>::random_uniform(A);
    MatrixGenerator<float>::random_uniform(B);
    
    // Compute expected result on CPU
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A.host_ptr()[i * K + k] * B.host_ptr()[k * N + j];
            }
            C_expected.host_ptr()[i * N + j] = sum;
        }
    }
    
    // Transfer to device and compute
    A.to_device();
    B.to_device();
    
    gemm_v1(A, B, C);
    
    // Verify results
    EXPECT_TRUE(verify_matrix(C, C_expected));
} 