#include <iostream>
#include "common/cuda_utils.cuh"
#include "common/matrix.cuh"
#include "kernels/gemm_v1.cu"

int main(int argc, char** argv) {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    Matrix<float> A(M, K);
    Matrix<float> B(K, N);
    Matrix<float> C(M, N);

    // Initialize matrices (you'll want to add proper initialization)
    
    // Transfer to device
    A.to_device();
    B.to_device();

    // Time the kernel
    CudaTimer timer;
    timer.start();
    
    gemm_v1(A, B, C);
    
    float ms = timer.stop();
    
    // Calculate and print performance metrics
    double flops = 2.0 * M * N * K;
    double gflops = (flops * 1e-9) / (ms * 1e-3);
    
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Time: " << ms << " ms" << std::endl;

    return 0;
} 