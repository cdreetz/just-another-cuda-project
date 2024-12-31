#include <cuda_runtime.h>
#include "../common/cuda_utils.cuh"
#include "../common/matrix.cuh"

// basic gemm
template<typename T>
__global__ void gemm_v0_kernel(
    size_t m, size_t n, size_t k, T alpha, T const* A,
    size_t lda, T const* B, size_t ldb, T beta, T* C,
    size_t ldc) {
    
    // calculate global thread indices
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // check if thread is within bounds
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum = {static_cast<T>(0)};
        // compute single element of C
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        // write the result
        C[C_row_idx * ldc + C_col_idx] = 
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

// host function to launch kernel
template<typename T>
void gemm_v0(size_t m, size_t n, size_t k, T const* alpha,
    T const* A, size_t lda, T const* B, size_t ldb,
    T const* beta, T* C, size_t ldc,
    cudaStream_t stream)
{
    dim3 block_dim{32U, 32U, 1U};
    dim3 grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y,
        1U};
    
    gemm_v0_kernel<T><<<grid_dim, block_dim, 0U, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);

    CUDA_CHECK();
}

// Explicit template instantiations
template void gemm_v0<float>(
    size_t m, size_t n, size_t k, float const* alpha,
    float const* A, size_t lda, float const* B, size_t ldb,
    float const* beta, float* C, size_t ldc,
    cudaStream_t stream);

template void gemm_v0<double>(
    size_t m, size_t n, size_t k, double const* alpha,
    double const* A, size_t lda, double const* B, size_t ldb,
    double const* beta, double* C, size_t ldc,
    cudaStream_t stream); 
