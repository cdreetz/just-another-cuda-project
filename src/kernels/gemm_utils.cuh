#pragma once
#include <cuda_runtime.h>

// Common GEMM kernel utilities
namespace gemm {

// Thread block tile dimensions
constexpr int BLOCK_SIZE = 32;
constexpr int WARP_SIZE = 32;

// Shared memory padding to avoid bank conflicts
constexpr int SMEM_PADDING = 8;

// Helper function to compute grid dimensions
inline dim3 calc_grid_dim(int M, int N, int block_size) {
    return dim3(
        (N + block_size - 1) / block_size,
        (M + block_size - 1) / block_size
    );
}

// Helper for shared memory allocation
template<typename T>
inline size_t calc_shared_memory_size(int block_size) {
    return (block_size + SMEM_PADDING) * block_size * sizeof(T);
}

} // namespace gemm 