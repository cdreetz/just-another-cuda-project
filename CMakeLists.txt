cmake_minimum_required(VERSION 3.18)
project(cuda_gemm CUDA CXX)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 86)  # For Pascal architecture, adjust as needed

# Enable CUDA and set flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)

# Create benchmark executable
add_executable(gemm_benchmark benchmarks/benchmark_gemm.cu)
target_link_libraries(gemm_benchmark 
    PRIVATE 
    gemm_lib
    CUDA::cudart
)

