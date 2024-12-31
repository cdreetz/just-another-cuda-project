cmake_minimum_required(VERSION 3.18)
project(cuda_gemm CUDA CXX)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 86)  # For your GPU architecture

# Enable CUDA and set flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Create executable from main.cu
add_executable(gemm_benchmark src/main.cu)
target_include_directories(gemm_benchmark PRIVATE src)
target_link_libraries(gemm_benchmark PRIVATE CUDA::cudart)

