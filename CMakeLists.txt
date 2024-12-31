cmake_minimum_required(VERSION 3.18)
project(gemm_optimizations CUDA CXX)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 80) # Adjust based on your GPU

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find required packages
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Add source directories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
# Main executable
cuda_add_executable(gemm_benchmark
    src/main.cu
)

# Link against CUDA libraries
target_link_libraries(gemm_benchmark
    PRIVATE
    CUDA::cublas
    CUDA::cudart
)

# Testing
enable_testing()
add_test(NAME gemm_tests COMMAND test_gemm)

# Install rules
install(TARGETS gemm_benchmark
    RUNTIME DESTINATION bin
)

# Custom targets for benchmarking and profiling
add_custom_target(run_benchmarks
    COMMAND ${CMAKE_BINARY_DIR}/gemm_benchmark
    DEPENDS gemm_benchmark
)

add_custom_target(profile
    COMMAND nvprof --profile-from-start off ${CMAKE_BINARY_DIR}/gemm_benchmark
    DEPENDS gemm_benchmark
)

