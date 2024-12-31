#!/bin/bash

# Directory setup
BUILD_DIR="build"
RESULTS_DIR="benchmarks/results"
mkdir -p "$RESULTS_DIR"

# Build the project
cmake -B "$BUILD_DIR" -S .
cmake --build "$BUILD_DIR" -j$(nproc)

# Run benchmarks for different matrix sizes
SIZES=(1024 2048 4096 8192)
OUTPUT_FILE="$RESULTS_DIR/benchmark_results_$(date +%Y%m%d_%H%M%S).csv"

echo "Size,Version,GFLOPS,Time(ms)" > "$OUTPUT_FILE"

for size in "${SIZES[@]}"; do
    echo "Running benchmark for size $size x $size..."
    "$BUILD_DIR/gemm_benchmark" "$size" >> "$OUTPUT_FILE"
done

echo "Benchmarks complete. Results saved to $OUTPUT_FILE" 