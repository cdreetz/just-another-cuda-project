gemm-optimizations/
├── CMakeLists.txt # Main CMake configuration
├── .gitignore # Git ignore file
├── README.md # Project documentation
├── scripts/
│ ├── benchmark.sh # Benchmark automation script
│ ├── profile.sh # NVIDIA profiler automation
│ └── test_all.sh # Run all tests
├── src/
│ ├── kernels/
│ │ ├── gemm_v1.cu # Basic GEMM implementation
│ │ ├── gemm_v2.cu # With shared memory
│ │ ├── gemm_v3.cu # With register blocking
│ │ └── gemm_utils.cuh # Common CUDA utilities
│ ├── common/
│ │ ├── cuda_utils.cuh # Error checking, timing
│ │ ├── matrix.cuh # Matrix operations
│ │ └── memory.cuh # Memory management
│ └── main.cu # Main benchmarking program
├── tests/
│ ├── test_utils/
│ │ ├── test_helpers.cuh
│ │ └── matrix_generators.cuh
│ └── kernels/
│ ├── test_gemm_v1.cu
│ ├── test_gemm_v2.cu
│ └── test_gemm_v3.cu
├── benchmarks/
│ ├── baseline/
│ │ ├── cublas_gemm.cu # cuBLAS reference
│ │ └── cutlass_gemm.cu # CUTLASS reference
│ └── results/
│ └── .gitkeep # Keep empty dir for results
└── tools/
├── visualization/ # Scripts for visualizing results
│ ├── plot_performance.py
│ └── roofline_model.py
└── profiling/ # Custom profiling tools
└── memory_analyzer.py
