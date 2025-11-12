cmake_minimum_required(VERSION 3.10)
project(baremetal_gemm CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Build main binary
add_executable(main_gemm_bm main_gemm_bm.cpp)

# Link CUDA libraries
target_link_libraries(main_gemm_bm
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)

# Include CUDA headers
target_include_directories(main_gemm_bm PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

