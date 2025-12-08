cmake_minimum_required(VERSION 3.10)

# Ensure CUDA arch selection works even when the driver cannot report a default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    if(DEFINED ENV{SODA_CUDA_ARCHS} AND NOT "$ENV{SODA_CUDA_ARCHS}" STREQUAL "")
        set(_soda_cuda_archs "$ENV{SODA_CUDA_ARCHS}")
        message(STATUS "Using CUDA arch list from SODA_CUDA_ARCHS=${_soda_cuda_archs}")
    else()
        set(_soda_cuda_archs "70;75;80;86;89;90")
        message(STATUS "CMAKE_CUDA_ARCHITECTURES not provided, defaulting to ${_soda_cuda_archs}")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_soda_cuda_archs}" CACHE STRING "" FORCE)
    unset(_soda_cuda_archs)
endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER OR NOT CMAKE_CUDA_COMPILER)
    set(_cuda_candidate_paths "")
    if(DEFINED ENV{CUDACXX})
        list(APPEND _cuda_candidate_paths "$ENV{CUDACXX}")
    endif()
    if(DEFINED ENV{CUDA_HOME})
        list(APPEND _cuda_candidate_paths "$ENV{CUDA_HOME}/bin/nvcc")
    endif()
    list(APPEND _cuda_candidate_paths
        /usr/local/cuda/bin/nvcc
        /usr/local/cuda-12.3/bin/nvcc
        /usr/local/cuda-12.2/bin/nvcc
        /usr/local/cuda-12.1/bin/nvcc
        /usr/local/cuda-12.0/bin/nvcc
        /usr/local/cuda-11.8/bin/nvcc
        /usr/local/cuda-11/bin/nvcc
    )
    foreach(candidate ${_cuda_candidate_paths})
        if(EXISTS "${candidate}")
            set(CMAKE_CUDA_COMPILER "${candidate}" CACHE FILEPATH "Path to nvcc" FORCE)
            message(STATUS "Using CUDA compiler at ${candidate}")
            break()
        endif()
    endforeach()
    unset(_cuda_candidate_paths)
endif()

if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "Could not find nvcc. Set CUDACXX or CMAKE_CUDA_COMPILER to the CUDA compiler path.")
endif()

project(baremetal_gemm CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Build main binary (compile as CUDA to support kernel launches)
set(BM_SOURCES
    main_gemm_bm.cpp
)

add_executable(main_gemm_bm ${BM_SOURCES})
set_source_files_properties(main_gemm_bm.cpp PROPERTIES LANGUAGE CUDA)
set_property(TARGET main_gemm_bm PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA libraries
target_link_libraries(main_gemm_bm
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::nvToolsExt
)

# Include CUDA headers
target_include_directories(main_gemm_bm PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Formatting target (clang-format)
find_program(CLANG_FORMAT_BIN clang-format)
if(CLANG_FORMAT_BIN)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_BIN} -i ${BM_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running clang-format on baremetal sources"
    )
    add_dependencies(main_gemm_bm format)
else()
    message(WARNING "clang-format not found; skipping format target")
endif()
