#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublasLt.h>

// Helper macros for error checking
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLASLT(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLASLt error at " << __FILE__ << ":" << __LINE__ \
                      << " - status " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct GemmParams {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    std::string order_a;  // "row" or "col"
    std::string order_b;  // "row" or "col"
    char trans_a;
    char trans_b;
    std::string dtype;
    float alpha;
    float beta;
    int warmup;
    int runs;
    int batch;  // For bmm, default 1
};

void parse_args(int argc, char** argv, GemmParams& params) {
    // Defaults
    params.m = 0;
    params.n = 0;
    params.k = 0;
    params.lda = 0;
    params.ldb = 0;
    params.ldc = 0;
    params.order_a = "row";
    params.order_b = "row";
    params.trans_a = 'N';
    params.trans_b = 'N';
    params.dtype = "f32";
    params.alpha = 1.0f;
    params.beta = 0.0f;
    params.warmup = 200;
    params.runs = 1000;
    params.batch = 1;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--m" && i+1 < argc) {
            params.m = std::atoi(argv[++i]);
        } else if (arg == "--n" && i+1 < argc) {
            params.n = std::atoi(argv[++i]);
        } else if (arg == "--k" && i+1 < argc) {
            params.k = std::atoi(argv[++i]);
        } else if (arg == "--lda" && i+1 < argc) {
            params.lda = std::atoi(argv[++i]);
        } else if (arg == "--ldb" && i+1 < argc) {
            params.ldb = std::atoi(argv[++i]);
        } else if (arg == "--ldc" && i+1 < argc) {
            params.ldc = std::atoi(argv[++i]);
        } else if (arg == "--order_a" && i+1 < argc) {
            params.order_a = argv[++i];
        } else if (arg == "--order_b" && i+1 < argc) {
            params.order_b = argv[++i];
        } else if (arg == "--trans_a" && i+1 < argc) {
            params.trans_a = argv[++i][0];
        } else if (arg == "--trans_b" && i+1 < argc) {
            params.trans_b = argv[++i][0];
        } else if (arg == "--dtype" && i+1 < argc) {
            params.dtype = argv[++i];
        } else if (arg == "--alpha" && i+1 < argc) {
            params.alpha = std::atof(argv[++i]);
        } else if (arg == "--beta" && i+1 < argc) {
            params.beta = std::atof(argv[++i]);
        } else if (arg == "--warmup" && i+1 < argc) {
            params.warmup = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i+1 < argc) {
            params.runs = std::atoi(argv[++i]);
        } else if (arg == "--batch" && i+1 < argc) {
            params.batch = std::atoi(argv[++i]);
        }
    }
    
    // Validation
    if (params.m <= 0 || params.n <= 0 || params.k <= 0) {
        std::cerr << "Error: M, N, K must be positive" << std::endl;
        exit(EXIT_FAILURE);
    }
}

cudaDataType_t get_cuda_dtype(const std::string& dtype) {
    if (dtype == "f32") return CUDA_R_32F;
    if (dtype == "f16") return CUDA_R_16F;
    if (dtype == "f64") return CUDA_R_64F;
    if (dtype == "bf16") return CUDA_R_16BF;
    return CUDA_R_32F;  // Default
}

cublasComputeType_t get_compute_type(const std::string& dtype) {
    if (dtype == "f32") return CUBLAS_COMPUTE_32F;
    if (dtype == "f16") return CUBLAS_COMPUTE_16F;
    if (dtype == "f64") return CUBLAS_COMPUTE_64F;
    return CUBLAS_COMPUTE_32F;  // Default
}

size_t get_element_size(const std::string& dtype) {
    if (dtype == "f32") return sizeof(float);
    if (dtype == "f16") return sizeof(uint16_t);
    if (dtype == "f64") return sizeof(double);
    if (dtype == "bf16") return sizeof(uint16_t);
    return sizeof(float);
}

cublasOperation_t get_transpose_op(char trans) {
    return (trans == 'T' || trans == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
}

void run_gemm(const GemmParams& params) {
    std::cout << "Running GEMM: M=" << params.m << " N=" << params.n << " K=" << params.k 
              << " order_a=" << params.order_a << " order_b=" << params.order_b
              << " trans_a=" << params.trans_a << " trans_b=" << params.trans_b
              << " dtype=" << params.dtype << " alpha=" << params.alpha << " beta=" << params.beta
              << " batch=" << params.batch << std::endl;
    
    // Get CUDA types
    cudaDataType_t cuda_dtype = get_cuda_dtype(params.dtype);
    cublasComputeType_t compute_type = get_compute_type(params.dtype);
    size_t elem_size = get_element_size(params.dtype);
    
    // Matrix dimensions
    int M = params.m;
    int N = params.n;
    int K = params.k;
    int lda = params.lda;
    int ldb = params.ldb;
    int ldc = params.ldc;
    int batch = params.batch;
    
    // Allocate device memory
    // For batched operations, allocate batch * matrix_size
    // Size depends on physical layout (row-major vs col-major)
    // Row-major [M, K] with ld: size = M * ld
    // Col-major [M, K] with ld: size = K * ld (since ld is stride between columns)
    size_t size_A = batch * M * lda * elem_size;  // Assuming row-major for simplicity
    size_t size_B;
    if (params.order_b == "col") {
        size_B = batch * N * ldb * elem_size;  // Col-major: N columns, ld stride per column
    } else {
        size_B = batch * K * ldb * elem_size;  // Row-major: K rows, ld stride per row
    }
    size_t size_C = batch * M * ldc * elem_size;  // Output always row-major
    
    void *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    // Initialize with zeros (beta might be non-zero)
    CHECK_CUDA(cudaMemset(d_A, 0, size_A));
    CHECK_CUDA(cudaMemset(d_B, 0, size_B));
    CHECK_CUDA(cudaMemset(d_C, 0, size_C));
    
    // Create cuBLASLt handle
    cublasLtHandle_t handle;
    CHECK_CUBLASLT(cublasLtCreate(&handle));
    
    // Create matrix descriptors
    cublasLtMatrixLayout_t A_desc, B_desc, C_desc;
    cublasOperation_t op_A = get_transpose_op(params.trans_a);
    cublasOperation_t op_B = get_transpose_op(params.trans_b);
    
    // Matrix A: M x K (or K x M if transposed)
    // In cuBLASLt, we describe matrices in column-major by default
    // For row-major PyTorch tensors, we need to be careful about ordering
    
    // A matrix: [M, K] with leading dimension lda
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&A_desc, cuda_dtype, 
                                               (op_A == CUBLAS_OP_N) ? M : K,
                                               (op_A == CUBLAS_OP_N) ? K : M,
                                               lda));
    
    // B matrix: [K, N] with leading dimension ldb
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&B_desc, cuda_dtype,
                                               (op_B == CUBLAS_OP_N) ? K : N,
                                               (op_B == CUBLAS_OP_N) ? N : K,
                                               ldb));
    
    // C matrix: [M, N] with leading dimension ldc
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&C_desc, cuda_dtype, M, N, ldc));
    
    // Set batch count if batched
    if (batch > 1) {
        int64_t batch_count = batch;
        // Batch stride is the number of elements between consecutive matrices
        // For row-major layout: stride = rows * leading_dim
        // For matrix A [M, K] with lda: stride = M * lda
        // For matrix B [K, N] with ldb: stride = K * ldb  
        // For matrix C [M, N] with ldc: stride = M * ldc
        int64_t stride_A = static_cast<int64_t>(M) * static_cast<int64_t>(lda);
        int64_t stride_B = static_cast<int64_t>(K) * static_cast<int64_t>(ldb);
        int64_t stride_C = static_cast<int64_t>(M) * static_cast<int64_t>(ldc);
        
        std::cout << "Batched GEMM: batch=" << batch 
                  << " stride_A=" << stride_A 
                  << " stride_B=" << stride_B 
                  << " stride_C=" << stride_C << std::endl;
        
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                         &batch_count, sizeof(batch_count)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                         &stride_A, sizeof(stride_A)));
        
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                         &batch_count, sizeof(batch_count)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                         &stride_B, sizeof(stride_B)));
        
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(C_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                         &batch_count, sizeof(batch_count)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(C_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                         &stride_C, sizeof(stride_C)));
    }
    
    // Set order based on input tensor layout
    cublasLtOrder_t order_a = (params.order_a == "col") ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
    cublasLtOrder_t order_b = (params.order_b == "col") ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
    cublasLtOrder_t order_c = CUBLASLT_ORDER_ROW;  // Output is always row-major
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                     &order_a, sizeof(order_a)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                     &order_b, sizeof(order_b)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(C_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                     &order_c, sizeof(order_c)));
    
    // Create matmul descriptor
    cublasLtMatmulDesc_t matmul_desc;
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&matmul_desc, compute_type, cuda_dtype));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                   &op_A, sizeof(op_A)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                   &op_B, sizeof(op_B)));
    
    // Get heuristic for algorithm selection
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
    
    // Query workspace size (allocate reasonable amount)
    size_t workspace_size = 1024 * 1024 * 32;  // 32 MB
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(preference,
                                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &workspace_size, sizeof(workspace_size)));
    
    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    
    // Get heuristic
    cublasLtMatmulHeuristicResult_t heuristic;
    int returned_results = 0;
    CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc,
                                                   A_desc, B_desc, C_desc, C_desc,
                                                   preference, 1, &heuristic, &returned_results));
    
    if (returned_results == 0) {
        std::cerr << "Error: No cuBLASLt algorithm found" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Alpha and beta (use float for now, could be templated)
    float alpha = params.alpha;
    float beta = params.beta;
    
    // Warmup runs
    for (int i = 0; i < params.warmup; i++) {
        CHECK_CUBLASLT(cublasLtMatmul(handle, matmul_desc,
                                      &alpha, d_A, A_desc, d_B, B_desc,
                                      &beta, d_C, C_desc, d_C, C_desc,
                                      &heuristic.algo, workspace, workspace_size,
                                      0));  // Use default stream
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Measurement runs (under nsys profiling)
    // Synchronize between iterations to measure best-case launch overhead
    // without queueing delays
    for (int i = 0; i < params.runs; i++) {
        CHECK_CUBLASLT(cublasLtMatmul(handle, matmul_desc,
                                      &alpha, d_A, A_desc, d_B, B_desc,
                                      &beta, d_C, C_desc, d_C, C_desc,
                                      &heuristic.algo, workspace, workspace_size,
                                      0));
        CHECK_CUDA(cudaDeviceSynchronize());  // Clear queue for next iteration
    }
    // Final synchronization is done in the loop
    
    // Cleanup
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    CHECK_CUBLASLT(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLASLT(cublasLtMatmulDescDestroy(matmul_desc));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(C_desc));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(B_desc));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(A_desc));
    CHECK_CUBLASLT(cublasLtDestroy(handle));
    
    std::cout << "GEMM completed successfully" << std::endl;
}

int main(int argc, char** argv) {
    GemmParams params;
    parse_args(argc, argv, params);
    
    run_gemm(params);
    
    return 0;
}

