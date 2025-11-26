#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>
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
    int algo_index;  // Algorithm index to use (for matching, from heuristic results)
    bool has_algo_index;
    int algo_id;  // Algorithm ID to use directly (from cublasLtMatmulAlgoGetIds)
    bool has_algo_id;
    bool list_algo_ids;  // If true, just list all algorithm IDs and exit
    bool null_kernel;  // If true, launch empty kernel to measure baseline launch tax
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
    params.has_algo_index = false;
    params.algo_index = 0;
    params.has_algo_id = false;
    params.algo_id = 0;
    params.list_algo_ids = false;
    params.null_kernel = false;
    
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
        } else if (arg == "--algo_index" && i+1 < argc) {
            // Algorithm index to use (for matching, from heuristic results)
            params.algo_index = std::atoi(argv[++i]);
            params.has_algo_index = true;
        } else if (arg == "--algo_id" && i+1 < argc) {
            // Algorithm ID to use directly (from cublasLtMatmulAlgoGetIds)
            params.algo_id = std::atoi(argv[++i]);
            params.has_algo_id = true;
        } else if (arg == "--list_algo_ids") {
            // List all available algorithm IDs and exit
            params.list_algo_ids = true;
        } else if (arg == "--null_kernel") {
            // Launch empty kernel to measure baseline launch tax
            params.null_kernel = true;
        }
    }
    
    // Validation
    if (!params.null_kernel && (params.m <= 0 || params.n <= 0 || params.k <= 0)) {
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

void list_all_algorithm_ids(const GemmParams& params) {
    // Get CUDA types
    cudaDataType_t cuda_dtype = get_cuda_dtype(params.dtype);
    cublasComputeType_t compute_type = get_compute_type(params.dtype);
    
    // Create cuBLASLt handle
    cublasLtHandle_t handle;
    CHECK_CUBLASLT(cublasLtCreate(&handle));
    
    // Query all algorithm IDs
    const int max_algo_ids = 10000;  // Large number to get all IDs
    int algo_ids[max_algo_ids];
    int returned_count = 0;
    
    cublasStatus_t status = cublasLtMatmulAlgoGetIds(
        handle,
        compute_type,
        cuda_dtype,  // scaleType
        cuda_dtype,  // Atype
        cuda_dtype,  // Btype
        cuda_dtype,  // Ctype
        cuda_dtype,  // Dtype
        max_algo_ids,
        algo_ids,
        &returned_count
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to get algorithm IDs: " << status << std::endl;
        CHECK_CUBLASLT(cublasLtDestroy(handle));
        exit(EXIT_FAILURE);
    }
    
    std::cout << "Found " << returned_count << " algorithm IDs:" << std::endl;
    for (int i = 0; i < returned_count; i++) {
        std::cout << algo_ids[i];
        if (i < returned_count - 1) std::cout << " ";
    }
    std::cout << std::endl;
    
    CHECK_CUBLASLT(cublasLtDestroy(handle));
}

__global__ void null_kernel() {
    // Empty kernel to measure baseline launch tax
}

void run_null_kernel(const GemmParams& params) {
    CHECK_CUDA(cudaDeviceSynchronize());
    
    for (int i = 0; i < params.warmup; i++) {
        null_kernel<<<1, 1>>>();
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    for (int i = 0; i < params.runs; i++) {
        null_kernel<<<1, 1>>>();
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

void run_gemm(const GemmParams& params) {
    if (params.null_kernel) {
        run_null_kernel(params);
        return;
    }
    
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
    
    // Create matrix descriptors - EXACTLY matching PyTorch's approach
    //
    // PyTorch Reference: aten/src/ATen/cuda/CUDABlas.cpp
    //   * CuBlasLtMatrixLayout constructor (lines 288-296):
    //     CuBlasLtMatrixLayout(type, rows, cols, ld, transpose)
    //     calls: cublasLtMatrixLayoutCreate(..., t ? cols : rows, t ? rows : cols, ld)
    //   * Usage in gemm_and_bias (line 1239-1241):
    //     CuBlasLtMatrixLayout Adesc(abcType, m, k, mat1_ld, transpose_mat1);
    //     CuBlasLtMatrixLayout Bdesc(abcType, k, n, mat2_ld, transpose_mat2);
    //     CuBlasLtMatrixLayout Cdesc(abcType, m, n, result_ld);
    //   * Usage in bgemm_internal_cublaslt (lines 361-363):
    //     CuBlasLtMatrixLayout Adesc(abcType, m, k, lda, opa == CUBLAS_OP_T);
    //     CuBlasLtMatrixLayout Bdesc(abcType, k, n, ldb, opb == CUBLAS_OP_T);
    //     CuBlasLtMatrixLayout Cdesc(abcType, m, n, ldc);
    //
    // Key insight: PyTorch swaps rows/cols when transpose=true:
    //   * If transpose: pass (cols, rows) to cublasLtMatrixLayoutCreate
    //   * If not transpose: pass (rows, cols) to cublasLtMatrixLayoutCreate
    // This matches cuBLASLt's column-major interpretation.
    //
    // PyTorch does NOT set ORDER attributes - it uses default column-major
    // and handles row-major tensors via transpose flags, not ORDER attribute.
    cublasLtMatrixLayout_t A_desc, B_desc, C_desc;
    cublasOperation_t op_A = get_transpose_op(params.trans_a);
    cublasOperation_t op_B = get_transpose_op(params.trans_b);
    
    // Matrix A: [M, K] with leading dimension lda
    // If transpose: cuBLASLt sees it as [K, M] (swapped dimensions)
    bool transpose_A = (op_A == CUBLAS_OP_T);
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&A_desc, cuda_dtype, 
                                               transpose_A ? K : M,  // rows: K if transposed, M otherwise
                                               transpose_A ? M : K,  // cols: M if transposed, K otherwise
                                               lda));
    
    // Matrix B: [K, N] with leading dimension ldb
    // If transpose: cuBLASLt sees it as [N, K] (swapped dimensions)
    bool transpose_B = (op_B == CUBLAS_OP_T);
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&B_desc, cuda_dtype,
                                               transpose_B ? N : K,  // rows: N if transposed, K otherwise
                                               transpose_B ? K : N,  // cols: K if transposed, N otherwise
                                               ldb));
    
    // Matrix C: [M, N] with leading dimension ldc (never transposed)
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
    
    // ORDER attributes: PyTorch does NOT set ORDER - it uses default column-major
    // 
    // PyTorch Reference: aten/src/ATen/cuda/CUDABlas.cpp
    //   * gemm_and_bias (lines 1239-1241): Creates layouts without setting ORDER
    //   * bgemm_internal_cublaslt (lines 361-363): Creates layouts without setting ORDER
    // 
    // PyTorch handles row-major tensors by:
    //   * Computing leading dimensions from tensor strides (see cublasCommonArgs, lines 109-111)
    //   * Using transpose flags to swap dimensions (see matrix layout creation above)
    //   * Relying on cuBLASLt's default column-major interpretation
    //
    // Why we differ: Our baremetal code receives explicit "row" vs "col" order parameters.
    // Without ORDER attributes, row-major data causes illegal memory access. We set ORDER
    // attributes when order_a or order_b is "row" to ensure correct memory interpretation.
    // This is a necessary deviation from PyTorch's approach due to our different input format.
    if (params.order_a == "row" || params.order_b == "row") {
    cublasLtOrder_t order_a = (params.order_a == "col") ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
    cublasLtOrder_t order_b = (params.order_b == "col") ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
    cublasLtOrder_t order_c = CUBLASLT_ORDER_ROW;  // Output is always row-major
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                     &order_a, sizeof(order_a)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                     &order_b, sizeof(order_b)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(C_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                     &order_c, sizeof(order_c)));
    }
    
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
    
    // Workspace size - match PyTorch's default and environment variable handling
    //
    // PyTorch Reference: aten/src/ATen/cuda/CUDABlas.cpp
    //   * _parseChosenWorkspaceSize() (lines 182-203):
    //     * Default: 1024 KiB (1 MB) - see comment "default size in KiB according to #73328"
    //     * Reads CUBLASLT_WORKSPACE_SIZE env var (in KiB units)
    //     * Returns size in bytes: workspace_size * 1024
    //   * _getWorkspaceSize() (lines 205-208): Returns cached workspace size
    //   * Usage in gemm_and_bias (line 1246):
    //     size_t workspaceSize = _getWorkspaceSize();
    //     preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);
    //   * See also: https://github.com/pytorch/pytorch/issues/73328
    //
    // Rationale: 1MB workspace allows cuBLASLt to select algorithms that require
    // temporary memory. Larger workspace (e.g., 128MB) can cause different algorithm
    // selection, leading to kernel mismatches.
    const char* workspace_env = std::getenv("CUBLASLT_WORKSPACE_SIZE");
    const size_t DEFAULT_WORKSPACE_KIB = 1024;  // PyTorch default: 1024 KiB = 1 MB
    size_t workspace_size = DEFAULT_WORKSPACE_KIB * 1024;  // Convert KiB to bytes
    if (workspace_env) {
        // Environment variable is in KiB units (matching PyTorch's convention)
        workspace_size = std::stoull(workspace_env) * 1024;  // Convert KiB to bytes
    }
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(preference,
                                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &workspace_size, sizeof(workspace_size)));
    
    // Search mode: PyTorch uses default heuristic mode (does NOT set SEARCH_MODE_ALL)
    //
    // PyTorch Reference: aten/src/ATen/cuda/CUDABlas.cpp
    //   * gemm_and_bias (lines 1267-1277): Calls cublasLtMatmulAlgoGetHeuristic with
    //     requestedResultCount=1, using default heuristic mode
    //   * bgemm_internal_cublaslt: No SEARCH_MODE_ALL setting found in preference setup
    //
    // Rationale: SEARCH_MODE_ALL forces exhaustive search, which can select different
    // algorithms than PyTorch's heuristic mode. We use default heuristic to match PyTorch.
    // (Commented out to match PyTorch's behavior - only enable for diagnostic algorithm sweeps)
    // #ifdef CUBLASLT_SEARCH_MODE_ALL
    // cublasLtMatmulPreference_t search_mode = CUBLASLT_SEARCH_MODE_ALL;
    // CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(preference,
    //                                                      CUBLASLT_MATMUL_PREF_SEARCH_MODE,
    //                                                      &search_mode, sizeof(search_mode)));
    // #endif
    
    // Pointer alignment - EXACTLY matching PyTorch's _getAlignment implementation
    //
    // PyTorch Reference: aten/src/ATen/cuda/CUDABlas.cpp
    //   * _getAlignment() (lines 171-179):
    //     uint32_t _getAlignment(uintptr_t address) {
    //       uint32_t alignment = 256;
    //       for (; ; alignment /= 2) {
    //         if (!(address % alignment)) {
    //           return alignment;
    //         }
    //       }
    //     }
    //   * Usage in gemm_and_bias (lines 1250-1257):
    //     uint32_t a_alignment = _getAlignment(reinterpret_cast<uintptr_t>(mat1_ptr));
    //     uint32_t b_alignment = _getAlignment(reinterpret_cast<uintptr_t>(mat2_ptr));
    //     uint32_t c_alignment = _getAlignment(reinterpret_cast<uintptr_t>(result_ptr));
    //     preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
    //     preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
    //     preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);
    //
    // Rationale: Alignment affects algorithm selection. Using a fixed value (e.g., 16 bytes)
    // can cause different algorithms than PyTorch, which computes actual pointer alignment.
    // Starting from 256 bytes and halving until alignment matches ensures we get the
    // maximum alignment that the pointer actually satisfies.
    auto getAlignment = [](uintptr_t address) -> uint32_t {
        const uint32_t MAX_ALIGNMENT = 256;  // Start from 256 bytes (PyTorch's max)
        uint32_t alignment = MAX_ALIGNMENT;
        for (; ; alignment /= 2) {
            if (!(address % alignment)) {
                return alignment;
            }
        }
    };
    
    uint32_t a_alignment = getAlignment(reinterpret_cast<uintptr_t>(d_A));
    uint32_t b_alignment = getAlignment(reinterpret_cast<uintptr_t>(d_B));
    uint32_t c_alignment = getAlignment(reinterpret_cast<uintptr_t>(d_C));
    
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(preference,
                                                         CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                                                         &a_alignment, sizeof(a_alignment)));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(preference,
                                                         CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                                                         &b_alignment, sizeof(b_alignment)));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(preference,
                                                         CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                                                         &c_alignment, sizeof(c_alignment)));
    
    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    
    // Select algorithm: either use direct ID or from heuristic results
    cublasLtMatmulAlgo_t selected_algo;
    
    if (params.has_algo_id) {
        // Use algorithm ID directly (from cublasLtMatmulAlgoGetIds)
        cublasStatus_t status = cublasLtMatmulAlgoInit(
            handle,
            compute_type,
            cuda_dtype,  // scaleType
            cuda_dtype,  // Atype
            cuda_dtype,  // Btype
            cuda_dtype,  // Ctype
            cuda_dtype,  // Dtype
            params.algo_id,
            &selected_algo
        );
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error: Failed to initialize algorithm ID " << params.algo_id 
                      << " (status: " << status << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
        
        // Test if algorithm is supported for this problem by trying a test run
        // (some algorithms can be initialized but not supported for specific problem sizes)
        float test_alpha = 1.0f;
        float test_beta = 0.0f;
        status = cublasLtMatmul(handle, matmul_desc,
                               &test_alpha, d_A, A_desc, d_B, B_desc,
                               &test_beta, d_C, C_desc, d_C, C_desc,
                               &selected_algo, workspace, workspace_size,
                               0);
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error: Algorithm ID " << params.algo_id 
                      << " not supported for this problem (status: " << status << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
        
        // Reset C matrix (test run modified it)
        CHECK_CUDA(cudaMemset(d_C, 0, size_C));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        std::cout << "Using algorithm ID " << params.algo_id << " (requested)" << std::endl;
    } else {
        // Algorithm selection via heuristic - match PyTorch's behavior
        //
        // PyTorch Reference: aten/src/ATen/cuda/CUDABlas.cpp
        //   * gemm_and_bias (lines 1267-1277):
        //     cublasLtMatmulAlgoGetHeuristic(
        //         ltHandle,
        //         computeDesc.descriptor(),
        //         Adesc.descriptor(),
        //         Bdesc.descriptor(),
        //         Cdesc.descriptor(),
        //         Cdesc.descriptor(),
        //         preference.descriptor(),
        //         1,  // requestedResultCount = 1 (only best algorithm)
        //         &heuristicResult,
        //         &returnedResult);
        //   * Uses heuristicResult.algo directly (line 1295)
        //
        // Rationale: PyTorch requests only 1 algorithm (the best heuristic result) and uses it.
        // Requesting more algorithms (e.g., 200) can return different ordering or different
        // algorithms, causing kernel mismatches. We default to 1 like PyTorch, but allow
        // requesting 200 when has_algo_index=true for diagnostic algorithm sweeps.
        const int PYTORCH_DEFAULT_ALGO_COUNT = 1;  // PyTorch requests only 1 algorithm
        const int DIAGNOSTIC_ALGO_COUNT = 200;     // For algorithm matching sweeps
        int requested_algo_count = params.has_algo_index ? DIAGNOSTIC_ALGO_COUNT : PYTORCH_DEFAULT_ALGO_COUNT;
        
        std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(requested_algo_count);
    int returned_results = 0;
    CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc,
                                                   A_desc, B_desc, C_desc, C_desc,
                                                       preference, requested_algo_count,
                                                       heuristic_results.data(), &returned_results));
    
    if (returned_results == 0) {
        std::cerr << "Error: No cuBLASLt algorithm found" << std::endl;
        exit(EXIT_FAILURE);
        }
        
        // Print available algorithm count (for debugging when doing algorithm sweeps)
        if (params.has_algo_index) {
            std::cout << "Available algorithms: 0-" << (returned_results - 1) << " (total: " << returned_results << ")" << std::endl;
        }
        
        // Select algorithm: use specified index if provided, otherwise use first (best) - matches PyTorch
        // PyTorch always uses index 0 (the best heuristic result)
        const int PYTORCH_ALGO_INDEX = 0;  // PyTorch uses first (best) algorithm from heuristic
        int selected_algo_idx = params.has_algo_index ? params.algo_index : PYTORCH_ALGO_INDEX;
        if (selected_algo_idx < 0 || selected_algo_idx >= returned_results) {
            std::cerr << "Error: Invalid algorithm index " << selected_algo_idx 
                      << " (available: 0-" << (returned_results - 1) << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
        
        selected_algo = heuristic_results[selected_algo_idx].algo;
        
        if (params.has_algo_index) {
            std::cout << "Using algorithm index " << selected_algo_idx << " (requested)" << std::endl;
        } else {
            std::cout << "Using algorithm index 0 (best heuristic result, matches PyTorch)" << std::endl;
        }
    }
    
    // Alpha and beta (use float for now, could be templated)
    float alpha = params.alpha;
    float beta = params.beta;
    
    // Warmup runs
    for (int i = 0; i < params.warmup; i++) {
        CHECK_CUBLASLT(cublasLtMatmul(handle, matmul_desc,
                                      &alpha, d_A, A_desc, d_B, B_desc,
                                      &beta, d_C, C_desc, d_C, C_desc,
                                      &selected_algo, workspace, workspace_size,
                                      0));  // Use default stream
        CHECK_CUDA(cudaDeviceSynchronize());  // Clear queue for next iteration
    }
    CHECK_CUDA(cudaDeviceSynchronize()); // Final synchronization because why not 
    
    // Measurement runs (under nsys profiling)
    // Synchronize between iterations to measure best-case launch overhead
    // without queueing delays
    for (int i = 0; i < params.runs; i++) {
        CHECK_CUBLASLT(cublasLtMatmul(handle, matmul_desc,
                                      &alpha, d_A, A_desc, d_B, B_desc,
                                      &beta, d_C, C_desc, d_C, C_desc,
                                      &selected_algo, workspace, workspace_size,
                                      0));
        CHECK_CUDA(cudaDeviceSynchronize());  // Clear queue for next iteration
    }
    CHECK_CUDA(cudaDeviceSynchronize()); // Final synchronization because why not
    
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
    
    if (params.list_algo_ids) {
        list_all_algorithm_ids(params);
        return 0;
    }
    
    run_gemm(params);
    return 0;
}
