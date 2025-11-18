#!/usr/bin/env python3
"""
Generate baremetal GEMM jobs from PyTorch unique event sequences.

Parses microbench/framework/pytorch/output/unique_gemm_kernel_sequences.json and emits
microbench/baremetal/output/jobs.json with all parameters needed to reproduce
each GEMM in baremetal cuBLASLt.
"""

import json
import os
import sys

# Module-level variable for microbench directory (set in __main__)
microbench_dir = None


def parse_dtype(dtype_str):
    """Map PyTorch dtype strings to cuBLASLt dtype codes."""
    dtype_map = {
        "float": "f32",
        "float32": "f32",
        "half": "f16",
        "float16": "f16",
        "bfloat16": "bf16",
        "double": "f64",
        "float64": "f64",
    }
    return dtype_map.get(dtype_str.lower(), "f32")


def infer_layout_and_ld(dims, strides):
    """
    Infer memory layout and leading dimension from dims and strides.
    
    For 2D tensor [M, N]:
    - strides [N, 1] -> row-major, ld=N, order='row'
    - strides [1, M] -> col-major, ld=M, order='col'
    
    Returns: (order, leading_dim) where order is 'row' or 'col'
    """
    if not dims or not strides or len(dims) != len(strides):
        # Default to row-major
        return ('row', dims[-1] if dims else 1)
    
    if len(dims) == 1:
        # 1D tensor (bias for addmm) - treat as row-major
        return ('row', dims[0])
    
    # For 2D: [M, N]
    M, N = dims[-2], dims[-1]
    stride_outer, stride_inner = strides[-2], strides[-1]
    
    # Row-major: stride pattern [N, 1]
    if stride_inner == 1 and stride_outer == N:
        return ('row', N)
    # Col-major: stride pattern [1, M]
    elif stride_inner == M and stride_outer == 1:
        return ('col', M)
    # Non-standard strides - infer from stride_inner
    elif stride_inner == 1:
        # Contiguous in last dim -> row-major
        return ('row', stride_outer)
    else:
        # Assume column-major
        return ('col', stride_inner)


def extract_gemm_params(sequence):
    """
    Extract GEMM parameters from a PyTorch event sequence.
    
    Returns dict with M, N, K, trans_a, trans_b, lda, ldb, ldc, alpha, beta, dtype
    """
    cpu_op = sequence.get("cpu_op", {})
    op_name = cpu_op.get("name", "")
    input_dims = cpu_op.get("input_dims", [])
    input_strides = cpu_op.get("input_strides", [])
    input_types = cpu_op.get("input_type", [])
    concrete_inputs = cpu_op.get("concrete_inputs", [])
    
    params = {}
    
    # Extract M, N, K based on operation type
    if op_name == "aten::addmm":
        # addmm(bias, mat1, mat2, alpha=1, beta=1) -> bias + alpha * mat1 @ mat2
        # input_dims: [[bias_dim], [M, K], [K, N], [], []]
        # Result: [M, N]
        if len(input_dims) >= 3:
            bias_dims = input_dims[0]
            mat1_dims = input_dims[1]
            mat2_dims = input_dims[2]
            
            if len(mat1_dims) == 2 and len(mat2_dims) == 2:
                M, K1 = mat1_dims
                K2, N = mat2_dims
                params["m"] = M
                params["n"] = N
                params["k"] = K1  # Should equal K2
                
                # Get layouts
                mat1_strides = input_strides[1] if len(input_strides) > 1 else []
                mat2_strides = input_strides[2] if len(input_strides) > 2 else []
                bias_strides = input_strides[0] if len(input_strides) > 0 else []
                
                params["order_a"], params["lda"] = infer_layout_and_ld(mat1_dims, mat1_strides)
                params["order_b"], params["ldb"] = infer_layout_and_ld(mat2_dims, mat2_strides)
                params["trans_a"] = 'N'  # No transpose operation
                params["trans_b"] = 'N'
                
                # For output C, dimensions are [M, N], assume row-major
                params["ldc"] = N
        
        # Extract alpha, beta scalars
        if len(concrete_inputs) >= 5:
            try:
                params["alpha"] = float(concrete_inputs[3]) if concrete_inputs[3] else 1.0
            except (ValueError, TypeError):
                params["alpha"] = 1.0
            try:
                params["beta"] = float(concrete_inputs[4]) if concrete_inputs[4] else 1.0
            except (ValueError, TypeError):
                params["beta"] = 1.0
        else:
            params["alpha"] = 1.0
            params["beta"] = 1.0
    
    elif op_name == "aten::mm":
        # mm(mat1, mat2) -> mat1 @ mat2
        # input_dims: [[M, K], [K, N]]
        if len(input_dims) >= 2:
            mat1_dims = input_dims[0]
            mat2_dims = input_dims[1]
            
            if len(mat1_dims) == 2 and len(mat2_dims) == 2:
                M, K1 = mat1_dims
                K2, N = mat2_dims
                params["m"] = M
                params["n"] = N
                params["k"] = K1
                
                mat1_strides = input_strides[0] if len(input_strides) > 0 else []
                mat2_strides = input_strides[1] if len(input_strides) > 1 else []
                
                params["order_a"], params["lda"] = infer_layout_and_ld(mat1_dims, mat1_strides)
                params["order_b"], params["ldb"] = infer_layout_and_ld(mat2_dims, mat2_strides)
                params["trans_a"] = 'N'
                params["trans_b"] = 'N'
                params["ldc"] = N
        
        # mm has no bias, so beta=0
        params["alpha"] = 1.0
        params["beta"] = 0.0
    
    elif op_name == "aten::bmm":
        # bmm(batch1, batch2) -> batched matmul
        # input_dims: [[B, M, K], [B, K, N]]
        if len(input_dims) >= 2:
            batch1_dims = input_dims[0]
            batch2_dims = input_dims[1]
            
            if len(batch1_dims) == 3 and len(batch2_dims) == 3:
                B1, M, K1 = batch1_dims
                B2, K2, N = batch2_dims
                params["m"] = M
                params["n"] = N
                params["k"] = K1
                params["batch"] = B1
                
                # For batched, we still need 2D layout for each matrix
                mat1_dims_2d = [M, K1]
                mat2_dims_2d = [K2, N]
                
                # Extract 2D strides (last 2 dims)
                batch1_strides = input_strides[0] if len(input_strides) > 0 else []
                batch2_strides = input_strides[1] if len(input_strides) > 1 else []
                
                mat1_strides_2d = batch1_strides[-2:] if len(batch1_strides) >= 2 else []
                mat2_strides_2d = batch2_strides[-2:] if len(batch2_strides) >= 2 else []
                
                params["order_a"], params["lda"] = infer_layout_and_ld(mat1_dims_2d, mat1_strides_2d)
                params["order_b"], params["ldb"] = infer_layout_and_ld(mat2_dims_2d, mat2_strides_2d)
                params["trans_a"] = 'N'
                params["trans_b"] = 'N'
                params["ldc"] = N
        
        params["alpha"] = 1.0
        params["beta"] = 0.0
    
    # Extract dtype (use first non-Scalar type)
    dtype_str = "float"
    for t in input_types:
        if t and t != "Scalar":
            dtype_str = t
            break
    params["dtype"] = parse_dtype(dtype_str)
    
    return params


def generate_jobs(input_file, output_file):
    """
    Parse PyTorch unique event sequences and generate baremetal jobs.
    """
    # Read PyTorch unique event sequences
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    sequences = data.get("sequences", [])
    
    jobs = []
    for idx, sequence in enumerate(sequences):
        job_id = f"{idx+1:04d}"
        
        # Extract GEMM parameters
        params = extract_gemm_params(sequence)
        
        # Skip if critical params missing
        if "m" not in params or "n" not in params or "k" not in params:
            print(f"Warning: Skipping sequence {job_id} - missing M/N/K", file=sys.stderr)
            continue
        
        # Get target kernel name
        kernel = sequence.get("kernel", {})
        target_kernel = kernel.get("name", "unknown")
        
        # Get operation type
        cpu_op = sequence.get("cpu_op", {})
        op_name = cpu_op.get("name", "").replace("aten::", "")
        
        # Build job entry
        job = {
            "id": job_id,
            "target_kernel": target_kernel,
            "op": op_name,
            "m": params["m"],
            "n": params["n"],
            "k": params["k"],
            "order_a": params.get("order_a", "row"),
            "order_b": params.get("order_b", "row"),
            "trans_a": params.get("trans_a", "N"),
            "trans_b": params.get("trans_b", "N"),
            "lda": params.get("lda", params["k"]),
            "ldb": params.get("ldb", params["n"]),
            "ldc": params.get("ldc", params["n"]),
            "dtype": params.get("dtype", "f32"),
            "alpha": params.get("alpha", 1.0),
            "beta": params.get("beta", 0.0),
            "warmup": 200,
            "runs": 1000,
        }
        
        # Add batch count for bmm
        if "batch" in params:
            job["batch"] = params["batch"]
        
        jobs.append(job)
    
    # Write jobs to output file
    output_data = {
        "summary": {
            "total_jobs": len(jobs),
            "source": input_file,
        },
        "jobs": jobs
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    rel_path = os.path.relpath(output_file, microbench_dir) if microbench_dir else output_file
    print(f"Generated {len(jobs)} jobs -> {rel_path}")
    return len(jobs)


if __name__ == "__main__":
    # Check if env.sh has been sourced
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)
    
    # Get paths from environment
    input_file = os.environ["PYTORCH_UNIQUE_KERNELS"]
    output_file = os.environ["BAREMETAL_JOBS"]
    microbench_dir = os.environ.get("MICROBENCH_DIR")  
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    generate_jobs(input_file, output_file)
