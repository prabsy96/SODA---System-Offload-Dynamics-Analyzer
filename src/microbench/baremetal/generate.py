#!/usr/bin/env python3
"""
Generate baremetal GEMM jobs from PyTorch unique event sequences.

Parses microbench/framework/pytorch/output/unique_gemm_sequences.json and emits
microbench/baremetal/output/jobs.json with all parameters needed to reproduce
each GEMM in baremetal cuBLASLt.
"""

import sys
from pathlib import Path
from soda.common import utils




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

def extract_dtype_from_input_types(input_types):
    """Extract dtype from input_types (use first non-Scalar type)."""
    # Extract dtype (use first non-Scalar type)
    dtype_str = "float"
    for t in input_types:
        if t and t != "Scalar":
            dtype_str = t
            break
    return utils.parse_dtype_to_cublaslt(dtype_str)


def extract_addmm_params(input_dims, input_strides, concrete_inputs):
    """Extract GEMM parameters for aten::addmm operation.
    
    addmm(bias, mat1, mat2, alpha=1, beta=1) -> bias + alpha * mat1 @ mat2
    input_dims: [[bias_dim], [M, K], [K, N], [], []]
    Result: [M, N]
    """
    params = {}
    
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
    
    # Extract alpha, beta scalars (defaults: alpha=1.0, beta=1.0 for addmm)
    params["alpha"], params["beta"] = utils.extract_alpha_beta(concrete_inputs)
    
    return params


def extract_mm_params(input_dims, input_strides):
    """Extract GEMM parameters for aten::mm operation.
    
    mm(mat1, mat2) -> mat1 @ mat2
    input_dims: [[M, K], [K, N]]
    """
    params = {}
    
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
    
    return params


def extract_bmm_params(input_dims, input_strides):
    """Extract GEMM parameters for aten::bmm operation.
    
    bmm(batch1, batch2) -> batched matmul
    input_dims: [[B, M, K], [B, K, N]]
    """
    params = {}
    
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
    
    return params


def extract_gemm_params(sequence):
    """
    Extract GEMM parameters from a PyTorch event sequence.
    
    Returns dict with M, N, K, trans_a, trans_b, lda, ldb, ldc, alpha, beta, dtype
    """
    aten_op = sequence["aten_op"]
    op_name = aten_op["name"]
    input_dims = aten_op["input_dims"]
    input_strides = aten_op["input_strides"]
    input_types = aten_op["input_type"]
    concrete_inputs = aten_op["concrete_inputs"]
    
    # Dispatch to operation-specific extractors
    if op_name == "aten::addmm":
        params = extract_addmm_params(input_dims, input_strides, concrete_inputs)
    elif op_name == "aten::mm":
        params = extract_mm_params(input_dims, input_strides)
    elif op_name == "aten::bmm":
        params = extract_bmm_params(input_dims, input_strides)
    else:
        params = {}
    
    # Extract dtype (common for all operations)
    params["dtype"] = extract_dtype_from_input_types(input_types)
    
    return params


def generate_jobs(target_sequences: dict, warmup: int, runs: int):
    """
    Generate baremetal jobs from PyTorch unique event sequences data.
    
    Args:
        target_sequences: Dictionary with 'sequences' key containing event sequences.
        warmup: Number of warmup runs.
        runs: Number of measurement runs.
    """
    output_file = utils.get_path("BAREMETAL_JOBS")
    sequences = target_sequences["sequences"]
    
    jobs = []
    
    # Add null kernel job (job 0000) for baseline launch tax measurement
    job_id = "0000"
    print(f"Generating job {job_id} (__null__)")
    jobs.append({
        "id": job_id,
        "name": "__null__",
        "grid": [1, 1, 1],  # Explicit value for null kernel, not a default
        "block": [1, 1, 1],  # Explicit value for null kernel, not a default
        "shared_memory": 0,  # Explicit value for null kernel, not a default
        "aten_op": None,
        "warmup": warmup,
        "runs": runs,
    })
    
    for idx, sequence in enumerate(sequences):
        job_id = f"{idx+1:04d}"
        print(f"Generating job {job_id}")
        
        # Extract GEMM parameters
        params = extract_gemm_params(sequence)
        
        # Skip if critical params missing
        if "m" not in params or "n" not in params or "k" not in params:
            print(f"Warning: Skipping sequence {job_id} - missing M/N/K", file=sys.stderr)
            continue
        
        # Build job entry
        kernel = sequence["kernel"]
        aten_op = sequence["aten_op"]
        job = {
            "id": job_id,
            "name": kernel["name"],
            "grid": kernel["grid"],
            "block": kernel["block"],
            "shared_memory": kernel["shared_memory"],
            "registers_per_thread": kernel["registers_per_thread"],
            "aten_op": aten_op,
            "m": params["m"],
            "n": params["n"],
            "k": params["k"],
            "order_a": params["order_a"],
            "order_b": params["order_b"],
            "trans_a": params["trans_a"],
            "trans_b": params["trans_b"],
            "lda": params["lda"],
            "ldb": params["ldb"],
            "ldc": params["ldc"],
            "dtype": params["dtype"],
            "alpha": params["alpha"],
            "beta": params["beta"],
            "warmup": warmup,
            "runs": runs,
        }
        
        # Add batch count for bmm
        if "batch" in params:
            job["batch"] = params["batch"]
        
        jobs.append(job)
    
    # Write jobs to output file
    jobs_data = {
        "summary": {
            "total_jobs": len(jobs),
        },
        "jobs": jobs
    }
    
    utils.save_json(output_file, jobs_data)
    
    print(f"Generated {len(jobs)} jobs -> {output_file}")
