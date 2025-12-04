#!/usr/bin/env python3
"""
Utilities for baremetal microbenchmarking: nsys profiling, trace extraction, and build helpers.
"""

import subprocess
import sqlite3
from typing import Optional, Tuple, List, Dict, Any

from soda.common import utils
from soda.common.data import Kernel

def nsys_profile_to_sql(
    trace_file_name: str,
    args: List[str],
    timeout: Optional[int] = None,
    cleanup: bool = False,
) -> Tuple[bool, Optional[str], str]:
    """
    Run `nsys profile` followed by `nsys export` to sqlite.

    Returns: (success, trace_file_sql|None, message)
    """
    traces_dir = utils.get_path("BAREMETAL_TRACES")
    utils.ensure_dir(traces_dir)
    trace_file = traces_dir / trace_file_name
    
    trace_file_rep = trace_file.with_suffix(".nsys-rep")
    trace_file_sql = trace_file_rep.with_suffix(".sqlite")

    # Build args for nsys profile and run 
    args = [
        "nsys",
        "profile",
        "--trace=cuda,osrt",
        "--output",
        str(trace_file),
        "--force-overwrite=true",
    ] + list(args)

    # Run nsys profile
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    # Check if nsys profile was successful
    success = result.returncode == 0
    if success:
        utils.ensure_file(trace_file_rep)
    else: 
        return False, None, result.stderr or result.stdout

    # Build args for nsys export and run
    args = [
        "nsys",
        "export",
        "--type=sqlite",
        "--output",
        str(trace_file_sql),
        "--force-overwrite=true",
        str(trace_file_rep),
    ]

    result = subprocess.run(
        args, 
        capture_output=True, 
        text=True
    )

    # Clean up trace file if requested
    if cleanup:
        utils.remove_file(trace_file_rep)

    # Check if nsys export was successful
    success = result.returncode == 0
    if success:
        utils.ensure_file(trace_file_sql)
        return True, str(trace_file_sql), ""
    else: 
        return False, None, result.stderr or result.stdout


def extract_kernels_from_trace(trace_file_sql, cleanup=True):
    """Extract all kernels from nsys sqlite trace.
    
    Args:
        trace_file_sql: Path to SQLite trace file
        cleanup: If True, delete the trace file after extraction.
    
    Returns: 
        List of Kernel objects.
    """
    kernels = []
    try:
        conn = sqlite3.connect(trace_file_sql)
        cursor = conn.cursor()
        
        # Query for all kernel events with all available fields
        cursor.execute("""
            SELECT k.start, k.end, k.correlationId, s.value, 
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ,
                   k.staticSharedMemory, k.dynamicSharedMemory,
                   k.deviceId, k.contextId, k.streamId,
                   k.registersPerThread
            FROM CUPTI_ACTIVITY_KIND_KERNEL as k
            JOIN StringIds as s ON k.demangledName = s.id
        """)
        
        # Fetch all rows and filter in Python
        for row in cursor.fetchall():
            start_ns, end_ns, corr_id, name, gx, gy, gz, bx, by, bz, static_smem, dyn_smem, device_id, context_id, stream_id, regs = row

            # Filter: only keep GEMM or null kernels
            if "gemm" not in name.lower() and "null" not in name.lower():
                continue
            
            kernels.append(Kernel(
                name=name,
                grid=[gx, gy, gz],
                block=[bx, by, bz],
                shared_memory=static_smem + dyn_smem,
                correlation=corr_id,
                ts=start_ns / 1000.0,  # Convert nanoseconds to microseconds
                dur=(end_ns - start_ns) / 1000.0,  # Convert nanoseconds to microseconds
                device=device_id,
                context=context_id,
                stream=stream_id,
                registers_per_thread=regs
            ))
            
        conn.close()
    except Exception as e:
        print(f"Error extracting kernel from trace: {e}")
        return []

    if cleanup:
        utils.remove_file(trace_file_sql)
    
    return kernels

def extract_launches_from_trace(trace_file_sql):
    """Extract CUDA launch events from nsys sqlite trace.
    
    Args:
        trace_file_sql: Path to SQLite trace file
    
    Returns:
        Dictionary mapping correlationId to launch info dict:
        {
            correlationId: {
                "ts": start_us,
                "dur": dur_us,
                "correlation": correlationId
            }
        }
    """
    cuda_launches = {}
    try:
        conn = sqlite3.connect(trace_file_sql)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT start, end, correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE nameId IN (SELECT id FROM StringIds WHERE value LIKE 'cudaLaunchKernel%')
        """)
        
        for row in cursor.fetchall():
            start_ns, end_ns, corr_id = row
            cuda_launches[corr_id] = {
                "ts": start_ns / 1000.0,  # Convert ns to us
                "dur": (end_ns - start_ns) / 1000.0,
                "correlation": corr_id,
            }
            
        conn.close()
    except Exception as e:
        print(f"Warning: Could not query CUDA launch events: {e}", file=sys.stderr)
        return {}
        
    return cuda_launches


def build_base_args(job):
    """Build base command line arguments for the C++ binary.
    
    Args:
        job: Job dictionary with GEMM parameters
    
    Returns:
        List of command line arguments 
    """
    binary_path = utils.get_path("BAREMETAL_BINARY")
    
    return [
        str(binary_path),
        "--m", str(job["m"]),
        "--n", str(job["n"]),
        "--k", str(job["k"]),
        "--lda", str(job["lda"]),
        "--ldb", str(job["ldb"]),
        "--ldc", str(job["ldc"]),

        "--order_a", job["order_a"],
        "--order_b", job["order_b"],
        "--trans_a", job["trans_a"],
        "--trans_b", job["trans_b"],
        
        "--dtype", job["dtype"],
        "--alpha", str(job["alpha"]),
        "--beta", str(job["beta"]),
    ]

def build_binary():
    """Build the C++ binary using cmake."""
    baremetal_dir = utils.get_path("BAREMETAL_MICROBENCH_DIR")
    print("Building C++ binary")
    build_dir = baremetal_dir / "build"
    
    # Configure
    result = subprocess.run(
        ["cmake", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=str(baremetal_dir),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"CMake configure failed:\n{result.stderr}")
    
    # Build
    result = subprocess.run(
        ["cmake", "--build", str(build_dir)],
        cwd=str(baremetal_dir),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Build failed:\n{result.stderr}")
    
    print("Build successful")
