#!/usr/bin/env python3
"""
Utilities for baremetal microbenchmarking: nsys profiling, trace extraction, and build helpers.
"""

import shutil
import subprocess
import sqlite3
import sys
from typing import Optional, Tuple, List, Dict, Any

from soda.common import utils
from soda.common.data import Kernel


def nsys_check_available() -> bool:
    """Verify that ``nsys`` is in PATH and can run."""
    return shutil.which("nsys") is not None


def nsys_profile(
    trace_file_name: str,
    args: List[str],
    timeout: Optional[int] = None,
    cleanup: bool = False,
    extra_env: Optional[dict] = None,
    trace_apis: str = "cuda,osrt,nvtx",
) -> Tuple[bool, Optional[str], str]:
    """
    Run `nsys profile` followed by `nsys export` to sqlite.

    Args:
        extra_env: Optional environment dict to pass to the subprocess.
                   If provided, it replaces the full process environment for
                   the nsys profile call (use ``dict(os.environ)`` + overrides).

    Returns: (success, trace_file_sql|None, message)
    """
    if not nsys_check_available():
        return False, None, (
            "nsys not found in PATH. "
            "Load the Nsight module (e.g. 'module load cuda12.8/nsight/12.8.1') "
            "or install NVIDIA Nsight Systems."
        )

    traces_dir = utils.get_path("BAREMETAL_TRACES")
    utils.ensure_dir(traces_dir)
    trace_file = traces_dir / trace_file_name

    trace_file_rep = trace_file.with_suffix(".nsys-rep")
    trace_file_sql = trace_file_rep.with_suffix(".sqlite")

    # Build args for nsys profile and run
    args = [
        "nsys",
        "profile",
        f"--trace={trace_apis}",
        "--output",
        str(trace_file),
        "--force-overwrite=true",
    ] + list(args)

    # Run nsys profile
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=extra_env,
        )
    except subprocess.TimeoutExpired:
        return False, None, f"TIMEOUT: nsys profile timed out after {timeout}s"

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

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, None, f"TIMEOUT: nsys export timed out after {timeout}s"

    # Clean up trace file if requested
    ALWAYS = True
    if cleanup or ALWAYS:
        utils.remove_file(trace_file_rep)

    # Check if nsys export was successful
    success = result.returncode == 0
    if success:
        utils.ensure_file(trace_file_sql)
        return True, str(trace_file_sql), ""
    else: 
        return False, None, result.stderr or result.stdout


def to_hashable(obj: Any) -> Any:
    """
    Recursively convert an object to a hashable type.
    
    - Lists become tuples
    - Dicts become tuples of (key, value) pairs
    - Other types are returned as-is
    
    Args:
        obj: Any Python object
    
    Returns:
        Hashable version of the object
    """
    if isinstance(obj, list):
        return tuple(to_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple((k, to_hashable(v)) for k, v in sorted(obj.items()))
    elif isinstance(obj, set):
        return frozenset(to_hashable(item) for item in obj)
    else:
        return obj

def extract_kernels_sql(trace_file_sql, filter_gemm_only: bool = True):
    """Extract kernels from nsys sqlite trace.

    Args:
        trace_file_sql: Path to SQLite trace file
        filter_gemm_only: If True (default), only keep GEMM and null kernels.
            Set to False to extract all kernel types (needed by the enhanced
            TaxBreak pipeline).

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

            # Filter: only keep GEMM or null kernels (legacy baremetal behavior)
            if filter_gemm_only:
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
        err = str(e)
        if "no such table" in err and "CUPTI_ACTIVITY_KIND_KERNEL" in err:
            # Expected for CPU-only ops (e.g., aten::arange) that dispatch no GPU kernels.
            # The nsys trace contains no kernel activity table for this replay.
            pass
        else:
            print(f"Error extracting kernel from trace: {e}")
        return []

    return kernels

def extract_launches_sql(trace_file_sql):
    """Extract CUDA launch events from nsys sqlite trace.
    
    Args:
        trace_file_sql: Path to SQLite trace file
    
    Returns:
        Dictionary mapping correlationId to launch info dict:
        {
            correlationId: {
                "ts": start_us,
                "dur": dur,
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
            WHERE nameId IN (
                SELECT id FROM StringIds
                WHERE value LIKE 'cudaLaunchKernel%'
                   OR value LIKE 'cuLaunchKernel%'
                   OR value LIKE 'cudaLaunchCooperativeKernel%'
            )
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

def extract_culib_markers_sql(trace_file_sql):
    """
    Extract NVTX range stamps from an nsys sqlite trace.

    Returns:
        List of dicts with name, ts, dur (microseconds).
    """
    ranges = []
    try:
        conn = sqlite3.connect(trace_file_sql)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT start, end, text
            FROM NVTX_EVENTS
            WHERE text IS NOT NULL
              AND start IS NOT NULL
              AND end IS NOT NULL
            """
        )
        for start_ns, end_ns, name in cursor.fetchall():
            ranges.append(
                {
                    "name": name,
                    "ts": start_ns / 1000.0,
                    "dur": (end_ns - start_ns) / 1000.0,
                }
            )
        conn.close()
    except Exception as e:
        print(f"Warning: Could not extract NVTX ranges: {e}")
        return []

    return ranges


def detect_vendor_library_events(trace_file_sql: str) -> Optional[bool]:
    """Return True/False/None based on cuBLAS or cuDNN API call presence.

    Supports two nsys schema generations:

    **Old nsys (< 2024.x):** creates ``CUBLAS_EVENTS`` / ``CUDNN_EVENTS`` tables
    when ``--trace=cublas,cudnn`` is active.  A row in either table → ``True``.

    **New nsys (≥ 2024.x):** no longer creates dedicated vendor tables.  Instead,
    cuBLAS/cuDNN API calls appear in ``CUPTI_ACTIVITY_KIND_RUNTIME`` alongside the
    regular CUDA runtime API calls (function names resolved via ``StringIds``).
    Additionally, ``META_DATA_CAPTURE`` records a
    ``('CAPTURE_EVENT_TYPE', 'CuBLAS')`` row for every trace that was collected
    with ``--trace=cublas``.

    Detection algorithm (short-circuit on first positive):
      1. ``CUBLAS_EVENTS`` / ``CUDNN_EVENTS`` table has rows → ``True``  (old nsys)
         Tables exist but empty → vendor tracing confirmed active (old-nsys path),
         skip META_DATA_CAPTURE check.
      2. ``META_DATA_CAPTURE`` has no ``CuBLAS``/``CuDNN`` capture-event row AND no
         legacy vendor table was found → vendor tracing was not active → ``None``
         (inconclusive; caller falls back to Phase-1 DB heuristic).
      3. ``CUPTI_ACTIVITY_KIND_RUNTIME`` JOIN ``StringIds`` finds a ``cublas%``,
         ``cublaslt%``, or ``cudnn%`` function name → ``True``  (new nsys,
         cuBLAS/cuBLASLt/cuDNN API called)
      4. All checks passed but no vendor API found → ``False``  (confirmed absent;
         tracing was active and produced GPU activity but no cuBLAS/cuDNN calls)

    Return values:
      - ``True``  — vendor API calls confirmed in trace; kernel is library-mediated.
      - ``False`` — vendor tracing was active but no API calls found; confirmed NOT
                    library-mediated (overrides Phase-1 DB heuristic).
      - ``None``  — cannot determine (file unreadable, or vendor tracing was not
                    active); caller should fall back to the Phase-1 DB heuristic.
    """
    try:
        conn = sqlite3.connect(trace_file_sql)
        cursor = conn.cursor()

        # --- Step 1: legacy CUBLAS_EVENTS / CUDNN_EVENTS (old nsys) ---
        # Track if the tables existed at all: even empty tables in old nsys proves
        # vendor tracing was active (equivalent to META_DATA_CAPTURE for new nsys).
        legacy_table_found = False
        for table in ("CUBLAS_EVENTS", "CUDNN_EVENTS"):
            try:
                cursor.execute(f"SELECT 1 FROM {table} LIMIT 1")
                legacy_table_found = True  # table exists (old nsys vendor tracing)
                if cursor.fetchone() is not None:
                    conn.close()
                    return True  # old nsys with active cuBLAS/cuDNN events
            except sqlite3.OperationalError:
                continue  # table absent — expected for nsys ≥ 2024.x

        # --- Step 2: confirm vendor tracing was active ---
        # Old nsys: legacy_table_found is sufficient evidence.
        # New nsys: check META_DATA_CAPTURE for CuBLAS/CuDNN capture type.
        if not legacy_table_found:
            vendor_tracing_active = False
            try:
                cursor.execute(
                    "SELECT 1 FROM META_DATA_CAPTURE "
                    "WHERE name='CAPTURE_EVENT_TYPE' AND value IN ('CuBLAS', 'CuDNN') "
                    "LIMIT 1"
                )
                vendor_tracing_active = cursor.fetchone() is not None
            except sqlite3.OperationalError:
                pass  # META_DATA_CAPTURE absent → very old nsys or minimal trace

            if not vendor_tracing_active:
                conn.close()
                return None  # inconclusive: vendor tracing was not enabled

        # --- Step 3: new nsys — cuBLAS/cuBLASLt/cuDNN calls land in CUPTI_ACTIVITY_KIND_RUNTIME ---
        try:
            cursor.execute(
                "SELECT 1 FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON r.nameId = s.id "
                "WHERE s.value LIKE 'cublas%' OR s.value LIKE 'cublaslt%' "
                "OR s.value LIKE 'cudnn%' "
                "LIMIT 1"
            )
            if cursor.fetchone() is not None:
                conn.close()
                return True  # cuBLAS/cuBLASLt/cuDNN API was called
        except sqlite3.OperationalError:
            pass  # table absent — fall through to confirmed-negative

        conn.close()
        # Step 4: vendor tracing was confirmed active but no API calls found.
        # This is a confirmed negative (e.g. PyTorch 2.6+ bypasses cuBLAS C API).
        return False

    except Exception:
        pass
    # File unreadable or catastrophic error → inconclusive.
    return None


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


def annotate_sequences_with_culib_phases(sequences, culib_sequences, expected_num_sequences):
    """
    Attach pre-linked cuBLASLt phases to each sequence.
    """
    # Sanity: expect warmup + measured runs, and 1-1 mapping.
    assert (
        len(culib_sequences) == len(sequences) == expected_num_sequences
    ), f"culib/sequences length mismatch: {len(culib_sequences)} vs {len(sequences)} vs {expected_num_sequences}"

    for seq, culib in zip(sequences, culib_sequences):
        seq["culib"] = culib
        seq["culib_temp"] = culib["temperature"]

def get_culib_phase(name: str):
    """
    Parse a cuBLASLt NVTX name like 'lib:setup:cold' into (phase, temp).
    """
    _, phase, temp = name.split(":")
    return phase, temp


def link_culib_sequences(markers, phases):
    """
    Link per-run lib markers by phases and order.
    Returns a list of culib dicts aligned by time.
    """
    def strip(marker):
        return {
            "ts": marker["ts"],
            "dur": marker["dur"],
        }

    assert "run" in phases, "run phase required to derive temperature"
    
    markers_by_phase = {phase: [] for phase in phases}

    # Walk all markers once and bucket by phase.
    for marker in markers:
        phase, _ = get_culib_phase(marker["name"])
        markers_by_phase[phase].append(marker)

    # Sort each phase by time.
    for phase in phases:
        markers_by_phase[phase].sort(key=lambda m: m["ts"])

    culib_sequences = []
    phase_marker_lists = [markers_by_phase[phase] for phase in phases]
    run_idx = phases.index("run")

    # Ensure each phase has the same count to avoid silent truncation.
    lengths = {phase: len(lst) for phase, lst in zip(phases, phase_marker_lists)}
    expected_len = next(iter(lengths.values()))
    for phase, length in lengths.items():
        assert length == expected_len, (
            f"cuBLASLt phase count mismatch: {lengths}"
        )

    for markers in zip(*phase_marker_lists):
        run_marker = markers[run_idx]
        _, temp = get_culib_phase(run_marker["name"])

        # All phases should share the same temperature tag.
        for phase, marker in zip(phases, markers):
            _, phase_temp = get_culib_phase(marker["name"])
            assert phase_temp == temp, (
                f"Temperature mismatch across phases: run={temp}, {phase}={phase_temp}"
            )

        # Basic per-range sanity.
        for phase, marker in zip(phases, markers):
            assert marker["dur"] >= 0.0, f"{phase} has negative duration: {marker['dur']}"

        # Ordering sanity: each phase must fully precede the next.
        for (prev_phase, prev_marker), (phase, marker) in zip(
            zip(phases, markers), zip(phases[1:], markers[1:])
        ):
            prev_end = prev_marker["ts"] + prev_marker["dur"]
            marker_end = marker["ts"] + marker["dur"]
            assert prev_end <= marker["ts"] <= marker_end, (
                f"Unexpected cuBLASLt phase ordering between {prev_phase} and {phase}: "
                f"{prev_phase}=({prev_marker['ts']},{prev_end}), "
                f"{phase}=({marker['ts']},{marker_end})"
            )

        sequence = {"temperature": temp}
        for phase, marker in zip(phases, markers):
            sequence[phase] = strip(marker)
        culib_sequences.append(sequence)

    return culib_sequences

