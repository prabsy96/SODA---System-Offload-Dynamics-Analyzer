"""NVBit in-process inference profiling wrapper.

Runs a full model inference under LD_PRELOAD=mem_reuse_tracker.so so that
NVBit can instrument the actual model.generate() call in execution order,
capturing L1/L2 cache state shaped by the real kernel sequence (shared expert
GEMMs -> routing -> routed expert GEMMs).

This is intentionally a separate subprocess from NCU isolation replay.
NVBit SASS instrumentation overhead (3-8x) would distort any timing or
compute-utilization measurements if combined with NCU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple


def nvbit_profile_inference(
    model_name: str,
    generation_config: Dict,
    expert_type_map: Dict[str, str],
    nvbit_lib_path: Path,
    output_log: Path,
    extra_env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = 600,
) -> Tuple[bool, Optional[Path], str]:
    """Run full model inference under NVBit LD_PRELOAD.

    The NVBit tool (mem_reuse_tracker.so) intercepts every cudaLaunchKernel,
    tags kernels by expert_type using the provided map, and records
    cache-line access patterns to output_log.

    Args:
        model_name: HuggingFace model name/path (e.g. "Qwen/Qwen1.5-MoE-A2.7B").
        generation_config: Dict with keys: batch_size, seq_len, max_new_tokens,
                           precision (e.g. "bfloat16").
        expert_type_map: Mapping of kernel_name -> expert_type for tagging.
                         Built from classify_kernel_entries() results.
        nvbit_lib_path: Path to compiled mem_reuse_tracker.so.
        output_log: Path where NVBit writes its JSON-lines output.
        extra_env: Additional environment variables to set/override.
        timeout: Subprocess timeout in seconds (default 600).

    Returns:
        (success, output_log_path, message)
    """
    nvbit_lib_path = Path(nvbit_lib_path)
    if not nvbit_lib_path.exists():
        return False, None, f"NVBit lib not found: {nvbit_lib_path}"

    # Write expert_type_map to a temp JSON file for the NVBit tool
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="soda_expert_map_"
    ) as f:
        json.dump(expert_type_map, f)
        map_path = f.name

    try:
        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)

        # NVBit tool reads these env vars at init
        env["LD_PRELOAD"] = str(nvbit_lib_path)
        env["SODA_NVBIT_LOG"] = str(output_log)
        env["SODA_EXPERT_MAP"] = map_path

        # Build the inference script inline
        batch_size = generation_config.get("batch_size", 1)
        seq_len = generation_config.get("seq_len", 128)
        max_new_tokens = generation_config.get("max_new_tokens", 1)
        precision = generation_config.get("precision", "bfloat16")

        dtype_map = {
            "bfloat16": "torch.bfloat16",
            "float16": "torch.float16",
            "float32": "torch.float32",
        }
        torch_dtype = dtype_map.get(precision, "torch.bfloat16")

        inference_script = (
            "import torch\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            f"model = AutoModelForCausalLM.from_pretrained(\n"
            f"    '{model_name}', torch_dtype={torch_dtype},\n"
            f"    device_map='auto', trust_remote_code=True)\n"
            f"tokenizer = AutoTokenizer.from_pretrained('{model_name}', trust_remote_code=True)\n"
            f"inputs = tokenizer(['hello world'] * {batch_size},\n"
            f"    return_tensors='pt', max_length={seq_len},\n"
            f"    padding='max_length', truncation=True)\n"
            f"inputs = {{k: v.cuda() for k, v in inputs.items()}}\n"
            f"# Warmup\n"
            f"with torch.no_grad():\n"
            f"    model.generate(**inputs, max_new_tokens={max_new_tokens})\n"
            f"torch.cuda.synchronize()\n"
            f"# Measured inference (NVBit instruments this)\n"
            f"with torch.no_grad():\n"
            f"    model.generate(**inputs, max_new_tokens={max_new_tokens})\n"
            f"torch.cuda.synchronize()\n"
        )

        cmd = [sys.executable, "-c", inference_script]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        if result.returncode != 0:
            msg = (
                f"NVBit inference failed (rc={result.returncode}):\n"
                f"{result.stderr[-2000:] if result.stderr else '(no stderr)'}"
            )
            return False, None, msg

        output_log = Path(output_log)
        if not output_log.exists() or output_log.stat().st_size == 0:
            return False, None, "NVBit log not written or empty"

        return True, output_log, "OK"

    except subprocess.TimeoutExpired:
        return False, None, f"NVBit inference timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return False, None, f"NVBit inference error: {exc}"
    finally:
        try:
            os.unlink(map_path)
        except OSError:
            pass


def build_expert_type_map(classified_kernels: Dict[str, Dict]) -> Dict[str, str]:
    """Build kernel_name -> expert_type mapping from classified kernel DB entries.

    The NVBit tool uses this map at launch-interception time to decide which
    kernels to instrument.

    Args:
        classified_kernels: List of classified entries from classify_kernel_entries().

    Returns:
        Dict mapping kernel name -> expert_type for non-"other" entries.
    """
    result: Dict[str, str] = {}
    for entry in classified_kernels:
        expert_type = entry.get("expert_type", "other")
        if expert_type == "other":
            continue
        kernel_name = entry.get("kernel", {}).get("name", "")
        if kernel_name:
            result[kernel_name] = expert_type
    return result
