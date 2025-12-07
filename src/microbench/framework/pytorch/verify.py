import json
import os
import re
from typing import Dict, Any
from soda.common import utils
from soda.common import print_utils
from soda.common.data import Sequence

def compare_sequences(target_sequences: list, actual_sequences: list, title: str = "Actual vs Target", full: bool = True):
    """Compare each sequence (assumes 1:1 mapping by index).
    
    Args:
        target_sequences: List of Sequence objects (target)
        actual_sequences: List of Sequence objects (actual)
        title: Prefix for comparison table titles (e.g., "Pytorch vs Target" or "Baremetal vs Target")
        full: If True, compare all kernel fields including derived metrics. If False, only compare core identity fields.
    """
    # Sanity check: number of sequences must match
    assert len(target_sequences) == len(actual_sequences), "Number of sequences must match"
    
    summary = []
    for target_sequence, actual_sequence in zip(target_sequences, actual_sequences):
        
        # Skip None entries (eg, skipped baremetal jobs)
        if target_sequence is None or actual_sequence is None:
            continue
        
        # Get sequence string
        sequence = target_sequence.get_str()
        
        # Get aten_op and kernel
        target_aten_op = target_sequence.aten_op
        target_kernel = target_sequence.kernel
        actual_aten_op = actual_sequence.aten_op
        actual_kernel = actual_sequence.kernel
        
        # Print sequence string
        print_utils.iter_start(f"Sequence: {sequence}")
        
        # Compare aten_op input conditions 
        assert actual_aten_op is not None and target_aten_op is not None, "Missing aten_op in sequence"
        aten_op_match = actual_aten_op.compare(target_aten_op, show_table=False, title=f"{title} ATen op")
        assert aten_op_match, f"ATen op mismatch: target={target_aten_op.name}, actual={actual_aten_op.name}"
        
        # Compare kernels
        assert actual_kernel is not None and target_kernel is not None, "Missing kernel in sequence"
        results = actual_kernel.compare(target_kernel, show_table=False, title=f"{title} kernel", full=full)
        kernel_match = results["match"]
        assert kernel_match, f"Kernel mismatch: target={target_kernel.name}, actual={actual_kernel.name}"
        
        summary.append([sequence, aten_op_match, kernel_match])
    
    print_utils.comp_table("Sequence comparison", ["Sequence", "aten_op", "kernel"], summary)
