import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any
from soda import utils
# Add src to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common import print_utils
from data import Sequence

def compare_sequences(target_sequences: list, actual_sequences: list):
    """Compare each sequence (assumes 1:1 mapping by index).
    
    Args:
        target_sequences: List of Sequence objects (target)
        actual_sequences: List of Sequence objects (actual)
    """
    # Sanity check: number of sequences must match
    assert len(target_sequences) == len(actual_sequences), "Number of sequences must match"
    
    summary = []
    for target_sequence, actual_sequence in zip(target_sequences, actual_sequences):
        
        # Get sequence string
        sequence = target_sequence.get_str()
        
        # Get cpu_op and kernel
        target_cpu_op = target_sequence.cpu_op
        target_kernel = target_sequence.kernel
        actual_cpu_op = actual_sequence.cpu_op
        actual_kernel = actual_sequence.kernel
        
        # Print sequence string
        print_utils.iter_start(f"Sequence: {sequence}")
        
        # Compare cpu_op input conditions 
        assert actual_cpu_op is not None and target_cpu_op is not None, "Missing cpu_op in sequence"
        cpu_op_match = actual_cpu_op.compare(target_cpu_op, show_table=True, title="Pytorch vs Target CPU op")
        assert cpu_op_match, f"CPU op mismatch: target={target_cpu_op.name}, actual={actual_cpu_op.name}"
        
        # Compare kernels
        assert actual_kernel is not None and target_kernel is not None, "Missing kernel in sequence"
        results = actual_kernel.compare(target_kernel, show_table=True, title="Pytorch vs Target kernel", full=True)
        kernel_match = results["match"]
        assert kernel_match, f"Kernel mismatch: target={target_kernel.name}, actual={actual_kernel.name}"
        
        summary.append([sequence, cpu_op_match, kernel_match])
    
    print_utils.comp_table("Sequence comparison", ["Sequence", "cpu_op", "kernel"], summary)