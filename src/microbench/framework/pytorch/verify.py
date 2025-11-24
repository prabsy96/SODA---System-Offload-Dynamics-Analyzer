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

def compare_sequences(target_sequences, actual_sequences):
    """Compare each sequence (assumes 1:1 mapping by index)."""
    # Sanity check: number of sequences must match
    assert len(target_sequences["sequences"]) == len(actual_sequences["sequences"]), "Number of sequences must match"
    
    summary = []
    for target_sequence, actual_sequence in zip(target_sequences["sequences"], actual_sequences["sequences"]):
        target_cpu_op = target_sequence.get("cpu_op")
        target_kernel = target_sequence.get("kernel")
        actual_cpu_op = actual_sequence.get("cpu_op")
        actual_kernel = actual_sequence.get("kernel")
        
        # Build sequence string
        sequence = utils.get_sequence_str(target_sequence)
        
        # Print sequence string
        print_utils.iter_start(f"Sequence: {sequence}")
        
        # Compare cpu_op input conditions
        cpu_op_match = utils.compare_cpu_ops(actual_cpu_op, target_cpu_op, show_table=True, title="Pytorch vs Target CPU op")
        assert cpu_op_match, f"CPU op mismatch: target={target_cpu_op.get('name')}, actual={actual_cpu_op.get('name')}"
        
        # Compare kernels
        assert actual_kernel is not None, f"Missing kernel in actual sequence"
        results = utils.compare_kernels(actual=actual_kernel, target=target_kernel, show_table=True, title="Pytorch vs Target kernel")
        kernel_match = results.get("match", False)
        assert kernel_match, f"Kernel mismatch: target={utils.clean_kernel_name(target_kernel['name'])}, actual={utils.clean_kernel_name(actual_kernel['name'])}"
        
        summary.append([sequence, cpu_op_match, kernel_match])
        
        print_utils.iter_end()
    
    print_utils.comp_table("Sequence comparison", ["Sequence", "cpu_op", "kernel"], summary)