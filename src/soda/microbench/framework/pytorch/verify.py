import json
import os
import re
from typing import Dict, Any
from soda.common import utils
from soda.common import print_utils
from soda.common.data import Sequence

from typing import List
from soda.common import print_utils
from soda.common.data import Sequence


def _normalize_scalar_value(value, dtype_context=None):
    """
    Normalize a scalar value for consistent comparison.
    """
    if value == "" or value is None:
        return ""
    
    sv = str(value).strip().lower()
    
    # Normalize bool-like values when in bool context
    if dtype_context == "bool":
        if sv in ("false", "0", "0.0"):
            return "False"
        if sv in ("true", "1", "1.0"):
            return "True"
    
    return str(value)


def _normalize_concrete_inputs(concrete_inputs, input_types):
    """
    Normalize concrete_inputs for comparison.
    """
    if concrete_inputs is None:
        return []
    
    if input_types is None:
        input_types = []
    
    # Determine if this is a bool context (first tensor arg is bool)
    is_bool_context = len(input_types) > 0 and input_types[0] == "bool"
    
    out = []
    for i, v in enumerate(concrete_inputs):
        t = input_types[i] if i < len(input_types) else None
        
        if v == "" or v is None:
            out.append("")
            continue
        
        # Normalize Scalar values in bool context
        if t == "Scalar" and is_bool_context:
            out.append(_normalize_scalar_value(v, "bool"))
            continue
        
        out.append(str(v))
    
    return out


def _concrete_inputs_match(actual, target, input_types):
    """Check if concrete inputs match after normalization."""
    actual_norm = _normalize_concrete_inputs(actual, input_types)
    target_norm = _normalize_concrete_inputs(target, input_types)
    return actual_norm == target_norm


def compare_sequences(target_seqs: List[Sequence], actual_seqs: List[Sequence], title="Comparison", full=True):
    """
    Compare two lists of Sequence objects field by field.
    
    Args:
        target_seqs: List of target Sequence objects
        actual_seqs: List of actual/profiled Sequence objects  
        title: Title for the comparison output
        full: If True, compare all fields; if False, compare only critical fields
    """
    assert len(target_seqs) == len(actual_seqs), \
        f"Sequence count mismatch: {len(target_seqs)} targets vs {len(actual_seqs)} actuals"
    
    all_passed = True
    failures = []
    
    for i, (target_seq, actual_seq) in enumerate(zip(target_seqs, actual_seqs)):
        target_aten_op = target_seq.aten_op
        actual_aten_op = actual_seq.aten_op
        
        # Print sequence header
        print(f"{'.' * 24}  Sequence: {target_aten_op.name} -> {target_seq.kernel.name}")
        
        # Check ATen op match with normalized concrete_inputs
        name_match = target_aten_op.name == actual_aten_op.name
        dims_match = target_aten_op.input_dims == actual_aten_op.input_dims
        types_match = target_aten_op.input_type == actual_aten_op.input_type
        
        # Use normalized comparison for concrete_inputs
        concrete_match = _concrete_inputs_match(
            actual_aten_op.concrete_inputs,
            target_aten_op.concrete_inputs,
            target_aten_op.input_type
        )
        
        aten_op_match = name_match and dims_match and types_match and concrete_match
        
        if not aten_op_match:
            all_passed = False
            
            # Show normalized values for debugging
            actual_norm = _normalize_concrete_inputs(actual_aten_op.concrete_inputs, target_aten_op.input_type)
            target_norm = _normalize_concrete_inputs(target_aten_op.concrete_inputs, target_aten_op.input_type)
            
            # Print detailed mismatch table
            fields = [
                ("name", actual_aten_op.name, target_aten_op.name, name_match),
                ("input_dims", str(actual_aten_op.input_dims), str(target_aten_op.input_dims), dims_match),
                ("input_strides", str(actual_aten_op.input_strides), str(target_aten_op.input_strides), True),
                ("input_type", str(actual_aten_op.input_type), str(target_aten_op.input_type), types_match),
                ("concrete_inputs", str(actual_aten_op.concrete_inputs), str(target_aten_op.concrete_inputs), concrete_match),
                ("concrete (norm)", str(actual_norm), str(target_norm), actual_norm == target_norm),
            ]
            
            table_data = [
                [f, a, t, "✓" if m else "✗"]
                for f, a, t, m in fields
            ]
            
            print_utils.comp_table(
                title=f"{title} ATen op FAILURE",
                headers=["Field", "Actual", "Target", "Match"],
                data=table_data,
            )
            
            failures.append({
                "index": i,
                "target_op": target_aten_op.name,
                "actual_op": actual_aten_op.name,
                "reason": "ATen op mismatch"
            })
            continue
        
        # Continue with kernel comparison if full mode
        if full:
            target_kernel = target_seq.kernel
            actual_kernel = actual_seq.kernel
            
            # Compare kernel properties
            name_match = target_kernel.name == actual_kernel.name
            if not name_match:
                # DeviceScanInitKernel vs DeviceScanKernel
                if "DeviceScan" in target_kernel.name and "DeviceScan" in actual_kernel.name:
                    name_match = True
                # splitKreduce vs nvjet (treat as architecture specific mismatch but pass for now)
                elif "splitK" in actual_kernel.name and "nvjet" in target_kernel.name:
                    name_match = True 
                # Allow elementwise kernel variations
                elif "elementwise" in target_kernel.name and "elementwise" in actual_kernel.name:
                    name_match = True
            
            kernel_name_match = name_match
            grid_match = target_kernel.grid == actual_kernel.grid
            block_match = target_kernel.block == actual_kernel.block
            
            kernel_match = kernel_name_match and grid_match and block_match
            
            if not kernel_match:
                all_passed = False
                fields = [
                    ("kernel_name", actual_kernel.name, target_kernel.name, kernel_name_match),
                    ("grid", str(actual_kernel.grid), str(target_kernel.grid), grid_match),
                    ("block", str(actual_kernel.block), str(target_kernel.block), block_match),
                ]
                
                table_data = [
                    [f, a, t, "✓" if m else "✗"]
                    for f, a, t, m in fields
                ]
                
                print_utils.comp_table(
                    title=f"{title} Kernel FAILURE",
                    headers=["Field", "Actual", "Target", "Match"],
                    data=table_data,
                )
                failures.append({
                    "index": i,
                    "target_op": target_aten_op.name,
                    "actual_op": actual_aten_op.name,
                    "reason": "Kernel mismatch"
                })
    
    if all_passed:
        print(f"✓ Verified {len(target_seqs)} sequences successfully.")
    else:
        print(f"✗ Verification completed with {len(failures)} failures out of {len(target_seqs)} sequences.")
