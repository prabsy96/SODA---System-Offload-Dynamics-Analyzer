import json
import os
import re
import sys
from pathlib import Path
from soda import utils

def _to_tuple_int(x):
    if isinstance(x, (list, tuple)):
        try:
            return tuple(int(v) for v in x)
        except Exception:
            return tuple()
    return tuple()

def _norm_shared_mem(v):
    if v in (None, '0'):
        return 0
    try:
        return int(v)
    except Exception:
        return 0

def _parse_scalar(s):
    try:
        return float(s) if s not in (None, '') else None
    except (TypeError, ValueError):
        return None

def extract_config(k):
    return {
        "grid": _to_tuple_int(k.get("grid") or ()),
        "block": _to_tuple_int(k.get("block") or ()),
        "shared_memory": _norm_shared_mem(k.get("shared_memory")),
    }

def print_input_details(cpu_op):
    """Print input details for an operation."""
    input_dims = cpu_op.get("input_dims", [])
    input_strides = cpu_op.get("input_strides", [])
    input_types = cpu_op.get("input_type", [])
    print(f"\t* Input details:")
    print(f"\t\t** dims: {input_dims}")
    if input_strides:
        print(f"\t\t** strides: {input_strides}")
    if input_types:
        print(f"\t\t** types: {input_types}")

def print_launch_config_table(original_config, matched_kernel):
    """Print side-by-side comparison table of launch configurations."""
    print(f"\t* Launch configuration verification:")
    print(f"\t\t{'Field':<20} {'Original':<25} {'Replayed':<25}")
    print(f"\t\t{'-'*20} {'-'*25} {'-'*25}")
    
    orig_grid = str(original_config['grid'])
    orig_block = str(original_config['block'])
    orig_shared = original_config['shared_memory']
    orig_regs = original_config.get("registers_per_thread") or "N/A"
    orig_occ = original_config.get("occupancy") or "N/A"
    orig_stream = original_config.get("stream") or "N/A"
    orig_dur = original_config.get("dur") or "N/A"
    
    replayed_grid = str(matched_kernel['grid'])
    replayed_block = str(matched_kernel['block'])
    replayed_shared = matched_kernel.get('shared_memory') or 'N/A'
    replayed_regs = matched_kernel.get('registers_per_thread') or 'N/A'
    replayed_occ = matched_kernel.get('occupancy') or 'N/A'
    replayed_stream = matched_kernel.get('stream') or 'N/A'
    replayed_dur = matched_kernel.get('dur') or 'N/A'
    
    print(f"\t\t{'grid':<20} {orig_grid:<25} {replayed_grid:<25}")
    print(f"\t\t{'block':<20} {orig_block:<25} {replayed_block:<25}")
    print(f"\t\t{'shared_mem':<20} {orig_shared:<25} {replayed_shared:<25}")
    print(f"\t\t{'registers':<20} {orig_regs:<25} {replayed_regs:<25}")
    print(f"\t\t{'occupancy':<20} {orig_occ:<25} {replayed_occ:<25}")
    print(f"\t\t{'stream':<20} {orig_stream:<25} {replayed_stream:<25}")
    print(f"\t\t{'dur (μs)':<20} {orig_dur:<25} {replayed_dur:<25}")

def print_summary(matches, partial_matches, mismatches):
    """Print verification summary."""
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"\tExact matches:\t{matches}")
    print(f"\tPartial matches (name only):\t{partial_matches}")
    print(f"\tMismatches:\t{mismatches}")
    print(f"\tTotal event sequences:\t{matches + partial_matches + mismatches}")
    print("=" * 80)
    
    if matches == matches + partial_matches + mismatches:
        print("\nAll event sequences match exactly")
    elif matches + partial_matches == matches + partial_matches + mismatches:
        print("\nAll event sequences match at family level (some config differences)")
    else:
        print("\nSome event sequences don't match")

def find_matching_replayed_sequences(original_sequence, replayed_sequences):
    """Find replayed event sequences matching the original sequence's operation."""
    cpu_op = original_sequence.get("cpu_op")
    if not cpu_op:
        return []
    
    # Match by operation name, dims, strides, dtypes, and alpha/beta scalars
    matching_sequences = []
    for replayed_sequence in replayed_sequences["sequences"]:
        replayed_cpu_op = replayed_sequence.get("cpu_op")
        if not replayed_cpu_op or replayed_cpu_op["name"] != cpu_op["name"]:
            continue
        
        # Check input dimensions
        if replayed_cpu_op.get("input_dims") != cpu_op.get("input_dims"):
            continue
        
        # Check input strides (layout affects cuBLAS dispatch)
        if replayed_cpu_op.get("input_strides") != cpu_op.get("input_strides"):
            continue
        
        # Check input dtypes
        if replayed_cpu_op.get("input_type") != cpu_op.get("input_type"):
            continue
        
        # Check alpha/beta scalars for addmm (affects epilogue fusion)
        if cpu_op["name"] == "aten::addmm":
            orig_concrete = cpu_op.get("concrete_inputs", [])
            replayed_concrete = replayed_cpu_op.get("concrete_inputs", [])
            # For addmm: [bias, mat1, mat2, alpha, beta] - numeric compare
            if len(orig_concrete) >= 5 and len(replayed_concrete) >= 5:
                oa = _parse_scalar(orig_concrete[3]); ob = _parse_scalar(orig_concrete[4])
                ra = _parse_scalar(replayed_concrete[3]); rb = _parse_scalar(replayed_concrete[4])
                if (oa is not None and ra is not None and oa != ra) or \
                   (ob is not None and rb is not None and ob != rb):
                    continue
        
        matching_sequences.append(replayed_sequence)
    
    return matching_sequences

def verify_kernel_match(original_kernel, original_config, replayed_sequences):
    """Check if any replayed kernel matches the original kernel."""
    original_kernel_name = original_kernel["name"]
    original_kernel_name_clean = utils.clean_kernel_name(original_kernel_name)
    exact_match = False
    name_match = False
    matched_kernel = None
    matched_sequence = None
    
    # Check each replayed kernel for name and config match
    for replayed_sequence in replayed_sequences:
        replayed_kernel = replayed_sequence.get("kernel")
        if replayed_kernel is None:
            continue
        
        replayed_name = replayed_kernel["name"]
        replayed_name_clean = utils.clean_kernel_name(replayed_name)
        if replayed_name_clean == original_kernel_name_clean:
            name_match = True
            matched_kernel = replayed_kernel
            matched_sequence = replayed_sequence
            
            # Extract normalized configs for exact match comparison
            replayed_config = extract_config(replayed_kernel)
            original_config_norm = extract_config({
                "grid": original_config["grid"],
                "block": original_config["block"],
                "shared_memory": original_config.get("shared_memory", 0)
            })
            
            # Check exact config match
            if (replayed_config == original_config_norm):
                exact_match = True
                break
    
    return exact_match, name_match, matched_kernel, matched_sequence

def verify_event_sequences(original_sequences, replayed_sequences):
    """Verify each event sequence and return match statistics."""
    matches = 0
    partial_matches = 0
    mismatches = 0
    
    # Track which replayed sequences have been matched 
    used_replayed_indices = set()
    
    # Verify each original event sequence against replayed sequences
    for i, original_sequence in enumerate(original_sequences["sequences"]):
        cpu_op = original_sequence.get("cpu_op")
        original_kernel = original_sequence.get("kernel")
        
        if cpu_op is None or original_kernel is None:
            continue
        
        print(f"\n[{i+1}] Operation:\t{cpu_op['name']}")
        print_input_details(cpu_op)
        
        # Find matching replayed sequences by operation, dims, strides, dtypes, alpha/beta
        all_matching_sequences = find_matching_replayed_sequences(original_sequence, replayed_sequences)
        
        # Filter out already-used replayed sequences 
        matching_sequences = []
        for idx, sequence in enumerate(replayed_sequences["sequences"]):
            if idx not in used_replayed_indices and sequence in all_matching_sequences:
                matching_sequences.append(sequence)
        
        if not matching_sequences:
            print(f"\tNo matching replayed event sequences found")
            mismatches += 1
            continue
        
        # Extract original kernel configuration
        original_config = {
            "grid": original_kernel["grid"],
            "block": original_kernel["block"],
            "shared_memory": original_kernel.get("shared_memory", 0),
            "registers_per_thread": original_kernel.get("registers_per_thread"),
            "occupancy": original_kernel.get("occupancy"),
            "stream": original_kernel.get("stream"),
            "dur": original_kernel.get("avg_dur", original_kernel.get("dur"))
        }
        
        # Check if any replayed kernel matches
        exact_match, name_match, matched_kernel, matched_sequence = verify_kernel_match(
            original_kernel, original_config, matching_sequences
        )
        
        # Mark matched replayed sequence as used 
        if matched_sequence is not None:
            for idx, sequence in enumerate(replayed_sequences["sequences"]):
                if (sequence.get("cpu_op") == matched_sequence.get("cpu_op") and
                    sequence.get("kernel", {}).get("name") == matched_sequence.get("kernel", {}).get("name")):
                    used_replayed_indices.add(idx)
                    break
        
        print(f"\t* Kernel: {utils.clean_kernel_name(original_kernel['name'])}")
        
        if matched_kernel:
            print_launch_config_table(original_config, matched_kernel)
            print(f"\t* Match: {'Exact match' if exact_match else 'Partial match' if name_match else 'No match'}")
            
            if exact_match:
                matches += 1
            elif name_match:
                partial_matches += 1
        else:
            print(f"\t* Match: No match")
            print(f"\tReplayed event sequences found ({len(matching_sequences)}):")
            for replayed_sequence in matching_sequences[:3]:
                replayed_k = replayed_sequence.get("kernel", {})
                if replayed_k:
                    print(f"\t\t- {utils.clean_kernel_name(replayed_k.get('name', ''))}\tgrid={replayed_k.get('grid')}\tblock={replayed_k.get('block')}\tshared_mem={replayed_k.get('shared_memory', 'N/A')}")
            if len(matching_sequences) > 3:
                print(f"\t\t... and {len(matching_sequences) - 3} more")
            mismatches += 1
    
    return matches, partial_matches, mismatches