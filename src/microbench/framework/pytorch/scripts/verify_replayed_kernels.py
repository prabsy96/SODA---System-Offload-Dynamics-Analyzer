import json
import os
import re
import sys

def get_kernel_short_name(kernel_name):
    """Extract a concise kernel name from the full signature."""
    match = re.search(r'^([^<(]+)', kernel_name)
    if match:
        short = match.group(1).strip()
    else:
        short = kernel_name
    # Remove any stray 'void' from the short name
    return short.replace('void', '').strip()

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
    orig_dur = original_config.get("duration_us") or "N/A"
    
    replayed_grid = str(matched_kernel['grid'])
    replayed_block = str(matched_kernel['block'])
    replayed_shared = matched_kernel.get('shared_memory') or 'N/A'
    replayed_regs = matched_kernel.get('registers_per_thread') or 'N/A'
    replayed_occ = matched_kernel.get('occupancy') or 'N/A'
    replayed_stream = matched_kernel.get('stream') or 'N/A'
    replayed_dur = matched_kernel.get('duration_us') or 'N/A'
    
    print(f"\t\t{'grid':<20} {orig_grid:<25} {replayed_grid:<25}")
    print(f"\t\t{'block':<20} {orig_block:<25} {replayed_block:<25}")
    print(f"\t\t{'shared_mem':<20} {orig_shared:<25} {replayed_shared:<25}")
    print(f"\t\t{'registers':<20} {orig_regs:<25} {replayed_regs:<25}")
    print(f"\t\t{'occupancy':<20} {orig_occ:<25} {replayed_occ:<25}")
    print(f"\t\t{'stream':<20} {orig_stream:<25} {replayed_stream:<25}")
    print(f"\t\t{'duration_us':<20} {orig_dur:<25} {replayed_dur:<25}")

def print_summary(matches, partial_matches, mismatches):
    """Print verification summary."""
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"\tExact matches:\t{matches}")
    print(f"\tPartial matches (name only):\t{partial_matches}")
    print(f"\tMismatches:\t{mismatches}")
    print(f"\tTotal kernel chains:\t{matches + partial_matches + mismatches}")
    print("=" * 80)
    
    if matches == matches + partial_matches + mismatches:
        print("\nAll kernel chains match exactly")
    elif matches + partial_matches == matches + partial_matches + mismatches:
        print("\nAll kernel chains match at family level (some config differences)")
    else:
        print("\nSome kernel chains don't match")

def find_matching_replayed_chains(original_chain, replayed_kernel_chains):
    """Find replayed kernel chains matching the original chain's operation."""
    cpu_op = original_chain.get("cpu_op")
    if not cpu_op:
        return []
    
    # Match by operation name, dims, strides, dtypes, and alpha/beta scalars
    matching_chains = []
    for replayed_chain in replayed_kernel_chains["causal_chains"]:
        replayed_cpu_op = replayed_chain.get("cpu_op")
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
        
        matching_chains.append(replayed_chain)
    
    return matching_chains

def verify_kernel_match(original_kernel, original_config, replayed_chains):
    """Check if any replayed kernel matches the original kernel."""
    original_kernel_name = original_kernel["name"]
    exact_match = False
    name_match = False
    matched_kernel = None
    matched_chain = None
    
    # Check each replayed kernel for name and config match
    for replayed_chain in replayed_chains:
        replayed_kernel = replayed_chain.get("kernel")
        if replayed_kernel is None:
            continue
        
        replayed_name = replayed_kernel["name"]
        if replayed_name == original_kernel_name:
            name_match = True
            matched_kernel = replayed_kernel
            matched_chain = replayed_chain
            
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
    
    return exact_match, name_match, matched_kernel, matched_chain

def verify_kernel_chains(original_kernel_chains, replayed_kernel_chains):
    """Verify each kernel chain and return match statistics."""
    matches = 0
    partial_matches = 0
    mismatches = 0
    
    # Track which replayed chains have been matched 
    used_replayed_indices = set()
    
    # Verify each original kernel chain against replayed chains
    for i, original_chain in enumerate(original_kernel_chains["causal_chains"]):
        cpu_op = original_chain.get("cpu_op")
        original_kernel = original_chain.get("kernel")
        
        if cpu_op is None or original_kernel is None:
            continue
        
        print(f"\n[{i+1}] Operation:\t{cpu_op['name']}")
        print_input_details(cpu_op)
        
        # Find matching replayed chains by operation, dims, strides, dtypes, alpha/beta
        all_matching_chains = find_matching_replayed_chains(original_chain, replayed_kernel_chains)
        
        # Filter out already-used replayed chains 
        replayed_chains = []
        for idx, chain in enumerate(replayed_kernel_chains["causal_chains"]):
            if idx not in used_replayed_indices and chain in all_matching_chains:
                replayed_chains.append(chain)
        
        if not replayed_chains:
            print(f"\tNo matching replayed kernel chains found")
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
            "duration_us": original_kernel.get("avg_duration_us") or original_kernel.get("duration_us")
        }
        
        # Check if any replayed kernel matches
        exact_match, name_match, matched_kernel, matched_chain = verify_kernel_match(
            original_kernel, original_config, replayed_chains
        )
        
        # Mark matched replayed chain as used 
        if matched_chain is not None:
            for idx, chain in enumerate(replayed_kernel_chains["causal_chains"]):
                if (chain.get("cpu_op") == matched_chain.get("cpu_op") and
                    chain.get("kernel", {}).get("name") == matched_chain.get("kernel", {}).get("name")):
                    used_replayed_indices.add(idx)
                    break
        
        print(f"\t* Kernel: {get_kernel_short_name(original_kernel['name'])}")
        
        if matched_kernel:
            print_launch_config_table(original_config, matched_kernel)
            print(f"\t* Match: {'Exact match' if exact_match else 'Partial match' if name_match else 'No match'}")
            
            if exact_match:
                matches += 1
            elif name_match:
                partial_matches += 1
        else:
            print(f"\t* Match: No match")
            print(f"\tReplayed kernel chains found ({len(replayed_chains)}):")
            for replayed_chain in replayed_chains[:3]:
                replayed_k = replayed_chain.get("kernel", {})
                if replayed_k:
                    print(f"\t\t- {get_kernel_short_name(replayed_k.get('name', ''))}\tgrid={replayed_k.get('grid')}\tblock={replayed_k.get('block')}\tshared_mem={replayed_k.get('shared_memory', 'N/A')}")
            if len(replayed_chains) > 3:
                print(f"\t\t... and {len(replayed_chains) - 3} more")
            mismatches += 1
    
    return matches, partial_matches, mismatches

def run_verification_pipeline(original_file, replayed_file):
    """
    Main pipeline: load -> verify -> save results.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load kernel chain files
    # Resolve file paths (absolute or relative to output dir)
    if os.path.isabs(original_file) or os.path.exists(original_file):
        original_path = original_file
    else:
        original_path = os.path.join(output_dir, original_file)
    
    if os.path.isabs(replayed_file) or os.path.exists(replayed_file):
        replayed_path = replayed_file
    else:
        replayed_path = os.path.join(output_dir, replayed_file)
    
    # Load original and replayed kernel chain data
    with open(original_path, "r") as f:
        original_kernel_chains = json.load(f)
    
    with open(replayed_path, "r") as f:
        replayed_kernel_chains = json.load(f)
    
    # Step 2: Setup logging
    # Redirect print to both stdout and log file
    log_path = os.path.join(output_dir, "verify_replayed_kernels.log")
    output_file = open(log_path, "w")
    
    import builtins
    original_print = builtins.print
    
    def print_and_write(*args, **kwargs):
        """Print to stdout and write to file."""
        original_print(*args, **kwargs)
        line = ' '.join(str(arg) for arg in args)
        output_file.write(line + '\n')
        output_file.flush()
    
    builtins.print = print_and_write
    
    # Step 3: Verify kernel chains
    print("=" * 80)
    print("Kernel chain verification: original vs replayed kernel chains")
    print("=" * 80)
    
    print("\n1. Metadata verification:")
    print(f"\tOriginal kernel chains:\t{len(original_kernel_chains['causal_chains'])} chains")
    print(f"\tReplayed kernel chains:\t{len(replayed_kernel_chains['causal_chains'])} chains")
    
    print("\n2. Kernel chain-by-chain verification:")
    print("-" * 80)
    
    matches, partial_matches, mismatches = verify_kernel_chains(
        original_kernel_chains, replayed_kernel_chains
    )
    
    # Step 4: Print summary
    print_summary(matches, partial_matches, mismatches)
    
    # Restore original print and close file
    builtins.print = original_print
    output_file.close()
    print(f"\nVerification output saved to {log_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_replayed_kernels.py <original_file> <replayed_file>")
        print("Example: python verify_replayed_kernels.py unique_gemm_kernel_chains.json replayed_gemm_kernel_chains.json")
        sys.exit(1)
    
    original_file = sys.argv[1]
    replayed_file = sys.argv[2]
    run_verification_pipeline(original_file, replayed_file)
