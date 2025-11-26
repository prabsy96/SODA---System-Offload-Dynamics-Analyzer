"""
Data structures for SODA kernel configurations and sequences.
"""

from typing import Optional, List, Tuple, Any, Dict
from common import print_utils


# Utility functions (moved here to avoid circular import with utils.py)
def clean_kernel_name(kernel_name: str) -> str:
    """
    Extract a clean kernel name from the full signature.
    
    Args:
        kernel_name: Full kernel name (may be a C++ function signature).
    
    Returns:
        Clean kernel name (just the kernel name, no namespace or template parameters).
    
    Examples:
        "void at::native::vectorized_elementwise_kernel<4, ...>" 
        -> "vectorized_elementwise_kernel"
        
        "void at::native::(anonymous namespace)::elementwise_kernel<...>"
        -> "elementwise_kernel"
        
        "turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn"
        -> "turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn"
    """
    # Extract everything before '<' (removes template parameters)
    # This handles cases where '(' appears in template params like "(anonymous namespace)"
    if not kernel_name:
        return None
    
    if '<' in kernel_name:
        clean_kernel_name = kernel_name.split('<')[0].strip()
    elif '(' in kernel_name:
        # If no '<' but has '(', extract before '(' (function parameters)
        clean_kernel_name = kernel_name.split('(')[0].strip()
    else:
        clean_kernel_name = kernel_name
    
    # Remove 'void' prefix if present
    clean_kernel_name = clean_kernel_name.replace('void', '').strip()
    
    # Extract just the kernel name (last part after '::')
    if '::' in clean_kernel_name:
        clean_kernel_name = clean_kernel_name.split('::')[-1]
    
    return clean_kernel_name.strip()

def to_tuple_int(x):
    """Convert list/tuple to tuple of ints for normalized comparison."""
    if isinstance(x, (list, tuple)):
        try:
            return tuple(int(v) for v in x)
        except Exception:
            return tuple()
    return tuple()

def norm_shared_mem(v):
    """Normalize shared memory value to int."""
    if v in (None, '0'):
        return 0
    try:
        return int(v)
    except Exception:
        return 0

class Kernel:
    """Kernel configuration class."""
    def __init__(self, 
                name: str, 
                grid: Optional[List[int]] = None, 
                block: Optional[List[int]] = None, 
                shared_memory: Optional[int] = None,
                registers_per_thread: Optional[int] = None,
                occupancy: Optional[float] = None,
                stream: Optional[int] = None,
                dur: Optional[float] = None,
                device: Optional[int] = None,
                context: Optional[int] = None,
                queued: Optional[bool] = None,
                 blocks_per_SM: Optional[float] = None,
                 warps_per_SM: Optional[float] = None,

                 # Metadata fields (not used for comparison)
                 type: Optional[str] = None,
                 external_id: Optional[int] = None,
                 correlation: Optional[int] = None,
                 ts: Optional[float] = None,
                 # Aggregated duration fields (not used for comparison)
                 avg_dur: Optional[float] = None,
                 min_dur: Optional[float] = None,
                 max_dur: Optional[float] = None,
                 all_dur: Optional[List[float]] = None):
        """Initialize kernel configuration with normalization."""
        # Normalize name
        self.name = clean_kernel_name(name) if name else "unknown"
        
        # Apply defaults if None
        if grid is None:
            grid = [1, 1, 1]  # Minimum valid CUDA grid dimensions
        if block is None:
            block = [1, 1, 1]  # Minimum valid CUDA block dimensions
        if shared_memory is None:
            shared_memory = 0
        
        # Normalize grid and block to tuples of ints
        self.grid = to_tuple_int(grid)
        self.block = to_tuple_int(block)
        
        # Normalize shared memory
        self.shared_memory = norm_shared_mem(shared_memory)
        
        # Optional performance fields
        self.registers_per_thread = registers_per_thread
        self.occupancy = occupancy
        self.stream = stream
        self.dur = dur
        self.device = device
        self.context = context
        self.queued = queued
        self.blocks_per_SM = blocks_per_SM
        self.warps_per_SM = warps_per_SM
        
        # Metadata fields (not used for comparison)
        self.type = type
        self.external_id = external_id
        self.correlation = correlation
        self.ts = ts
        
        # Aggregated duration fields (not used for comparison)
        self.avg_dur = avg_dur
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.all_dur = all_dur
    
    def print(self, title: str = "Target kernel"):
        """Print kernel configuration as a table."""
        data = [
            ["name", self.name],
            ["grid", str(list(self.grid))],
            ["block", str(list(self.block))],
            ["shared_memory", str(self.shared_memory)],
        ]
        
        # Add optional fields if they exist
        if self.registers_per_thread is not None:
            data.append(["registers_per_thread", str(self.registers_per_thread)])
        if self.occupancy is not None:
            data.append(["occupancy", str(self.occupancy)])
        if self.stream is not None:
            data.append(["stream", str(self.stream)])
        if self.device is not None:
            data.append(["device", str(self.device)])
        if self.context is not None:
            data.append(["context", str(self.context)])
        if self.queued is not None:
            data.append(["queued", str(self.queued)])
        if self.blocks_per_SM is not None:
            data.append(["blocks_per_SM", str(self.blocks_per_SM)])
        if self.warps_per_SM is not None:
            data.append(["warps_per_SM", str(self.warps_per_SM)])
        
        print_utils.comp_table(title, ["Field", "Value"], data)
    
    def get_signature(self, full: bool = False) -> dict:
        """
        Extract kernel signature for comparison.
        
        Args:
            full: If True, include additional signature fields
        
        Returns:
            Dictionary with kernel signature fields:
            - name, grid, block, shared_memory (always included)
            - registers_per_thread, occupancy, stream, dur (if full=True)
            - Excludes: type, external_id, correlation, ts, avg_dur, min_dur, max_dur, all_dur
        """
        signature = {
            "name": self.name,
            "grid": self.grid,
            "block": self.block,
            "shared_memory": self.shared_memory,
        }
        
        if full:
            # Include all optional performance fields explicitly (excluding metadata/aggregated fields)
            if self.registers_per_thread is not None:
                signature["registers_per_thread"] = self.registers_per_thread
            if self.occupancy is not None:
                signature["occupancy"] = self.occupancy
            if self.stream is not None:
                signature["stream"] = self.stream
            if self.device is not None:
                signature["device"] = self.device
            if self.context is not None:
                signature["context"] = self.context
            if self.queued is not None:
                signature["queued"] = self.queued
            if self.blocks_per_SM is not None:
                signature["blocks_per_SM"] = self.blocks_per_SM
            if self.warps_per_SM is not None:
                signature["warps_per_SM"] = self.warps_per_SM
        
        return signature
    
    def compare(self, other: 'Kernel', show_table: bool = False, title: str = "Kernel comparison", full: bool = False) -> Dict[str, Any]:
        """Compare this kernel with another kernel and return hierarchical match results.
        
        All fields are shown in the comparison table, but only certain fields count toward
        the overall match result based on the `full` parameter.
        
        Args:
            other: Other Kernel object to compare against
            show_table: If True, print comparison table with match indicators
            title: Optional custom title for the comparison table
            full: If True, match result includes additional performance fields (occupancy, 
                  stream, device, context, queued, blocks_per_SM, warps_per_SM). When False, match 
                  result includes basic fields (name, grid, block, shared_memory, registers_per_thread).
                  registers_per_thread is always compared if available (since it can be extracted from traces).
                  Metadata fields (type, external_id, correlation) are shown but never count toward match.
                  Duration fields (dur, ts, avg_dur, min_dur, max_dur, all_dur) are never shown or compared.
        
        Returns:
            Dictionary with match results:
            - "match": overall boolean match (based on full parameter)
            - "name": boolean match
            - "grid": [x_match, y_match, z_match] list of booleans
            - "block": [x_match, y_match, z_match] list of booleans
            - other fields: boolean matches (all shown regardless of full parameter)
        """
        if other is None:
            return {"match": False}
        
        match = True
        table_data = []
        results = {}
        
        # Compare name
        name_match = (self.name == other.name)
        match = match and name_match
        results["name"] = name_match
        table_data.append(["name", self.name, other.name, name_match])
        
        # Compare grid (per-dimension)
        grid_x_match = len(self.grid) > 0 and len(other.grid) > 0 and self.grid[0] == other.grid[0]
        grid_y_match = len(self.grid) > 1 and len(other.grid) > 1 and self.grid[1] == other.grid[1]
        grid_z_match = len(self.grid) > 2 and len(other.grid) > 2 and self.grid[2] == other.grid[2]
        grid_match = [grid_x_match, grid_y_match, grid_z_match]
        match = match and all(grid_match)
        results["grid"] = grid_match
        table_data.append(["grid", self.grid, other.grid, grid_match])
        
        # Compare block (per-dimension)
        block_x_match = len(self.block) > 0 and len(other.block) > 0 and self.block[0] == other.block[0]
        block_y_match = len(self.block) > 1 and len(other.block) > 1 and self.block[1] == other.block[1]
        block_z_match = len(self.block) > 2 and len(other.block) > 2 and self.block[2] == other.block[2]
        block_match = [block_x_match, block_y_match, block_z_match]
        match = match and all(block_match)
        results["block"] = block_match
        table_data.append(["block", self.block, other.block, block_match])
        
        # Compare shared_memory
        shared_memory_match = (self.shared_memory == other.shared_memory)
        match = match and shared_memory_match
        results["shared_memory"] = shared_memory_match
        table_data.append(["shared_memory", self.shared_memory, other.shared_memory, shared_memory_match])
        
        # Compare registers_per_thread 
        registers_match = (self.registers_per_thread == other.registers_per_thread)
        match = match and registers_match
        results["registers_per_thread"] = registers_match
        table_data.append(["registers_per_thread", self.registers_per_thread, other.registers_per_thread, registers_match])
        
        # Additional fields (only shown and compared if full=True)
        if full:
            fields = [
                ("occupancy", self.occupancy, other.occupancy),
                ("stream", self.stream, other.stream),
                ("device", self.device, other.device),
                ("context", self.context, other.context),
                ("queued", self.queued, other.queued),
                ("blocks_per_SM", self.blocks_per_SM, other.blocks_per_SM),
                ("warps_per_SM", self.warps_per_SM, other.warps_per_SM),
            ]
            
            for field, self_val, other_val in fields:
                field_match = self_val == other_val
                match = match and field_match
                results[field] = field_match
                table_data.append([field, self_val, other_val, field_match])
        
        # Note: Never show or compare metadata fields (type, external_id, correlation)
        # Note: Never show or compare duration fields (dur, ts, avg_dur, min_dur, max_dur, all_dur)
        
        results["match"] = match
        
        if show_table:
            outcome = "[green]SUCCESS[/green]" if match else "[red]FAILURE[/red]"
            print_utils.comp_table(f"{title} {outcome}", ["Field", "Actual", "Target", "Match"], table_data)
        
        return results
    
    @classmethod
    def from_dict(cls, kernel_dict: Optional[Dict[str, Any]]) -> Optional['Kernel']:
        """Create Kernel from dictionary."""
        if not kernel_dict:
            return None
        return cls(
            name=kernel_dict.get("name"),
            grid=kernel_dict.get("grid"),
            block=kernel_dict.get("block"),
            shared_memory=kernel_dict.get("shared_memory"),
            registers_per_thread=kernel_dict.get("registers_per_thread"),
            occupancy=kernel_dict.get("occupancy"),
            stream=kernel_dict.get("stream"),
            device=kernel_dict.get("device"),
            context=kernel_dict.get("context"),
            queued=kernel_dict.get("queued"),
            blocks_per_SM=kernel_dict.get("blocks_per_SM"),
            warps_per_SM=kernel_dict.get("warps_per_SM"),
            # Metadata 
            type=kernel_dict.get("type"),
            external_id=kernel_dict.get("external_id"),
            correlation=kernel_dict.get("correlation"),
            ts=kernel_dict.get("ts"),
            # Duration fields
            dur=kernel_dict.get("dur"),
            avg_dur=kernel_dict.get("avg_dur"),
            min_dur=kernel_dict.get("min_dur"),
            max_dur=kernel_dict.get("max_dur"),
            all_dur=kernel_dict.get("all_dur")
        )


class CPUOp:
    """CPU operation class."""
    def __init__(self, name: str,
                 input_dims: Optional[List[List[int]]] = None,
                 input_strides: Optional[List[List[int]]] = None,
                 input_type: Optional[List[str]] = None,
                 concrete_inputs: Optional[List[Any]] = None,
                 ts: Optional[float] = None,
                 dur: Optional[float] = None,
                 external_id: Optional[int] = None):
        """Initialize CPU operation."""
        self.name = name or ""
        self.input_dims = input_dims or []
        self.input_strides = input_strides or []
        self.input_type = input_type or []
        self.concrete_inputs = concrete_inputs or []
        self.ts = ts
        self.dur = dur
        self.external_id = external_id
    
    def get_alpha_beta(self, default_alpha: float = 1.0, default_beta: float = 1.0) -> Tuple[float, float]:
        """Extract alpha and beta scalars from concrete_inputs for addmm operations.
        
        Args:
            default_alpha: Default alpha value if not found (default: 1.0)
            default_beta: Default beta value if not found (default: 1.0)
        
        Returns:
            Tuple of (alpha, beta) floats
        """
        def _parse_scalar(value, default):
            if value is None or value == '':
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        
        # Alpha is at index 3, beta is at index 4
        if len(self.concrete_inputs) >= 5:
            alpha = _parse_scalar(self.concrete_inputs[3], default_alpha)
            beta = _parse_scalar(self.concrete_inputs[4], default_beta)
        else:
            alpha = default_alpha
            beta = default_beta
        
        return alpha, beta
    
    def get_signature(self) -> dict:
        """
        Extract canonical operation signature (input conditions) from cpu_op.
        
        This signature uniquely identifies an operation's inputs and determines
        which kernel should be dispatched. Used consistently across:
        - Matching operations (verify.py)
        - Op signature extraction (report.py)  
        - Job generation (generate.py)
        
        Returns:
            Dictionary with canonical signature fields:
            - name: operation name
            - input_dims: input dimensions 
            - input_strides: input strides 
            - input_type: input types 
            - concrete_inputs: concrete input values 
        """
        return {
            "name": self.name,
            "input_dims": self.input_dims,
            "input_strides": self.input_strides,
            "input_type": self.input_type,
            "concrete_inputs": self.concrete_inputs,
        }
    
    def compare(self, other: 'CPUOp', show_table: bool = False, title: str = "CPU op comparison") -> bool:
        """Compare this CPU op with another CPU op and return True if all fields match.
        
        Compares: name, input_dims, input_strides, input_type, and concrete_inputs
        (alpha/beta scalars for addmm operations).
        
        Args:
            other: Other CPUOp object to compare against
            show_table: If True, print comparison table with match indicators
            title: Optional custom title for the comparison table
        
        Returns:
            True if all fields match, False otherwise.
        """
        if other is None:
            return False
        
        match = True
        table_data = []
        
        # Compare name
        name_match = (self.name == other.name)
        match = match and name_match
        table_data.append(["name", self.name, other.name, name_match])
        
        # Compare input_dims
        dims_match = (self.input_dims == other.input_dims)
        match = match and dims_match
        table_data.append(["input_dims", self.input_dims, other.input_dims, dims_match])
        
        # Compare input_strides
        strides_match = (self.input_strides == other.input_strides)
        match = match and strides_match
        table_data.append(["input_strides", self.input_strides, other.input_strides, strides_match])
        
        # Compare input_type
        type_match = (self.input_type == other.input_type)
        match = match and type_match
        table_data.append(["input_type", self.input_type, other.input_type, type_match])
        
        # Handle concrete_inputs separately
        if self.name == "aten::addmm":
            # For addmm, compare alpha/beta separately
            if len(self.concrete_inputs) >= 5 and len(other.concrete_inputs) >= 5:
                # Parse alpha/beta scalars using CPUOp method
                actual_alpha, actual_beta = self.get_alpha_beta()
                target_alpha, target_beta = other.get_alpha_beta()
                
                # Check alpha/beta scalars for addmm (affects epilogue fusion)
                alpha_match = (actual_alpha == target_alpha)
                beta_match = (actual_beta == target_beta)
                match = match and alpha_match and beta_match
                
                table_data.append(["alpha (concrete_inputs[3])", actual_alpha, target_alpha, alpha_match])
                table_data.append(["beta (concrete_inputs[4])", actual_beta, target_beta, beta_match])
            else:
                # Fallback: compare concrete_inputs as-is if not enough elements
                concrete_match = (self.concrete_inputs == other.concrete_inputs)
                match = match and concrete_match
                table_data.append(["concrete_inputs", self.concrete_inputs, other.concrete_inputs, concrete_match])
        else:
            # For non-addmm ops, compare concrete_inputs as-is
            concrete_match = (self.concrete_inputs == other.concrete_inputs)
            match = match and concrete_match
            table_data.append(["concrete_inputs", self.concrete_inputs, other.concrete_inputs, concrete_match])
        
        if show_table:
            outcome = "[green]SUCCESS[/green]" if match else "[red]FAILURE[/red]"
            print_utils.comp_table(f"{title} {outcome}", ["Field", "Actual", "Target", "Match"], table_data)
        
        return match
    
    @classmethod
    def from_dict(cls, cpu_op_dict: Optional[Dict[str, Any]]) -> Optional['CPUOp']:
        """Create CPUOp from dictionary."""
        if not cpu_op_dict:
            return None
        return cls(
            name=cpu_op_dict.get("name", ""),
            input_dims=cpu_op_dict.get("input_dims"),
            input_strides=cpu_op_dict.get("input_strides"),
            input_type=cpu_op_dict.get("input_type"),
            concrete_inputs=cpu_op_dict.get("concrete_inputs"),
            ts=cpu_op_dict.get("ts"),
            dur=cpu_op_dict.get("dur"),
            external_id=cpu_op_dict.get("external_id")
        )


class Sequence:
    """Sequence class containing cpu_op and kernel."""
    def __init__(self, cpu_op: Optional[CPUOp] = None, kernel: Optional[Kernel] = None):
        """Initialize sequence with cpu_op and kernel."""
        self.cpu_op = cpu_op
        self.kernel = kernel
    
    def get_str(self) -> str:
        """Get sequence string representation: "{op_name} -> {kernel_name}"."""
        return f"{self.cpu_op.name} -> {clean_kernel_name(self.kernel.name)}"
    
    @classmethod
    def from_dict(cls, sequence_dict: Dict[str, Any]) -> 'Sequence':
        """Create Sequence from dictionary."""
        cpu_op = CPUOp.from_dict(sequence_dict.get("cpu_op"))
        kernel = Kernel.from_dict(sequence_dict.get("kernel"))
        return cls(cpu_op=cpu_op, kernel=kernel)

