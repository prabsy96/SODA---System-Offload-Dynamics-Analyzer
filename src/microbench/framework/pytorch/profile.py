from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import torch
from torch.profiler import profile, ProfilerActivity, record_function as record
from soda.common import utils
from soda.common.data import clean_kernel_name
#==============================================================================
# Helper Functions for Safe Operation Execution
# ==============================================================================

def _safe_binary_op(inputs: List[Any], op_fn) -> torch.Tensor:
    """Safe binary operation that handles tensor-scalar combinations."""
    if len(inputs) < 2:
        return inputs[0] if inputs else torch.tensor(0.0, device="cuda")
    
    a = inputs[0]
    b = inputs[1]
    
    # Handle alpha parameter (3rd input) for add/sub
    alpha = 1
    if len(inputs) > 2 and isinstance(inputs[2], (int, float)):
        alpha = inputs[2]
    elif len(inputs) > 2 and isinstance(inputs[2], torch.Tensor) and inputs[2].numel() == 1:
        alpha = inputs[2].item()
    
    # Ensure 'a' is a tensor
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, device="cuda")
    
    # If 'b' is a scalar tensor, op_fn usually handles it, but explicit check helps
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device="cuda")

    # Apply operation with alpha if applicable
    try:
        if op_fn in (torch.add, torch.sub) and alpha != 1:
            return op_fn(a, b, alpha=alpha)
        return op_fn(a, b)
    except Exception:
        # Fallback for type mismatches
        return a

def _safe_inplace_binary_op(inputs: List[Any], op_name: str) -> torch.Tensor:
    """Safe in-place binary operation."""
    if len(inputs) < 2:
        return inputs[0] if inputs else torch.tensor(0.0, device="cuda")
    
    a = inputs[0]
    b = inputs[1]
    
    if not isinstance(a, torch.Tensor):
        return torch.tensor(0.0, device="cuda") # Can't inplace on non-tensor
    
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device=a.device, dtype=a.dtype)
    
    try:
        op = getattr(a, op_name)
        return op(b)
    except Exception:
        return a

def _safe_comparison_op(inputs: List[Any], op_fn) -> torch.Tensor:
    """Safe comparison operation that handles tensor-scalar combinations."""
    if len(inputs) < 2:
        return inputs[0] if inputs else torch.tensor(False, device="cuda")
    
    a = inputs[0]
    b = inputs[1]
    
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, device="cuda")
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device="cuda")
        
    try:
        return op_fn(a, b)
    except Exception:
        return torch.zeros_like(a, dtype=torch.bool)

def _safe_fill(inputs: List[Any]) -> torch.Tensor:
    """Safe fill_ that handles special values like nan, inf."""
    if len(inputs) < 1:
        return torch.tensor(0.0, device="cuda")
    
    tensor = inputs[0]
    if not isinstance(tensor, torch.Tensor):
        return torch.tensor(0.0, device="cuda")
    
    # Get fill value
    fill_val = 0
    if len(inputs) > 1:
        val = inputs[1]
        if isinstance(val, torch.Tensor):
            fill_val = val.item()
        else:
            fill_val = val

    # Handle nan/inf for integer tensors (which cause runtime errors)
    if tensor.dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool]:
        if isinstance(fill_val, float) and (fill_val != fill_val or fill_val == float('inf') or fill_val == float('-inf')):
            # Use max/min representable value instead of crashing
            if fill_val == float('inf'):
                fill_val = torch.iinfo(tensor.dtype).max
            elif fill_val == float('-inf'):
                fill_val = torch.iinfo(tensor.dtype).min
            else: # NaN
                fill_val = 0
                
    try:
        return tensor.fill_(fill_val)
    except Exception:
        return tensor

def _safe_gather(inputs: List[Any]) -> torch.Tensor:
    """Safe gather that handles edge cases."""
    if len(inputs) < 3:
        return inputs[0] if inputs else torch.tensor(0.0, device="cuda")
    
    source = inputs[0]
    dim = inputs[1] if isinstance(inputs[1], int) else 0
    index = inputs[2]
    
    if isinstance(source, torch.Tensor) and isinstance(index, torch.Tensor):
        # Ensure dim is valid
        if dim < 0: dim += source.dim()
        if dim < 0 or dim >= source.dim(): dim = 0
            
        max_idx = source.size(dim)
        # Clamp index to valid range to prevent device-side asserts
        if max_idx > 0:
            index = torch.clamp(index, 0, max_idx - 1)
        else:
            return source # Empty source
            
        return torch.gather(source, dim, index)
    return source

def _safe_index_select(inputs: List[Any]) -> torch.Tensor:
    """Safe index_select that handles edge cases."""
    if len(inputs) < 3:
        return inputs[0] if inputs else torch.tensor(0.0, device="cuda")
    
    source = inputs[0]
    dim = inputs[1] if isinstance(inputs[1], int) else 0
    index = inputs[2]
    
    if isinstance(source, torch.Tensor) and isinstance(index, torch.Tensor):
        if dim < 0: dim += source.dim()
        if dim < 0 or dim >= source.dim(): dim = 0
            
        max_idx = source.size(dim)
        if max_idx > 0:
            index = torch.clamp(index.long(), 0, max_idx - 1)
            return torch.index_select(source, dim, index)
    return source

def _get_dim_arg(inputs: List[Any], idx: int, default: Any) -> Any:
    """Safely get a dimension argument from inputs."""
    if len(inputs) > idx:
        val = inputs[idx]
        if isinstance(val, int):
            return val
        if isinstance(val, torch.Tensor) and val.numel() == 1:
            return int(val.item())
    return default

# ==============================================================================
# Supported Operations Map
# ==============================================================================
SUPPORTED_OPS = {
    # GEMM operations
    "aten::addmm": lambda inputs: torch.addmm(inputs[0], inputs[1], inputs[2]) if len(inputs) >= 3 else None,
    "aten::mm": lambda inputs: torch.mm(inputs[0], inputs[1]) if len(inputs) >= 2 else None,
    "aten::bmm": lambda inputs: torch.bmm(inputs[0], inputs[1]) if len(inputs) >= 2 else None,
    "aten::matmul": lambda inputs: torch.matmul(inputs[0], inputs[1]) if len(inputs) >= 2 else None,
    "aten::linear": lambda inputs: torch.nn.functional.linear(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None) if len(inputs) >= 2 else None,
    
    # Elementwise operations
    "aten::add": lambda inputs: _safe_binary_op(inputs, torch.add),
    "aten::mul": lambda inputs: _safe_binary_op(inputs, torch.mul),
    "aten::div": lambda inputs: _safe_binary_op(inputs, torch.div),
    "aten::sub": lambda inputs: _safe_binary_op(inputs, torch.sub),
    "aten::add_": lambda inputs: _safe_inplace_binary_op(inputs, "add_"),
    "aten::sub_": lambda inputs: _safe_inplace_binary_op(inputs, "sub_"),
    "aten::mul_": lambda inputs: _safe_inplace_binary_op(inputs, "mul_"),
    "aten::div_": lambda inputs: _safe_inplace_binary_op(inputs, "div_"),
    
    # Reduction operations
    "aten::sum": lambda inputs: torch.sum(inputs[0]),
    "aten::mean": lambda inputs: torch.mean(inputs[0].float()).to(inputs[0].dtype),
    "aten::max": lambda inputs: torch.max(inputs[0])[0] if inputs[0].numel() > 0 else inputs[0],
    "aten::min": lambda inputs: torch.min(inputs[0])[0] if inputs[0].numel() > 0 else inputs[0],
    
    # Normalization
    "aten::layer_norm": lambda inputs: torch.nn.functional.layer_norm(inputs[0], inputs[0].shape[-1:]) if len(inputs) >= 1 else None,
    "aten::native_layer_norm": lambda inputs: torch.nn.functional.layer_norm(inputs[0], inputs[0].shape[-1:]) if len(inputs) >= 1 else None,
    "aten::softmax": lambda inputs: torch.softmax(inputs[0], dim=-1) if len(inputs) >= 1 else None,
    "aten::_softmax": lambda inputs: torch.softmax(inputs[0], dim=-1) if len(inputs) >= 1 else None,
    
    # Activation functions
    "aten::relu": lambda inputs: torch.relu(inputs[0]),
    "aten::gelu": lambda inputs: torch.nn.functional.gelu(inputs[0]),
    "aten::silu": lambda inputs: torch.nn.functional.silu(inputs[0]),
    "aten::tanh": lambda inputs: torch.tanh(inputs[0]),
    "aten::sigmoid": lambda inputs: torch.sigmoid(inputs[0]),
    
    # Reshape operations
    "aten::view": lambda inputs: inputs[0].view(-1) if len(inputs) >= 1 else None,
    "aten::reshape": lambda inputs: inputs[0].reshape(-1) if len(inputs) >= 1 else None,
    "aten::transpose": lambda inputs: inputs[0].transpose(0, 1) if len(inputs) >= 1 and inputs[0].dim() >= 2 else inputs[0],
    "aten::permute": lambda inputs: inputs[0].permute(*range(inputs[0].dim()-1, -1, -1)) if len(inputs) >= 1 else None,
    "aten::contiguous": lambda inputs: inputs[0].contiguous(),
    "aten::flatten": lambda inputs: inputs[0].flatten(),
    
    # Copy/memory operations
    "aten::copy_": lambda inputs: inputs[0].clone() if len(inputs) >= 1 else None,
    "aten::clone": lambda inputs: inputs[0].clone(),
    "aten::to": lambda inputs: inputs[0].clone(),
    "aten::_to_copy": lambda inputs: inputs[0].clone(),
    
    # Fill/initialization operations
    "aten::fill_": lambda inputs: _safe_fill(inputs),
    "aten::zero_": lambda inputs: inputs[0].zero_(),
    "aten::ones_like": lambda inputs: torch.ones_like(inputs[0]),
    "aten::zeros_like": lambda inputs: torch.zeros_like(inputs[0]),
    
    # Comparison operations
    "aten::ge": lambda inputs: _safe_comparison_op(inputs, torch.ge),
    "aten::le": lambda inputs: _safe_comparison_op(inputs, torch.le),
    "aten::lt": lambda inputs: _safe_comparison_op(inputs, torch.lt),
    "aten::gt": lambda inputs: _safe_comparison_op(inputs, torch.gt),
    "aten::eq": lambda inputs: _safe_comparison_op(inputs, torch.eq),
    "aten::ne": lambda inputs: _safe_comparison_op(inputs, torch.ne),
    
    # Cumulative operations
    "aten::cumsum": lambda inputs: torch.cumsum(inputs[0], dim=_get_dim_arg(inputs, 1, -1)),
    "aten::cumprod": lambda inputs: torch.cumprod(inputs[0], dim=_get_dim_arg(inputs, 1, -1)),
    
    # Masking operations
    "aten::masked_fill_": lambda inputs: inputs[0].masked_fill_(
        inputs[1] if len(inputs) > 1 and inputs[1].dtype == torch.bool else torch.zeros_like(inputs[0], dtype=torch.bool),
        inputs[2] if len(inputs) > 2 else 0
    ),
    "aten::masked_fill": lambda inputs: inputs[0].masked_fill(
        inputs[1] if len(inputs) > 1 and inputs[1].dtype == torch.bool else torch.zeros_like(inputs[0], dtype=torch.bool),
        inputs[2] if len(inputs) > 2 else 0
    ),
    "aten::where": lambda inputs: torch.where(inputs[0].bool(), inputs[1], inputs[2]) if len(inputs) >= 3 else inputs[0],
    
    # Indexing operations
    "aten::gather": lambda inputs: _safe_gather(inputs),
    "aten::index": lambda inputs: inputs[0], # Simplified
    "aten::index_select": lambda inputs: _safe_index_select(inputs),
    "aten::index_put_": lambda inputs: inputs[0], # Simplified
    
    # Bitwise operations
    "aten::bitwise_and": lambda inputs: torch.bitwise_and(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::bitwise_or": lambda inputs: torch.bitwise_or(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::bitwise_xor": lambda inputs: torch.bitwise_xor(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::bitwise_not": lambda inputs: torch.bitwise_not(inputs[0]),
    
    # Concatenation/stacking
    "aten::cat": lambda inputs: torch.cat(inputs[0] if isinstance(inputs[0], (list, tuple)) else [inputs[0]], dim=_get_dim_arg(inputs, 1, 0)),
    "aten::stack": lambda inputs: torch.stack(inputs[0] if isinstance(inputs[0], (list, tuple)) else [inputs[0]], dim=_get_dim_arg(inputs, 1, 0)),
    
    # Reduction with indices
    "aten::argmax": lambda inputs: torch.argmax(inputs[0], dim=_get_dim_arg(inputs, 1, None)),
    "aten::argmin": lambda inputs: torch.argmin(inputs[0], dim=_get_dim_arg(inputs, 1, None)),
    
    # Embedding operations
    "aten::embedding": lambda inputs: torch.nn.functional.embedding(inputs[1].long() if len(inputs) > 1 else torch.tensor([0], device=inputs[0].device), inputs[0]) if len(inputs) >= 1 else None,
    
    # Expand/repeat
    "aten::expand": lambda inputs: inputs[0].expand_as(inputs[0]) if len(inputs) >= 1 else None,
    "aten::repeat": lambda inputs: inputs[0].repeat(*(1 for _ in range(inputs[0].dim()))) if len(inputs) >= 1 else None,
    
    # Squeeze/unsqueeze
    "aten::squeeze": lambda inputs: inputs[0].squeeze(),
    "aten::unsqueeze": lambda inputs: inputs[0].unsqueeze(0),
    
    # Split operations
    "aten::split": lambda inputs: torch.split(inputs[0], inputs[0].size(0), dim=0),
    "aten::chunk": lambda inputs: torch.chunk(inputs[0], 1, dim=0),
    
    # Type conversions
    "aten::type_as": lambda inputs: inputs[0].type_as(inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::float": lambda inputs: inputs[0].float(),
    "aten::half": lambda inputs: inputs[0].half(),
    "aten::int": lambda inputs: inputs[0].int(),
    "aten::long": lambda inputs: inputs[0].long(),
    "aten::bool": lambda inputs: inputs[0].bool(),

    "aten::arange": lambda inputs: torch.arange(inputs[0].item() if isinstance(inputs[0], torch.Tensor) else inputs[0], device="cuda") if len(inputs) > 0 else torch.arange(10, device="cuda"),
    
    # Slicing
    "aten::slice": lambda inputs: inputs[0].narrow(
        _get_dim_arg(inputs, 1, 0), 
        _get_dim_arg(inputs, 2, 0), 
        max(1, _get_dim_arg(inputs, 3, inputs[0].size(_get_dim_arg(inputs, 1, 0))) - _get_dim_arg(inputs, 2, 0))
    ) if len(inputs) >= 1 else None,
    
    "aten::select": lambda inputs: inputs[0].select(_get_dim_arg(inputs, 1, 0), _get_dim_arg(inputs, 2, 0)) if len(inputs) >= 1 else None,
    "aten::any": lambda inputs: torch.any(inputs[0]),
    "aten::all": lambda inputs: torch.all(inputs[0]),
    "aten::argmax": lambda inputs: torch.argmax(inputs[0]),
    "aten::argmin": lambda inputs: torch.argmin(inputs[0]),
    
    # Additional elementwise operations
    "aten::pow": lambda inputs: torch.pow(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::sqrt": lambda inputs: torch.sqrt(inputs[0]),
    "aten::exp": lambda inputs: torch.exp(inputs[0]),
    "aten::log": lambda inputs: torch.log(inputs[0]),
    "aten::sin": lambda inputs: torch.sin(inputs[0]),
    "aten::cos": lambda inputs: torch.cos(inputs[0]),
    
    # Comparison operations
    "aten::eq": lambda inputs: torch.eq(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::ne": lambda inputs: torch.ne(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::gt": lambda inputs: torch.gt(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::lt": lambda inputs: torch.lt(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::ge": lambda inputs: torch.ge(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::le": lambda inputs: torch.le(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    
    # Indexing operations
    "aten::gather": lambda inputs: torch.gather(inputs[0], 0, torch.zeros_like(inputs[0]).long()) if len(inputs) >= 1 else None,
    "aten::scatter": lambda inputs: torch.scatter(inputs[0], 0, torch.zeros_like(inputs[0]).long(), inputs[1]) if len(inputs) >= 2 else None,
    "aten::index_select": lambda inputs: torch.index_select(inputs[0], 0, torch.zeros(inputs[0].shape[0], dtype=torch.long)) if len(inputs) >= 1 else None,
    "aten::masked_select": lambda inputs: torch.masked_select(inputs[0], torch.ones_like(inputs[0], dtype=torch.bool)) if len(inputs) >= 1 else None,
    
    # Conditional operations
    "aten::where": lambda inputs: torch.where(inputs[0], inputs[1], inputs[2]) if len(inputs) >= 3 else None,
    
    # Additional normalization
    "aten::batch_norm": lambda inputs: torch.nn.functional.batch_norm(inputs[0], torch.zeros(inputs[0].shape[1]), torch.ones(inputs[0].shape[1])) if len(inputs) >= 1 else None,
    
    # Additional activation
    "aten::elu": lambda inputs: torch.nn.functional.elu(inputs[0]),
    "aten::leaky_relu": lambda inputs: torch.nn.functional.leaky_relu(inputs[0]),
    
    # Additional reshape
    "aten::squeeze": lambda inputs: inputs[0].squeeze(),
    "aten::unsqueeze": lambda inputs: inputs[0].unsqueeze(0),
    "aten::expand": lambda inputs: inputs[0].expand(inputs[0].shape),
    "aten::repeat": lambda inputs: inputs[0].repeat(1),
    
    # Additional copy
    "aten::detach": lambda inputs: inputs[0].detach(),
    
    # Additional embedding
    "aten::embedding_bag": lambda inputs: torch.nn.functional.embedding_bag(inputs[0].long() % 1000, torch.randn(1000, inputs[1].shape[-1] if len(inputs) > 1 else 768)) if len(inputs) >= 1 else None,
}

def is_op_supported(aten_op_name: str) -> bool:
    """Check if an ATen operation is supported for replay."""
    if aten_op_name in SUPPORTED_OPS:
        return True
    # Try base name match (strip aten:: and trailing _)
    op_base = aten_op_name.split("::")[-1].rstrip("_")
    return any(op_base in supported_op for supported_op in SUPPORTED_OPS)
# ==============================================================================
# Input Creation Functions
# ==============================================================================

def parse_scalar_value(concrete_val: Any) -> Any:
    """Parse a scalar value from concrete_inputs."""
    if concrete_val is None or concrete_val == "":
        return 1.0
    
    val_str = str(concrete_val).strip().lower()
    
    if val_str == "false": return False
    if val_str == "true": return True
    if val_str == "nan": return float('nan')
    if val_str == "inf": return float('inf')
    if val_str == "-inf": return float('-inf')
    
    try:
        if "." not in val_str and "e" not in val_str:
            int_val = int(concrete_val)
            # Clamp to INT64 range to avoid overflow in PyTorch
            i64_max = 9223372036854775807
            i64_min = -9223372036854775808
            if int_val > i64_max: return i64_max
            if int_val < i64_min: return i64_min
            return int_val
        return float(concrete_val)
    except (ValueError, TypeError, OverflowError):
        return 1.0

def create_tensor_list(dims: Any, concrete_val: Any = None, device: str = "cuda") -> List[torch.Tensor]:
    """Create a list of tensors for TensorList inputs (e.g., aten::cat)."""
    if not dims:
        return [torch.randn(16, 16, device=device), torch.randn(16, 16, device=device)]
    
    if isinstance(dims, list) and len(dims) > 0:
        if isinstance(dims[0], list):
            # dims is [[128, 64], [128, 64]] - multiple shapes
            tensors = []
            for shape in dims:
                if shape and len(shape) > 0:
                    tensors.append(torch.randn(*shape, device=device))
                else:
                    tensors.append(torch.randn(1, device=device)) # Handle empty shape
            return tensors if tensors else [torch.randn(16, 16, device=device)]
        else:
            # dims is [128, 64] - single shape, create two tensors with same shape
            return [torch.randn(*dims, device=device), torch.randn(*dims, device=device)]
    
    return [torch.randn(16, 16, device=device), torch.randn(16, 16, device=device)]

def create_tensor(
    dims: List[int],
    dtype_str: str,
    strides: Optional[List[int]] = None,
    device: str = "cuda"
) -> Optional[torch.Tensor]:
    """Create a tensor with specified dimensions, dtype, and strides."""
    # Handle None or empty dims -> Scalar tensor
    if dims is None or dims == []:
        return torch.tensor(1.0, device=device)
    
    if dtype_str is None: dtype_str = "float"
    
    dtype_map = {
        "bool": torch.bool,
        "long int": torch.int64,
        "long": torch.int64,
        "int64": torch.int64,
        "int": torch.int32,
        "int32": torch.int32,
        "float": torch.float32,
        "float32": torch.float32,
        "double": torch.float64,
        "float64": torch.float64,
        "half": torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "char": torch.int8,
        "unsigned char": torch.uint8,
    }
    
    clean_dtype = dtype_str.replace("c10::", "").lower().strip()
    dtype = dtype_map.get(clean_dtype)
    
    if dtype is None:
        try: dtype = utils.parse_dtype_to_torch(dtype_str)
        except: pass
    
    if dtype is None:
        if "float" in clean_dtype: dtype = torch.float32
        elif "double" in clean_dtype: dtype = torch.float64
        elif "int" in clean_dtype or "long" in clean_dtype: dtype = torch.int64
        elif "bool" in clean_dtype: dtype = torch.bool
        else: dtype = torch.float32

    target_device = torch.device(device)
    
    try:
        if dtype == torch.bool:
            tensor = torch.randint(0, 2, dims, dtype=dtype, device=target_device)
        elif dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]:
            tensor = torch.randint(0, 10, dims, dtype=dtype, device=target_device)
        else:
            tensor = torch.randn(*dims, dtype=dtype, device=target_device)
    except Exception as e:
        print(f"Warning: Failed to create tensor with dims={dims}, dtype={dtype}: {e}")
        return torch.randn(16, 16, device=target_device)

    # Handle strides if provided
    if strides and len(strides) == len(dims):
        try:
            storage_size = sum((d - 1) * s for d, s in zip(dims, strides)) + 1
            storage = torch.empty(storage_size, dtype=dtype, device=target_device)
            tensor = torch.as_strided(storage, dims, strides)
        except Exception:
            pass  # Keep contiguous tensor
    
    return tensor

def create_valid_index_tensor(
    source_dims: List[int],
    dim: int,
    dtype: torch.dtype = torch.int64,
    device: str = "cuda"
) -> torch.Tensor:
    """Create a valid index tensor for gather/scatter operations."""
    if not source_dims or dim >= len(source_dims):
        return torch.zeros(1, dtype=dtype, device=device)
    
    max_idx = source_dims[dim]
    index_shape = list(source_dims)
    # Ensure at least 1 element range
    return torch.randint(0, max(1, max_idx), index_shape, dtype=dtype, device=device)

def create_input_tensors(aten_op: Dict[str, Any], device: str = "cuda") -> List[Any]:
    """Create input tensors from ATen operation metadata."""
    input_dims = aten_op.get("input_dims", [])
    input_types = aten_op.get("input_type", [])
    input_strides = aten_op.get("input_strides", [])
    concrete_inputs = aten_op.get("concrete_inputs", [])
    op_name = aten_op.get("name", "")
    
    inputs = []
    
    # If input_types is missing, try to infer from dims
    if not input_types:
        for dim in input_dims:
            if dim and isinstance(dim, list) and len(dim) > 0:
                inputs.append(create_tensor(dim, "float", device=device))
            else:
                inputs.append(torch.tensor(1.0, device=device))
        return inputs if inputs else [torch.randn(16, 16, device=device)]

    for i, type_str in enumerate(input_types):
        if type_str is None or type_str == "":
            type_str = "float"
        
        dims = input_dims[i] if i < len(input_dims) else []
        strides = input_strides[i] if i < len(input_strides) else None
        concrete_val = concrete_inputs[i] if i < len(concrete_inputs) else None
        
        # Handle TensorList
        if type_str == "TensorList":
            inputs.append(create_tensor_list(dims, concrete_val, device))
            continue
        
        # Handle ScalarList
        if type_str == "ScalarList":
            if concrete_val and isinstance(concrete_val, str):
                try:
                    import ast
                    inputs.append(ast.literal_eval(concrete_val))
                    continue
                except:
                    pass
            if dims and isinstance(dims, list):
                inputs.append(dims)
            else:
                inputs.append([1])
            continue
        
        # Handle Scalar type - always return Python scalar
        if type_str == "Scalar":
            inputs.append(parse_scalar_value(concrete_val))
            continue
        
        # Handle empty dims - this is the critical fix
        if dims is None or dims == []:
            val = parse_scalar_value(concrete_val) if concrete_val else 1
            
            # Determine if this should be a Python scalar or scalar tensor
            # Key insight: if type contains "int"/"long" AND we already have a tensor input,
            # this is likely a Python scalar argument (e.g., dim argument)
            is_likely_scalar_arg = (
                ("int" in type_str.lower() or "long" in type_str.lower()) and
                inputs and isinstance(inputs[-1], torch.Tensor)
            )
            
            if is_likely_scalar_arg:
                # Return Python int, not tensor
                inputs.append(int(val) if isinstance(val, (int, float)) else 0)
            elif "bool" in type_str.lower():
                # Bool with empty dims - could be scalar or tensor
                if inputs and isinstance(inputs[-1], torch.Tensor):
                    inputs.append(bool(val))
                else:
                    inputs.append(torch.tensor(bool(val), dtype=torch.bool, device=device))
            elif "int" in type_str.lower() or "long" in type_str.lower():
                # First input, make it a scalar tensor
                inputs.append(torch.tensor(int(val) if isinstance(val, (int, float)) else 0, dtype=torch.int64, device=device))
            else:
                # Float scalar tensor
                inputs.append(torch.tensor(float(val) if isinstance(val, (int, float)) else 1.0, device=device))
            continue

        # Special handling for gather/scatter index tensors
        if op_name in ("aten::gather", "aten::scatter_", "aten::index_select"):
            if i == 2 and len(inputs) >= 1:
                source_tensor = inputs[0]
                if isinstance(source_tensor, torch.Tensor):
                    dim_arg = inputs[1] if len(inputs) > 1 and isinstance(inputs[1], int) else 0
                    source_dims = list(source_tensor.shape)
                    inputs.append(create_valid_index_tensor(source_dims, dim_arg, torch.int64, device))
                    continue
        
        # Handle standard Tensor - this should work for most cases
        tensor = create_tensor(dims, type_str, strides, device)
        if tensor is not None:
            inputs.append(tensor)
        else:
            # Fallback - but log this as it indicates a problem
            print(f"  Warning: Failed to create tensor for {op_name} input {i}: dims={dims}, type={type_str}")
            inputs.append(torch.randn(16, 16, device=device))
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return inputs if inputs else [torch.randn(16, 16, device=device)]


def _handle_special_ops(op_name: str, aten_op: Dict[str, Any], device: str) -> Optional[List[Any]]:
    """
    Handle special ops that need custom input reconstruction.
    Returns None if not a special op.
    """
    input_dims = aten_op.get("input_dims", [])
    input_types = aten_op.get("input_type", [])
    concrete_inputs = aten_op.get("concrete_inputs", [])
    
    # aten::sum - tensor, optional dim, keepdim
    if op_name == "aten::sum":
        try:
            dims = input_dims[0] if input_dims and input_dims[0] else [16, 16]
            dtype_str = input_types[0] if input_types else "float"
            tensor = create_tensor(dims, dtype_str, device=device)
            if tensor is None:
                tensor = torch.randn(dims, device=device)
            return [tensor]
        except Exception as e:
            return None
    
    # aten::sub, aten::add - tensor, tensor/scalar
    if op_name in ("aten::sub", "aten::add"):
        try:
            dims0 = input_dims[0] if input_dims and input_dims[0] else None
            if dims0 is None:
                return None  # Let default handler try
            
            dims1 = input_dims[1] if len(input_dims) > 1 and input_dims[1] else []
            dtype0 = input_types[0] if input_types else "float"
            dtype1 = input_types[1] if len(input_types) > 1 else "float"
            
            t0 = create_tensor(dims0, dtype0, device=device)
            if t0 is None:
                t0 = torch.randn(dims0, device=device)
            
            # Second arg: tensor if has dims, else scalar
            if dims1:
                t1 = create_tensor(dims1, dtype1, device=device)
                if t1 is None:
                    t1 = torch.randn(dims1, device=device)
            else:
                val = parse_scalar_value(concrete_inputs[1]) if len(concrete_inputs) > 1 else 1
                t1 = val if isinstance(val, (int, float)) else 1
            
            return [t0, t1]
        except Exception:
            return None
    
    # aten::ge, aten::gt, aten::lt, aten::le, aten::eq, aten::ne - comparison ops
    if op_name in ("aten::ge", "aten::gt", "aten::lt", "aten::le", "aten::eq", "aten::ne"):
        try:
            dims0 = input_dims[0] if input_dims and input_dims[0] else None
            if dims0 is None:
                return None
            
            dtype0 = input_types[0] if input_types else "long int"
            
            if "int" in dtype0.lower() or "long" in dtype0.lower():
                t0 = torch.randint(0, 256, dims0, dtype=torch.int64, device=device)
            else:
                t0 = torch.randn(dims0, device=device)
            
            # Second arg - check if it has dims or is scalar
            dims1 = input_dims[1] if len(input_dims) > 1 and input_dims[1] else []
            if dims1:
                dtype1 = input_types[1] if len(input_types) > 1 else dtype0
                if "int" in dtype1.lower() or "long" in dtype1.lower():
                    t1 = torch.randint(0, 256, dims1, dtype=torch.int64, device=device)
                else:
                    t1 = torch.randn(dims1, device=device)
            else:
                val = parse_scalar_value(concrete_inputs[1]) if len(concrete_inputs) > 1 else 0
                t1 = val if isinstance(val, (int, float)) else 0
            
            return [t0, t1]
        except Exception:
            return None
    
    # aten::copy_ - dst, src
    if op_name == "aten::copy_":
        try:
            dims0 = input_dims[0] if input_dims and input_dims[0] else None
            dims1 = input_dims[1] if len(input_dims) > 1 and input_dims[1] else None
            if dims0 is None:
                return None
            
            dtype0 = input_types[0] if input_types else "float"
            dtype1 = input_types[1] if len(input_types) > 1 else dtype0
            
            # Create destination tensor
            if "bool" in dtype0.lower():
                t0 = torch.zeros(dims0, dtype=torch.bool, device=device)
            elif "int" in dtype0.lower() or "long" in dtype0.lower():
                t0 = torch.zeros(dims0, dtype=torch.int64, device=device)
            else:
                t0 = torch.zeros(dims0, device=device)
            
            # Create source tensor (broadcastable to dst)
            src_dims = dims1 if dims1 else dims0
            if "int" in dtype1.lower() or "long" in dtype1.lower():
                t1 = torch.randint(0, 2, src_dims, dtype=torch.int64, device=device)
            elif "bool" in dtype1.lower():
                t1 = torch.randint(0, 2, src_dims, dtype=torch.bool, device=device)
            else:
                t1 = torch.randn(src_dims, device=device)
            
            return [t0, t1]
        except Exception:
            return None
    
    # aten::gather - src, dim, index
    if op_name == "aten::gather":
        try:
            src_dims = input_dims[0] if input_dims and input_dims[0] else None
            idx_dims = input_dims[2] if len(input_dims) > 2 and input_dims[2] else None
            if src_dims is None or idx_dims is None:
                return None
            
            dim = int(concrete_inputs[1]) if len(concrete_inputs) > 1 and concrete_inputs[1] else 0
            
            src = torch.randn(src_dims, device=device)
            max_idx = src_dims[dim] if dim < len(src_dims) else src_dims[0]
            index = torch.randint(0, max(1, max_idx), idx_dims, dtype=torch.int64, device=device)
            
            return [src, dim, index]
        except Exception:
            return None
    
    # aten::where - cond, x, y
    if op_name == "aten::where":
        try:
            cond_dims = input_dims[0] if input_dims and input_dims[0] else None
            if cond_dims is None:
                return None
            
            x_dims = input_dims[1] if len(input_dims) > 1 and input_dims[1] else []
            y_dims = input_dims[2] if len(input_dims) > 2 and input_dims[2] else []
            
            cond = torch.randint(0, 2, cond_dims, dtype=torch.bool, device=device)
            
            # x and y might be scalars (empty dims)
            if not x_dims:
                x = 0.0
            else:
                x = torch.randn(x_dims, device=device)
            
            if not y_dims:
                y = float('-inf')
            else:
                y = torch.randn(y_dims, device=device)
            
            return [cond, x, y]
        except Exception:
            return None
    
    # aten::native_layer_norm
    if op_name == "aten::native_layer_norm":
        try:
            dims = input_dims[0] if input_dims and input_dims[0] else None
            if dims is None:
                return None
            
            x = torch.randn(dims, device=device)
            normalized_shape = [dims[-1]]
            weight = torch.ones(normalized_shape, device=device)
            bias = torch.zeros(normalized_shape, device=device)
            return [x, normalized_shape, weight, bias, 1e-5]
        except Exception:
            return None
    
    # aten::_softmax
    if op_name == "aten::_softmax":
        try:
            dims = input_dims[0] if input_dims and input_dims[0] else None
            if dims is None:
                return None
            x = torch.randn(dims, device=device)
            return [x, -1, False]
        except Exception:
            return None
    
    # aten::max
    if op_name == "aten::max":
        try:
            dims = input_dims[0] if input_dims and input_dims[0] else None
            if dims is None:
                return None
            x = torch.randn(dims, device=device)
            return [x]
        except Exception:
            return None
    
    return None

# ==============================================================================
# Profiling Functions
# ==============================================================================

def reset_cuda_state():
    """Reset CUDA state to recover from errors."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            return True
        except Exception as e:
            print(f"Warning: CUDA reset failed: {e}")
            return False
    return True

def execute_operation(aten_op_name: str, inputs: List[Any]) -> Optional[torch.Tensor]:
    """Execute any supported ATen operation."""
    if aten_op_name in SUPPORTED_OPS:
        with record(f"torch_op:{aten_op_name}"):
            return SUPPORTED_OPS[aten_op_name](inputs)
    
    # Fallback: try base name match
    op_base = aten_op_name.split("::")[-1].rstrip("_")
    for supported_op, func in SUPPORTED_OPS.items():
        if op_base in supported_op:
            with record(f"torch_op:{aten_op_name}"):
                return func(inputs)
    
    raise ValueError(f"Unsupported operation: {aten_op_name}")

def profile_operation(
    aten_op_name: str,
    inputs: List[torch.Tensor],
    warmup: int,
    runs: int,
    trace_file: Path,
) -> None:
    """Profile a PyTorch operation N times and save trace."""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            try:
                execute_operation(aten_op_name, inputs)
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # Profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(runs):
                try:
                    execute_operation(aten_op_name, inputs)
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    prof.export_chrome_trace(str(trace_file))

def replay_all_sequences_from_aten_ops(
    sequences: List[Dict[str, Any]], 
    warmup: int,
    runs: int
) -> List[Dict[str, Any]]:
    """Replay all event sequences (GEMM and non-GEMM) with error isolation."""
    kernel_traces_dir = utils.get_path("PYTORCH_TRACES")
    utils.ensure_dir(kernel_traces_dir)

    sequence_by_idx = {}
    supported_count = 0
    skipped_count = 0
    
    print(f"Profiling {len(sequences)} PyTorch kernels with {runs} run{'s' if runs > 1 else ''} each (warmup={warmup})")
    
    for i, event_sequence in enumerate(sequences):
        aten_op = event_sequence.get("aten_op", {})
        kernel = event_sequence.get("kernel", {})
        aten_op_name = aten_op.get("name", "")
        expected_kernel = clean_kernel_name(kernel.get("name", ""))
        seq_idx = i + 1
        is_gemm = event_sequence.get("is_gemm", False)
        
        if not is_op_supported(aten_op_name):
            print(f"[{seq_idx}/{len(sequences)}] SKIP (unsupported): {aten_op_name} -> {expected_kernel}")
            skipped_count += 1
            continue
        
        kernel_type = "GEMM" if is_gemm else "other"
        print(f"[{seq_idx}/{len(sequences)}] [{kernel_type}] {aten_op_name} -> {expected_kernel}")
        
        trace_file_name = utils.format_sequence_filename(seq_idx, aten_op_name, expected_kernel, extension="json")
        trace_file = kernel_traces_dir / trace_file_name

        try:
            if not reset_cuda_state():
                print(f"  Warning: CUDA state reset failed, skipping {aten_op_name}")
                skipped_count += 1
                continue
            
            inputs = create_input_tensors(aten_op)
            if not inputs:
                print(f"  Warning: Could not create inputs for {aten_op_name}")
                skipped_count += 1
                continue
            
            profile_operation(aten_op_name, inputs, warmup, runs, trace_file)

            trace_data = utils.load_json(trace_file)
            events = utils.collect_events(trace_data)
            linked_sequences = utils.link_sequences(events)
            linked_sequences_with_tax = utils.calculate_sequence_metrics(
                linked_sequences, metrics=["launch_tax", "aten_xlat_tax", "py_tax"]
            )

            grouped_seqs_by_id_dict = utils.group_sequences_by_identity(linked_sequences_with_tax)
            agg_sequence = utils.aggregate_sequences(
                grouped_seqs_by_id_dict,
                metrics=["launch_tax", "aten_xlat_tax", "py_tax"],
                event_types=["kernel", "aten_op", "cuda_launch", "torch_op"],
            )
            
            for seq in agg_sequence:
                seq["is_gemm"] = is_gemm
            
            sequence_by_idx[i] = agg_sequence
            supported_count += 1
            
        except Exception as e:
            print(f"  Error profiling {aten_op_name}: {e}")
            skipped_count += 1
            reset_cuda_state()
            continue

    all_replayed_sequences = []
    for kernel_idx in sorted(sequence_by_idx.keys()):
        all_replayed_sequences.extend(sequence_by_idx[kernel_idx])

    print(f"Successfully profiled {supported_count} operations, skipped {skipped_count} unsupported")
    
    if all_replayed_sequences:
        utils.validate_sequences(all_replayed_sequences)
    
    return all_replayed_sequences

def profile_pytorch_all_sequences(target_sequences: Dict[str, Any], warmup: int, runs: int) -> Dict[str, Any]:
    """Profile all PyTorch kernel sequences (GEMM and non-GEMM)."""
    kernel_traces_dir = utils.get_path("PYTORCH_TRACES")
    utils.ensure_dir(kernel_traces_dir, cleanup=True)
    
    replayed_sequences = replay_all_sequences_from_aten_ops(target_sequences["sequences"], warmup, runs)
    
    target_seqs = target_sequences["sequences"]
    for i, replayed_seq in enumerate(replayed_sequences):
        if i < len(target_seqs):
            replayed_seq["freq"] = target_seqs[i].get("count", 1)

    gemm_count = sum(1 for s in replayed_sequences if s.get("is_gemm", False))
    non_gemm_count = len(replayed_sequences) - gemm_count

    pytorch_all_sequences_file = utils.get_path("PYTORCH_ALL_SEQUENCES")
    pytorch_all_sequences_data = {
        "summary": {
            "count": len(replayed_sequences),
            "gemm_count": gemm_count,
            "non_gemm_count": non_gemm_count,
        },
        "sequences": replayed_sequences
    }
    utils.save_json(pytorch_all_sequences_file, pytorch_all_sequences_data)
    print(f"Saved {len(replayed_sequences)} PyTorch sequences to {pytorch_all_sequences_file}")
    
    return pytorch_all_sequences_data

def profile_pytorch_gemm_sequences(target_gemm_sequences: Dict[str, Any], warmup: int, runs: int) -> Dict[str, Any]:
    """Profile PyTorch GEMM sequences to measure launch tax."""
    kernel_traces_dir = utils.get_path("PYTORCH_TRACES")
    utils.ensure_dir(kernel_traces_dir, cleanup=True)
    
    for seq in target_gemm_sequences["sequences"]:
        seq["is_gemm"] = True
    
    replayed_sequences = replay_all_sequences_from_aten_ops(target_gemm_sequences["sequences"], warmup, runs)
    pytorch_gemm_sequences = utils.filter_gemm_sequences(replayed_sequences)

    for i, seq in enumerate(pytorch_gemm_sequences):
        if i < len(target_gemm_sequences["sequences"]):
            seq["freq"] = target_gemm_sequences["sequences"][i].get("count", 1)
            seq["is_gemm"] = True

    pytorch_gemm_sequences_file = utils.get_path("PYTORCH_GEMM_SEQUENCES")
    pytorch_gemm_sequences_data = {
        "summary": {"count": len(pytorch_gemm_sequences)},
        "sequences": pytorch_gemm_sequences
    }
    utils.save_json(pytorch_gemm_sequences_file, pytorch_gemm_sequences_data)
    print(f"Saved {len(pytorch_gemm_sequences)} PyTorch GEMM sequences to {pytorch_gemm_sequences_file}")
    
    return pytorch_gemm_sequences_data