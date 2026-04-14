"""Gate hook instrumentation for capturing per-token expert routing decisions.

Installs forward hooks on MoE gate modules to record which experts are
selected for each token at each layer during inference.  Works with
HuggingFace MoE models (Qwen1.5-MoE, Mixtral, OLMoE, DeepSeek-MoE).

Usage::

    capture = RoutingCapture(model)
    capture.install_hooks()
    model.generate(...)
    capture.save(output_dir / "routing_decisions.npz")
    capture.remove_hooks()
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Gate module name patterns for HuggingFace MoE architectures.
# The gate is an nn.Linear whose out_features == num_experts.
_GATE_NAME_PATTERNS = [
    re.compile(r"layers\.(\d+)\..*\.gate$"),             # Qwen, OLMoE, DeepSeek
    re.compile(r"layers\.(\d+)\.block_sparse_moe\.gate$"),  # Mixtral
    re.compile(r"layers\.(\d+)\..*moe.*\.gate$"),        # generic fallback
    re.compile(r"layers\.(\d+)\..*router.*$"),            # router naming variants
]


# ---------------------------------------------------------------------------
# Gate module discovery
# ---------------------------------------------------------------------------

def discover_gate_modules(
    model: Any,
    num_experts: Optional[int] = None,
) -> List[Tuple[int, str, Any]]:
    """Find all gate/router modules in a HuggingFace MoE model.

    Args:
        model: HuggingFace model (e.g., AutoModelForCausalLM).
        num_experts: Expected number of experts.  If provided, validates
            that gate.out_features == num_experts.  If None, inferred from
            model.config.

    Returns:
        List of (layer_idx, module_name, module) tuples, sorted by layer_idx.
    """
    if num_experts is None:
        config = getattr(model, "config", None)
        if config is not None:
            num_experts = (
                getattr(config, "num_local_experts", None)
                or getattr(config, "num_experts", None)
            )

    gates: List[Tuple[int, str, Any]] = []

    for name, module in model.named_modules():
        # Check name against known patterns.
        for pattern in _GATE_NAME_PATTERNS:
            m = pattern.search(name)
            if m is None:
                continue
            layer_idx = int(m.group(1))

            # Validate: must be nn.Linear-like with out_features == num_experts.
            out_features = getattr(module, "out_features", None)
            if out_features is None:
                # Some models wrap the gate in a custom module.
                weight = getattr(module, "weight", None)
                if weight is not None and hasattr(weight, "shape"):
                    out_features = weight.shape[0]

            if num_experts is not None and out_features != num_experts:
                continue

            gates.append((layer_idx, name, module))
            break  # matched — don't check other patterns

    gates.sort(key=lambda x: x[0])
    return gates


def get_num_experts(model: Any) -> Optional[int]:
    """Extract num_experts from model config."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    return (
        getattr(config, "num_local_experts", None)
        or getattr(config, "num_experts", None)
    )


def get_top_k(model: Any) -> int:
    """Extract top-k experts per token from model config.  Defaults to 2."""
    config = getattr(model, "config", None)
    if config is None:
        return 2
    return (
        getattr(config, "num_experts_per_tok", None)
        or getattr(config, "top_k", None)
        or 2
    )


# ---------------------------------------------------------------------------
# Routing capture
# ---------------------------------------------------------------------------

class RoutingCapture:
    """Captures per-token expert routing decisions during inference.

    Attributes:
        records: List of dicts, one per gate forward call, containing:
            - layer_idx (int)
            - selected_experts (np.ndarray, shape [num_tokens, top_k])
            - routing_probs (np.ndarray, shape [num_tokens, num_experts])
            - routing_logits (np.ndarray, shape [num_tokens, num_experts])
    """

    def __init__(
        self,
        model: Any,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.model = model
        self.num_experts = num_experts or get_num_experts(model)
        self.top_k = top_k or get_top_k(model)
        self.records: List[Dict] = []
        self._hooks: List[Any] = []
        self._gates: List[Tuple[int, str, Any]] = []

    def install_hooks(self) -> int:
        """Register forward hooks on all discovered gate modules.

        Returns:
            Number of hooks installed.
        """
        self._gates = discover_gate_modules(self.model, self.num_experts)
        if not self._gates:
            return 0

        for layer_idx, name, module in self._gates:
            hook = module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

        return len(self._hooks)

    def _make_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook closure for a specific layer."""
        import torch

        def hook_fn(module: Any, input: Any, output: Any) -> None:
            # Gate output = logits over experts: (..., num_experts)
            logits = output.detach().float()

            # Flatten to 2D: (num_tokens, num_experts)
            if logits.dim() > 2:
                logits = logits.reshape(-1, logits.shape[-1])
            elif logits.dim() == 1:
                logits = logits.unsqueeze(0)

            probs = torch.softmax(logits, dim=-1)
            _, topk_ids = torch.topk(logits, k=min(self.top_k, logits.shape[-1]), dim=-1)

            self.records.append({
                "layer_idx": layer_idx,
                "selected_experts": topk_ids.cpu().numpy(),
                "routing_probs": probs.cpu().numpy(),
                "routing_logits": logits.cpu().numpy(),
            })

        return hook_fn

    def remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self) -> None:
        """Clear captured records (keeps hooks installed)."""
        self.records.clear()

    @property
    def num_layers_captured(self) -> int:
        """Number of unique layers that have records."""
        return len(set(r["layer_idx"] for r in self.records))

    def get_layer_records(self, layer_idx: int) -> List[Dict]:
        """Return all records for a specific layer."""
        return [r for r in self.records if r["layer_idx"] == layer_idx]

    def save(
        self,
        output_path: Path,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Save routing decisions to a compressed .npz file.

        Args:
            output_path: Path for the output file (e.g., routing_decisions.npz).
            metadata: Optional dict with prompt_category, seq_len, etc.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)

        if not self.records:
            # Save empty marker.
            np.savez_compressed(
                output_path,
                metadata=json.dumps(metadata or {}),
            )
            return output_path

        # Group records by layer_idx.
        layers = sorted(set(r["layer_idx"] for r in self.records))
        save_dict: Dict[str, Any] = {}

        for layer_idx in layers:
            layer_records = self.get_layer_records(layer_idx)
            # Concatenate across inference steps (prefill + decode tokens).
            experts = np.concatenate(
                [r["selected_experts"] for r in layer_records], axis=0
            )
            probs = np.concatenate(
                [r["routing_probs"] for r in layer_records], axis=0
            )
            save_dict[f"experts_layer_{layer_idx}"] = experts
            save_dict[f"probs_layer_{layer_idx}"] = probs

        save_dict["layer_indices"] = np.array(layers)
        save_dict["metadata"] = json.dumps({
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "num_layers": len(layers),
            "total_records": len(self.records),
            **(metadata or {}),
        })

        np.savez_compressed(output_path, **save_dict)
        return output_path

    @staticmethod
    def load(path: Path) -> Dict:
        """Load routing decisions from a .npz file.

        Returns:
            Dict with keys: metadata (dict), layers (dict mapping
            layer_idx -> {"experts": ndarray, "probs": ndarray}).
        """
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data.get("metadata", "{}")))

        layers = {}
        layer_indices = data.get("layer_indices", np.array([]))
        for layer_idx in layer_indices:
            idx = int(layer_idx)
            experts_key = f"experts_layer_{idx}"
            probs_key = f"probs_layer_{idx}"
            layers[idx] = {
                "experts": data[experts_key] if experts_key in data else None,
                "probs": data[probs_key] if probs_key in data else None,
            }

        return {"metadata": metadata, "layers": layers}
