"""
Cerebras-specific tracing and profiling hooks.
Replaces the GPU-specific `torch.profiler` based ModelTracer with one that intercepts
metrics from `cerebras.pytorch` DataExecutor using @cstorch.step_closure.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

try:
    import cerebras.pytorch as cstorch
except ImportError:
    cstorch = None

LOGGER = logging.getLogger("soda.cerebras")

class CerebrasTracer:
    def __init__(self, args: Any):
        self.args = args
        self.output_dir = Path(getattr(args, "output_dir", "output"))
        self.model_name = getattr(args, "model", "unknown")
        
        # We store step metrics here
        self.step_metrics: List[Dict[str, Any]] = []
        self.compile_logs_dir = None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log_step_metrics(self, step: int, loss: float, rate: float, global_rate: float, host_turnaround_s: float = 0.0):
        """
        Callback to be invoked via @cstorch.step_closure
        """
        step_latency = 1.0 / rate if rate and rate > 0 else 0.0
        metrics = {
            "step": step,
            "loss": loss,
            "data_pumping_rate_tok_s": rate,
            "global_rate_tok_s": global_rate,
            "step_latency_s": step_latency,
            "host_turnaround_s": host_turnaround_s,
            "host_sync_idle_s": max(0.0, host_turnaround_s - step_latency)
        }
        self.step_metrics.append(metrics)
        LOGGER.info(f"[CerebrasTracer] Step {step}: Rate={rate:.2f} tok/s, Latency={metrics['step_latency_s']:.4f} s, HostTurnaround={metrics['host_turnaround_s']:.4f} s, HostIdle={metrics['host_sync_idle_s']:.4f} s")

    def export_metrics(self) -> Path:
        """
        Dumps the accumulated step metrics to a JSON file.
        """
        out_path = self.output_dir / f"{self.model_name.replace('/', '_')}_cerebras_trace.json"
        with open(out_path, "w") as f:
            json.dump({
                "model": self.model_name,
                "metrics": self.step_metrics
            }, f, indent=2)
        LOGGER.info(f"Exported Cerebras metrics to {out_path}")
        return out_path
