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
        self.workload = dict(getattr(args, "workload", {}) or {})
        self.warmup_snapshot: Dict[str, Any] | None = None
        self.final_snapshot: Dict[str, Any] | None = None
        self.generation_verified = False
        
        # We store step metrics here
        self.step_metrics: List[Dict[str, Any]] = []
        self.compile_logs_dir = None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log_step_metrics(self, step: int, loss: float, rate: float, global_rate: float, host_turnaround_s: float = 0.0):
        """
        Callback to be invoked via @cstorch.step_closure
        """
        step_latency = 1.0 / rate if rate and rate > 0 else 0.0
        output_tokens = int(self.workload.get("output_tokens_per_sample", 1))
        metrics = {
            "step": step,
            "loss": loss,
            "rate_samples_s": rate,
            "global_rate_samples_s": global_rate,
            # Retained for compatibility, but now contains actual generated
            # token throughput instead of mislabeled sample throughput.
            "data_pumping_rate_tok_s": rate * output_tokens,
            "global_rate_tok_s": global_rate * output_tokens,
            "step_latency_s": step_latency,
            "host_turnaround_s": host_turnaround_s,
            "host_sync_idle_s": max(0.0, host_turnaround_s - step_latency)
        }
        self.step_metrics.append(metrics)
        LOGGER.info(
            "[CerebrasTracer] Step %s: Rate=%.2f samples/s, OutputRate=%.2f "
            "tokens/s, Latency=%.4f s, HostTurnaround=%.4f s, HostIdle=%.4f s",
            step,
            rate,
            metrics["data_pumping_rate_tok_s"],
            metrics["step_latency_s"],
            metrics["host_turnaround_s"],
            metrics["host_sync_idle_s"],
        )

    @staticmethod
    def snapshot_rate_tracker(rate_tracker: Any) -> Dict[str, Any]:
        """Materialize one internally consistent SDK rate-tracker snapshot."""
        if rate_tracker is None:
            return {}
        snapshot = rate_tracker() if callable(rate_tracker) else None
        if isinstance(snapshot, dict):
            return dict(snapshot)

        result = {}
        for name in (
            "total_samples",
            "total_time",
            "rate",
            "global_rate",
            "flops_utilization",
        ):
            value = getattr(rate_tracker, name, None)
            result[name] = value() if callable(value) else value
        return result

    def set_warmup_snapshot(self, snapshot: Dict[str, Any] | None) -> None:
        self.warmup_snapshot = dict(snapshot or {})

    def set_final_snapshot(self, snapshot: Dict[str, Any] | None) -> None:
        self.final_snapshot = dict(snapshot or {})

    def export_metrics(self) -> Path:
        """
        Dumps the accumulated step metrics to a JSON file.
        """
        out_path = self.output_dir / f"{self.model_name.replace('/', '_')}_cerebras_trace.json"
        with open(out_path, "w") as f:
            json.dump({
                "model": self.model_name,
                "workload": self.workload,
                "metrics": self.step_metrics
            }, f, indent=2)
        LOGGER.info(f"Exported Cerebras metrics to {out_path}")
        return out_path

    def export_performance(self, rate_tracker: Any) -> Path:
        """Export DataExecutor telemetry without claiming hardware counters.

        ``flops_utilization`` is compiler-derived model FLOP utilization (MFU),
        not PE occupancy or a WSE hardware FLOP counter.
        """
        snapshot = self.final_snapshot or self.snapshot_rate_tracker(rate_tracker)

        def value(name: str, default: Any = None) -> Any:
            if name in snapshot:
                return snapshot[name]
            metric = getattr(rate_tracker, name, default)
            return metric() if callable(metric) else metric

        def number(name: str, default: Any = None) -> float | None:
            metric = value(name, default)
            return None if metric is None else float(metric)

        total_samples = value("total_samples", 0)
        baseline_samples = int((self.warmup_snapshot or {}).get("total_samples") or 0)
        baseline_time = float((self.warmup_snapshot or {}).get("total_time") or 0.0)
        final_samples = int(value("total_samples", 0) or 0)
        final_time = float(number("total_time", 0.0) or 0.0)
        measured_samples = max(0, final_samples - baseline_samples)
        measured_time = max(0.0, final_time - baseline_time)
        measured_rate = measured_samples / measured_time if measured_time else 0.0
        output_tokens_per_sample = int(
            self.workload.get("output_tokens_per_sample", 1)
        )
        generated_tokens = measured_samples * output_tokens_per_sample
        measured_steps = int(self.workload.get("measured_steps", 0) or 0)
        batch_execution_time_ms = (
            measured_time * 1000.0 / measured_steps if measured_steps else None
        )
        run_completed = self.final_snapshot is not None and measured_samples > 0
        benchmark_valid = run_completed and self.generation_verified

        payload = {
            "schema_version": 2,
            "total_samples": int(total_samples or 0),
            "total_time_s": number("total_time"),
            "rate_samples_s": number("rate"),
            "global_rate_samples_s": number("global_rate"),
            "measured_samples": measured_samples,
            "measured_time_s": measured_time,
            "measured_rate_samples_s": measured_rate,
            "generated_tokens": generated_tokens,
            "output_tokens_per_sec": (
                generated_tokens / measured_time if measured_time else 0.0
            ),
            "batch_execution_time_ms": batch_execution_time_ms,
            "flops_utilization_pct": number("flops_utilization"),
            "generation_verified": self.generation_verified,
            "run_completed": run_completed,
            "benchmark_valid": benchmark_valid,
            "primary_metric": (
                "batch_execution_time_ms"
                if self.workload.get("mode") == "prefill"
                else "output_tokens_per_sec"
            ),
            "workload": self.workload,
            "metric_scope": {
                "throughput": (
                    "warmup-excluded DataExecutor rate-tracker sample/time deltas; "
                    "generated tokens are measured samples times the fixed output length"
                ),
                "flops_utilization": (
                    "compiler-derived model FLOP utilization; not PE occupancy "
                    "or a hardware FLOP counter"
                ),
            },
        }
        out_path = self.output_dir / "performance.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        LOGGER.info("Exported Cerebras performance telemetry to %s", out_path)
        return out_path

    def export_appliance_summary(self, profiler: Any) -> Path | None:
        """Export a runtime-provided appliance summary when the SDK supplies one."""
        if not getattr(profiler, "appliance_response", None):
            return None
        get_summary = getattr(profiler, "get_summary", None)
        if not callable(get_summary):
            return None
        out_path = self.output_dir / "appliance_summary.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(str(get_summary()))
            f.write("\n")
        LOGGER.info("Exported Cerebras appliance summary to %s", out_path)
        return out_path
