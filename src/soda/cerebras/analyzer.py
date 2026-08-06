"""
Cerebras-specific reporting frontend (Pathway B).
Replaces the GPU-centric SodaAnalyzer with a tailored analyzer that processes
Cerebras step metrics and compile-time logs to produce a performance report.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any

from soda.common import print_utils

LOGGER = logging.getLogger("soda.cerebras.analyzer")

class CerebrasAnalyzer:
    def __init__(self, trace_file: Path, compile_summary_file: Path = None):
        self.trace_file = trace_file
        self.compile_summary_file = compile_summary_file
        
        self.metrics_data: Dict[str, Any] = {}
        self.compile_data: Dict[str, Any] = {}
        self.performance_data: Dict[str, Any] = {}
        
    def run(self):
        """
        Loads the trace JSONs and outputs a SODA-style Cerebras performance report.
        """
        if not self.trace_file.exists():
            LOGGER.error(f"Trace file {self.trace_file} not found.")
            return
            
        with open(self.trace_file, "r") as f:
            self.metrics_data = json.load(f)

        performance_file = self.trace_file.parent / "performance.json"
        if performance_file.exists():
            with open(performance_file, "r", encoding="utf-8") as f:
                self.performance_data = json.load(f)
            
        if self.compile_summary_file and self.compile_summary_file.exists():
            with open(self.compile_summary_file, "r") as f:
                self.compile_data = json.load(f)
                
        self.generate_report()
        self.export_markdown_report()
        
    def generate_report(self):
        """
        Prints the SODA Cerebras Performance Summary.
        """
        print_utils.section_start("SODA Cerebras Wafer-Scale Performance Summary")
        
        model_name = self.metrics_data.get("model", "Unknown")
        print(f"Model: {model_name}\n")
        workload = self.metrics_data.get("workload", {})
        if workload:
            print(
                "Workload: "
                f"{workload.get('mode')} "
                f"({workload.get('prompt_tokens_per_sample')} input / "
                f"{workload.get('output_tokens_per_sample')} output tokens)"
            )

        if self.performance_data.get("schema_version", 0) >= 2:
            print(
                "Benchmark Valid: "
                f"{self.performance_data.get('benchmark_valid', False)}"
            )
            if workload.get("mode") == "prefill":
                print(
                    "Prefill + First-Token Batch Latency: "
                    f"{(self.performance_data.get('batch_execution_time_ms') or 0):.2f} ms"
                )
            else:
                print(
                    "Measured Output Throughput: "
                    f"{self.performance_data.get('output_tokens_per_sec', 0):.2f} tokens/s"
                )
            print(
                "Measured Window: "
                f"{self.performance_data.get('measured_samples', 0)} samples in "
                f"{self.performance_data.get('measured_time_s', 0):.4f} s"
            )
        
        step_metrics = self.metrics_data.get("metrics", [])
        if step_metrics:
            avg_rate = sum(
                m.get("rate_samples_s", m.get("data_pumping_rate_tok_s", 0))
                for m in step_metrics
            ) / len(step_metrics)
            avg_latency = sum(m.get("step_latency_s", 0) for m in step_metrics) / len(step_metrics)
            print(f"Average Data Pumping Rate : {avg_rate:.2f} samples/s")
            print(f"Average Step Latency      : {avg_latency:.4f} s")
            print(f"Total Steps Traced        : {len(step_metrics)}")
        else:
            print("No step metrics found.")
            
        if self.compile_data:
            print_utils.section_start("Wafer-Side Module Compilation Timings")
            modules = self.compile_data.get("module_timings", [])
            if not modules:
                print("No module timings available.")
            else:
                print(f"{'Module Name':<40} | {'Op Type':<15} | {'Latency (us)':<12} | {'PEs Used':<8}")
                print("-" * 80)
                # Sort by latency descending
                modules.sort(key=lambda x: x.get("estimated_latency_us", 0), reverse=True)
                for mod in modules[:10]: # Top 10
                    print(f"{mod.get('module_name')[:38]:<40} | {mod.get('op_type')[:13]:<15} | {mod.get('estimated_latency_us'):<12.2f} | {mod.get('wafer_area_pe'):<8}")
                if len(modules) > 10:
                    print(f"... and {len(modules) - 10} more modules.")
        
        print("\nReport generation complete.")

    def export_markdown_report(self) -> Path:
        """
        Generates and saves a Markdown performance report.
        """
        output_dir = self.trace_file.parent
        report_path = output_dir / "soda_cerebras_report.md"
        
        model_name = self.metrics_data.get("model", "Unknown")
        step_metrics = self.metrics_data.get("metrics", [])
        workload = self.metrics_data.get("workload", {})
        
        lines = [
            f"# SODA Cerebras Stage 1 Performance Report",
            f"",
            f"**Model:** `{model_name}`  ",
            f"**Architecture:** Cerebras CS-3 Wafer-Scale Engine  ",
            f"**Workload:** `{workload.get('mode', 'legacy')}` — "
            f"`{workload.get('prompt_tokens_per_sample', 'unknown')}` input / "
            f"`{workload.get('output_tokens_per_sample', 'unknown')}` output tokens  ",
            f"",
            f"## Execution Metrics",
            f"",
        ]

        if self.performance_data.get("schema_version", 0) >= 2:
            primary_line = (
                f"- **Prefill + first-token batch latency:** "
                f"`{(self.performance_data.get('batch_execution_time_ms') or 0):.2f} ms`"
                if workload.get("mode") == "prefill"
                else f"- **Warmup-excluded output throughput:** "
                f"`{self.performance_data.get('output_tokens_per_sec', 0):.2f} tokens/s`"
            )
            lines.extend([
                primary_line,
                f"- **Measured samples:** `{self.performance_data.get('measured_samples', 0)}`",
                f"- **Measured execution time:** "
                f"`{self.performance_data.get('measured_time_s', 0):.4f} s`",
                f"- **Generation length verified:** "
                f"`{self.performance_data.get('generation_verified', False)}`",
                f"- **Benchmark valid:** "
                f"`{self.performance_data.get('benchmark_valid', False)}`",
                f"",
            ])
        
        if step_metrics:
            avg_rate = sum(
                m.get("rate_samples_s", m.get("data_pumping_rate_tok_s", 0))
                for m in step_metrics
            ) / len(step_metrics)
            avg_latency = sum(m.get("step_latency_s", 0) for m in step_metrics) / len(step_metrics)
            lines.extend([
                f"- **Average Throughput (Data Pumping Rate):** `{avg_rate:.2f} samples/s`",
                f"- **Average Step Latency:** `{avg_latency:.4f} s` (`{avg_latency * 1000:.2f} ms`)",
                f"- **Total Profiled Steps:** `{len(step_metrics)}`",
                f"",
            ])
        
        if self.compile_data:
            modules = self.compile_data.get("module_timings", [])
            lines.extend([
                f"## Wafer-Side Module Compilation Timings",
                f"",
                f"| Module Name | Op Type | Latency (µs) | PEs Used |",
                f"| :--- | :--- | :--- | :--- |",
            ])
            modules.sort(key=lambda x: x.get("estimated_latency_us", 0), reverse=True)
            for mod in modules:
                lines.append(
                    f"| `{mod.get('module_name')}` | `{mod.get('op_type')}` | `{mod.get('estimated_latency_us'):.2f}` | `{mod.get('wafer_area_pe')}` |"
                )
            lines.append("")
            
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
            
        LOGGER.info(f"Exported Cerebras Markdown report to {report_path}")
        return report_path
