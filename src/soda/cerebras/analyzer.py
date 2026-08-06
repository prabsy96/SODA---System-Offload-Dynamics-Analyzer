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
        
    def run(self):
        """
        Loads the trace JSONs and outputs a SODA-style Cerebras performance report.
        """
        if not self.trace_file.exists():
            LOGGER.error(f"Trace file {self.trace_file} not found.")
            return
            
        with open(self.trace_file, "r") as f:
            self.metrics_data = json.load(f)
            
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
        
        step_metrics = self.metrics_data.get("metrics", [])
        if step_metrics:
            avg_rate = sum(m.get("data_pumping_rate_tok_s", 0) for m in step_metrics) / len(step_metrics)
            avg_latency = sum(m.get("step_latency_s", 0) for m in step_metrics) / len(step_metrics)
            print(f"Average Data Pumping Rate : {avg_rate:.2f} tokens/s")
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
        
        lines = [
            f"# SODA Cerebras Stage 1 Performance Report",
            f"",
            f"**Model:** `{model_name}`  ",
            f"**Architecture:** Cerebras CS-3 Wafer-Scale Engine  ",
            f"",
            f"## Execution Metrics",
            f"",
        ]
        
        if step_metrics:
            avg_rate = sum(m.get("data_pumping_rate_tok_s", 0) for m in step_metrics) / len(step_metrics)
            avg_latency = sum(m.get("step_latency_s", 0) for m in step_metrics) / len(step_metrics)
            lines.extend([
                f"- **Average Throughput (Data Pumping Rate):** `{avg_rate:.2f} tok/s`",
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

