"""
Parser for Cerebras Ahead-Of-Time (AOT) compiler logs.
Extracts wafer-side PyTorch module timings and hardware mapping latencies.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List

LOGGER = logging.getLogger("soda.cerebras.compiler_parser")

class CerebrasCompilerParser:
    def __init__(self, compile_dir: Path):
        self.compile_dir = compile_dir
        self.module_timings: List[Dict[str, Any]] = []
        
    def parse(self):
        """
        Parses the Cerebras compile directory for performance metrics.
        Usually, Cerebras dumps execution graph JSONs and latency estimates in the 
        compile output directory (e.g., inside cs_xx/ or performance_model/).
        """
        if not self.compile_dir.exists():
            LOGGER.warning(f"Compile directory {self.compile_dir} does not exist. Skipping compile log parsing.")
            return

        # Look for typical Cerebras trace or performance JSON files.
        # This is an illustrative parsing mechanism as exact file names vary by CS SDK version.
        perf_files = list(self.compile_dir.rglob("*performance*.json"))
        if not perf_files:
            LOGGER.warning(f"No performance JSONs found in {self.compile_dir}")
            return
            
        for p_file in perf_files:
            try:
                with open(p_file, "r") as f:
                    data = json.load(f)
                    
                # Assuming structure contains a "nodes" or "modules" array with latency estimates
                if "modules" in data:
                    for mod in data["modules"]:
                        self.module_timings.append({
                            "module_name": mod.get("name", "unknown"),
                            "op_type": mod.get("op_type", "unknown"),
                            "estimated_latency_us": mod.get("latency_us", 0.0),
                            "wafer_area_pe": mod.get("num_pe", 0), # processing elements
                        })
            except Exception as e:
                LOGGER.error(f"Error parsing {p_file}: {e}")
                
        LOGGER.info(f"Parsed {len(self.module_timings)} module timings from compiler logs.")
        
    def export_summary(self, out_path: Path):
        """
        Exports the parsed compile-time timings to a JSON file.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "source_dir": str(self.compile_dir),
                "module_timings": self.module_timings
            }, f, indent=2)
        LOGGER.info(f"Exported compiler timings to {out_path}")

