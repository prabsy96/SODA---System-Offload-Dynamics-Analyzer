"""
Test script to run the parser and analyzer together.
"""
from pathlib import Path
from soda.cerebras.compiler_parser import CerebrasCompilerParser
from soda.cerebras.analyzer import CerebrasAnalyzer

def main():
    base_dir = Path(__file__).parent.resolve()
    
    # 1. Parse compiler logs
    compile_dir = base_dir / "cs_out"
    parser = CerebrasCompilerParser(compile_dir)
    parser.parse()
    
    output_dir = base_dir / "soda_cerebras_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parsed_compile_file = output_dir / "compiled_timings.json"
    parser.export_summary(parsed_compile_file)
    
    # 2. Run Analyzer
    trace_file = output_dir / "GPT-2_cerebras_trace.json"
    analyzer = CerebrasAnalyzer(trace_file, parsed_compile_file)
    analyzer.run()

if __name__ == "__main__":
    main()

