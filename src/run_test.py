"""
Test script to run the parser and analyzer together.
"""
from pathlib import Path
from soda.cerebras.compiler_parser import CerebrasCompilerParser
from soda.cerebras.analyzer import CerebrasAnalyzer

def main():
    # 1. Parse compiler logs
    compile_dir = Path("cs_out")
    parser = CerebrasCompilerParser(compile_dir)
    parser.parse()
    
    parsed_compile_file = Path("soda_cerebras_output/compiled_timings.json")
    parser.export_summary(parsed_compile_file)
    
    # 2. Run Analyzer
    trace_file = Path("soda_cerebras_output/GPT-2_cerebras_trace.json")
    analyzer = CerebrasAnalyzer(trace_file, parsed_compile_file)
    analyzer.run()

if __name__ == "__main__":
    main()
