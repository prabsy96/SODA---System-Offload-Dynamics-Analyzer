import sys
import json
from pathlib import Path
from soda.cerebras.compiler_parser import CerebrasCompilerParser

def main():
    target_dir = Path(sys.argv[1])
    print(f"Parsing Cerebras compilation artifacts in {target_dir}...")
    
    parser = CerebrasCompilerParser(target_dir)
    parser.parse()
    
    # We will generate a mock trace file since this was a compile_only run
    mock_trace = target_dir / "trace.json"
    mock_trace_data = {
        "model": "Llama3 Tiny (Compile-Only)",
        "metrics": []
    }
    with open(mock_trace, "w") as f:
        json.dump(mock_trace_data, f)
        
    compile_summary_file = target_dir / "compile_summary.json"
    with open(compile_summary_file, "w") as f:
        json.dump({"module_timings": parser.module_timings}, f)
        
    from soda.cerebras.analyzer import CerebrasAnalyzer
    analyzer = CerebrasAnalyzer(trace_file=mock_trace, compile_summary_file=compile_summary_file)
    analyzer.run()
    print("Done!")

if __name__ == "__main__":
    main()
