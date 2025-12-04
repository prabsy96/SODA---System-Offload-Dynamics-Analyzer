#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Example showing programmatic use of SODA.

Loads a Llama model, profiles one shape, and saves the report without
going through the CLI entrypoint.
"""

import os
import sys
from argparse import Namespace
from pathlib import Path

from soda import SodaAnalyzer, ModelTracer, SodaLogger
from soda.common import utils


def ensure_env_loaded() -> None:
    """Exit early if env.sh was not sourced."""
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)


def build_args(model: str, batch_size: int, seq_len: int, max_new_tokens: int = 1) -> Namespace:
    """Convenience wrapper around the built-in parser."""
    cli_args = [
        "--model", model,
        "--output-dir", os.environ.get("SODA_OUTPUT", "."),
        "--compile-type", "eager",
        "--device", "cuda",
        "--precision", "bfloat16",
        "--batch-size", str(batch_size),
        "--seq-len", str(seq_len),
        "--max-new-tokens", str(max_new_tokens),
    ]
    return utils.parse_and_validate_args(cli_args)


def main() -> None:
    """Example usage of SODA programmatically."""
    ensure_env_loaded()

    model = "meta-llama/Llama-3.2-3B"
    args = build_args(model=model, batch_size=1, seq_len=128, max_new_tokens=1)

    # If you prefer to skip CLI parsing, uncomment and edit the namespace
    # below. All supported args are mentioned below.
    #
    # args = Namespace(
    #     model=model,
    #     output_dir=Path(os.environ.get("SODA_OUTPUT", ".")),
    #     compile_type="eager",
    #     device="cuda",
    #     precision="bfloat16",
    #     batch_size=1,
    #     seq_len=128,
    #     max_new_tokens=1,
    #     fusion=None,
    #     prox_score=1.0,
    #     seed=42,
    #     microbench=False,
    #     runs=5,
    #     warmup=10,
    # )

    tracer = ModelTracer(args=args)
    SodaLogger(tracer.output_dir, is_console=True, is_file=True)
    tracer.run()

    analyzer = SodaAnalyzer(tracer=tracer, args=args)
    report_path = analyzer.run()
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
