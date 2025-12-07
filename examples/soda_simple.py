#!/usr/bin/env python3
"""
Example showing programmatic use of SODA.
"""

import os
import sys
from pathlib import Path

from soda import SodaAnalyzer, ModelTracer
from soda.common import utils


def ensure_env_loaded() -> None:
    """Exit early if env.sh was not sourced."""
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Example usage of SODA programmatically."""
    ensure_env_loaded()

    # model = "meta-llama/Llama-3.2-3B"
    # model = "meta-llama/Llama-3.2-1B"
    model = "gpt2"
    cli_args = [
        "--model", model,
        "--output-dir", str(Path(os.environ.get("SODA_OUTPUT", "."))),
        "--compile-type", "eager",
        "--device", "cuda",
        "--precision", "bfloat16",
        "--seq-len", "128",
        "--max-new-tokens", "1",
        "--batch-size", "1",
        "--warmup", "5",
        # Extra parser knobs (fusion/microbench/etc.) left at defaults:
        # "--fusion", "2",
        # "--prox-score", "1.0",
        # "--seed", "42",
        # "--microbench",
        # "--runs", "5",
        # "--warmup", "10",
        # "--version",
    ]
    args = utils.parse_and_validate_args(cli_args)

    tracer = ModelTracer(args=args)
    tracer.run()

    analyzer = SodaAnalyzer(tracer=tracer, args=args)
    report_path = analyzer.run()
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
