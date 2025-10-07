"""
SODA: System-Offload-Dynamics-Analyzer

A lightweight profiler for analyzing the performance dynamics of PyTorch models, focusing on CPU/GPU interactions, kernel launches, and execution dependencies.
This tool is inspired by the principles of detailed system analysis seen in tools like SKIP.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from . import util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def main() -> int:
    """Main entry point for the SODA CLI."""
    parser = argparse.ArgumentParser(
        description="SODA: System-Offload-Dynamics-Analyzer. A profiler for PyTorch models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Hugging Face model identifier to load and analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./soda_results"),
        help="Directory to save analysis artifacts (traces, reports, etc.).",
    )
    parser.add_argument(
        "-c",
        "--compile_type",
        default="eager",
        choices=["eager", "torch.compile", "flash-attention"],
        help="Execution mode for the model.",
    )
    parser.add_argument(
        "-d", "--device", default="cuda", choices=["cpu", "cuda"], help="Device to run the model on."
    )
    parser.add_argument(
        "-p",
        "--precision",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Precision for model weights and operations.",
    )
    parser.add_argument(
        "-sl", "--seq_len", type=int, default=512, help="Sequence length for synthetic input."
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=1, help="Batch size for synthetic input."
    )
    parser.add_argument(
        "-f",
        "--fusion",
        nargs="+",
        type=int,
        help="List of kernel chain lengths to analyze for fusion opportunities.",
    )
    parser.add_argument(
        "-ps",
        "--prox_score",
        type=float,
        default=1.0,
        help="Proximity score threshold (0.0 to 1.0) for fusion recommendations.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    args = parser.parse_args()

    # --- Argument Validation and Setup ---
    if args.device == "cpu" and args.precision == "float16":
        logging.warning("float16 is not supported on CPU. Forcing float32.")
        args.precision = "float32"

    if not torch.cuda.is_available() and args.device == "cuda":
        logging.error("CUDA is not available. Please select --device cpu.")
        return 1

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Results will be saved to: {args.output_dir.resolve()}")

    try:
        # --- Model Loading ---
        logging.info(f"Loading model: {args.model} with precision {args.precision}...")
        model_obj = util.Model(
            model_name=args.model,
            device=args.device,
            compile_type=args.compile_type,
            precision=args.precision,
        )

        if "bert" in args.model or "roberta" in args.model:
            model, tokenizer = model_obj.load_encoder()
            is_decoder = False
        else:
            model, tokenizer = model_obj.load_decoder()
            is_decoder = True
        logging.info("Model loaded successfully.")

        # --- Synthetic Data Generation ---
        logging.info(f"Generating synthetic input: batch_size={args.batch_size}, seq_len={args.seq_len}")
        token_ids = {
            "input_ids": torch.randint(
                1, tokenizer.vocab_size, size=(args.batch_size, args.seq_len), device=args.device
            ),
            "attention_mask": torch.ones(
                args.batch_size, args.seq_len, device=args.device
            ),
        }

        # --- Tracing and Analysis ---
        trace_dir_name = f"{args.model.replace('/', '_')}_{args.compile_type}_bs{args.batch_size}_sl{args.seq_len}"
        trace_obj = util.TraceModel(
            name=trace_dir_name,
            file="trace.json",
            path=str(args.output_dir),
            model=model,
        )

        #  Single warm-up and trace run is performed.
        logging.info("Starting model profiling run...")
        if is_decoder:
            json_file = trace_obj.trace_forward_pass_for_decoder(
                token_ids, tokenizer, args.batch_size, args.seq_len
            )
        else:
            json_file = trace_obj.trace_forward_pass_for_encoder(token_ids, tokenizer)
        logging.info(f"Chrome trace file generated at: {json_file}")

        # --- Data Processing and Reporting ---
        logging.info("Analyzing trace data to generate reports...")
        begin_time = trace_obj.generateCPUSpecificOPs(json_file)
        kernel_trace_file = trace_obj.generateGPUSpecificOPs(json_file)
        dependence_list, exec_time, num_kernels, end_time, idle_time, kernel_only_file = trace_obj.GenDependency(kernel_trace_file)

        runtime = end_time - begin_time
        launch_time = trace_obj.LaunchTax(dependence_list, num_kernels)
        trace_obj.AKD(kernel_only_file)

        logging.info("--- Performance Metrics ---")
        logging.info(f"Inference runtime (ms): {runtime / 1000:.4f}")
        logging.info(f"Total launch overhead (ms): {launch_time:.4f}")
        logging.info(f"Mean GPU idle time (ms): {idle_time / 1000:.4f}")
        logging.info(f"Total kernel execution time (ms): {exec_time / 1000:.4f}")
        logging.info(f"Number of kernels launched: {num_kernels}")
        if num_kernels > 0:
            logging.info(f"Avg. launch latency per kernel (ms): {launch_time / num_kernels:.4f}")
            logging.info(f"Avg. execution time per kernel (ms): {(exec_time / 1000) / num_kernels:.4f}")

        trace_obj.topKkernels(dependence_list)

        if args.fusion:
            logging.info("--- Kernel Fusion Analysis ---")
            for f in args.fusion:
                trace_obj.kernelchains(dependence_list, f, args.prox_score)

        logging.info("Analysis complete.")
        return 0

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
