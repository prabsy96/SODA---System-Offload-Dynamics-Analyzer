"""
Example minimal runner for Cerebras ModelZoo using the SODA CerebrasTracer.
"""
import argparse
import logging
import os
import sys

try:
    import cerebras.pytorch as cstorch
    from cerebras.pytorch.utils.data import DataExecutor
    from cerebras.modelzoo.cli.main import ModelZooCLI
except ImportError:
    print("cerebras.pytorch or modelzoo not found. Ensure you are in the Cerebras environment.")
    sys.exit(1)

from pathlib import Path

from soda.cerebras.tracer import CerebrasTracer

def main():
    default_out = Path(__file__).parent.parent.resolve() / "soda_cerebras_output"
    parser = argparse.ArgumentParser(description="SODA Cerebras Inference Runner")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output-dir", type=str, default=str(default_out), help="Output directory for traces")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to trace")

    
    # We parse known args so we can pass the rest to ModelZooCLI if needed
    args, unknown = parser.parse_known_args()
    
    tracer = CerebrasTracer(args)
    
    # Step closure hook
    @cstorch.step_closure
    def log_soda_metrics(loss, step):
        # We need access to the executor to get the rate
        executor = cstorch.backend().executor
        if executor and hasattr(executor, 'profiler'):
            rate = executor.profiler.rate()
            global_rate = executor.profiler.global_rate()
            tracer.log_step_metrics(step.item(), loss.item(), rate, global_rate)
    
    print(f"Initializing Cerebras ModelZoo wrapper for {args.model}...")
    
    # Normally here you would initialize your dataloader and model via ModelZoo,
    # and call cstorch.utils.data.DataExecutor. For this minimal example, we show
    # the integration point.
    
    # --- Example pseudo-code for the training loop ---
    # model = load_model(args.model)
    # dataloader = get_dataloader(...)
    # executor = DataExecutor(dataloader, ...)
    # 
    # for step, batch in enumerate(executor):
    #     loss = model(batch)
    #     loss.backward()
    #     optimizer.step()
    #     log_soda_metrics(loss, cstorch.make_tensor(step))
    # -------------------------------------------------
    
    # Since this is just a wrapper demonstration to pipe into SODA, we export metrics at the end:
    print("Mocking a run and saving metrics...")
    for i in range(args.steps):
        # mock rate = ~50 tokens/sec
        tracer.log_step_metrics(i, 0.5, 50.0 + i, 50.0)
        
    tracer.export_metrics()
    print("Run complete.")

if __name__ == "__main__":
    main()
