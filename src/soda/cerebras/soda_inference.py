"""
SODA Wrapper around Cerebras ModelZoo run_gpt_inference.py.
Injects CerebrasTracer metrics collection into the GPT/Llama inference execution loop.
"""
import sys
import os
import time
import logging
from pathlib import Path

# Append SODA src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from soda.cerebras.tracer import CerebrasTracer

class MockArgs:
    def __init__(self, output_dir, model_name):
        self.output_dir = output_dir
        self.model = model_name

def create_soda_inference_dataloader(params):
    from copy import deepcopy
    import torch
    from cerebras.modelzoo.data.nlp.gpt.GptHDF5MapDataProcessor import GptHDF5MapDataProcessor
    from cerebras.modelzoo.common.input_utils import PaddingSample

    params = deepcopy(params)
    inference_input = params["inference_input"]

    def map_fn(x):
        if isinstance(x, PaddingSample):
            start_token = params["model"].get("start_token", 1)
            if isinstance(start_token, list):
                start_token = start_token[0]
            x.fill_(start_token)
            
        if hasattr(x, "shape") and len(x.shape) == 2 and x.shape[0] == 3:
            input_ids = x[0]
        else:
            input_ids = x

        max_tokens = params["model"].get("max_tokens", 1) or 1
        rand_uniform = torch.rand((max_tokens,), dtype=torch.float32)
        stop_sequences = torch.zeros((1, 1), dtype=torch.int32)
        return {
            "input_ids": input_ids,
            "rand_uniform": rand_uniform,
            "stop_sequences": stop_sequences,
        }

    return GptHDF5MapDataProcessor(
        {
            "batch_size": inference_input["batch_size"],
            "data_dir": inference_input["data_dir"],
            "max_sequence_length": inference_input.get("max_sequence_length"),
            "drop_last": inference_input.get("drop_last", False),
            "pad_last": True,
            "dataset_map_fn": map_fn,
        }
    ).create_dataloader()

def run_soda_gpt_inference():
    import cerebras.pytorch as cstorch
    from cerebras.modelzoo.common.run_gpt_inference import (
        get_parser,
        set_defaults,
        RunConfigParamsValidator,
        setup_artifact_dir,
        setup_logging,
        get_cluster_config,
        get_model_checkpoint,
        load_from_checkpoint_file,
        format_rate,
    )
    from cerebras.modelzoo.common.utils.run.cli_parser import get_params_from_args
    from cerebras.modelzoo.common.utils.run.utils import DeviceType
    from cerebras.modelzoo.common.input_utils import validate_streaming_and_micro_batch_size

    import yaml
    import numpy as np
    from torch.utils._pytree import tree_map

    parser_fn = lambda: [get_parser()]
    parser_args = {
        "parser_epilog": "SODA Inference Benchmark",
        "modes": ["inference"],
    }

    params = get_params_from_args(
        argv=sys.argv[1:],
        extra_args_parser_fn=parser_fn,
        device_type=DeviceType.CSX,
        **parser_args,
    )

    set_defaults(params)

    runconfig = params["runconfig"]
    RunConfigParamsValidator(parser_fn).validate(runconfig)

    log_steps = runconfig.get("log_steps", 1)

    artifact_dir = Path(
        setup_artifact_dir(runconfig["model_dir"], mode="inference")
    )
    setup_logging(
        runconfig.get("logging"),
        runconfig.get("streamer_logging"),
        logging_dir=artifact_dir,
        model_dir=runconfig["model_dir"],
    )

    with open(os.path.join(artifact_dir, f"params_inference.yaml"), "w") as f:
        yaml.dump(params, f, default_flow_style=False)

    cluster_config = get_cluster_config(params)

    compile_only = runconfig.get("compile_only", False)
    validate_only = runconfig.get("validate_only", False)

    input_params = params.get("inference_input", {})
    micro_batch_size = input_params.get("micro_batch_size", "auto")
    if "batch_size" in input_params:
        validate_streaming_and_micro_batch_size(
            input_params["batch_size"],
            micro_batch_size,
            cluster_config.num_csx,
        )

    cstorch.backends.csx.performance.micro_batch_size = micro_batch_size
    cstorch.backends.csx.debug.retrace_every_iteration = runconfig.get(
        "retrace_every_iteration", False
    )
    cstorch.backends.csx.debug.lazy_initialization = runconfig.get(
        "lazy_initialization", True
    )

    soda_output_dir = Path(runconfig["model_dir"]) / "soda_output"
    mock_args = MockArgs(output_dir=str(soda_output_dir), model_name=params["model"].get("name", "gpt"))
    tracer = CerebrasTracer(mock_args)
    logging.info(f"SODA Tracer initialized for inference. Output dir: {soda_output_dir}")

    backend = cstorch.backend(
        "CSX",
        artifact_dir=artifact_dir,
        compile_dir=runconfig.get("compile_dir"),
        compile_only=compile_only,
        validate_only=validate_only,
        cluster_config=cluster_config,
    )

    cstorch.amp.set_half_dtype(params["model"].get("fp16_type", "float16"))

    model_name = params["model"].get("name", "").lower()
    # Filter out extra keys not part of Pydantic model configs
    model_kwargs = {k: v for k, v in params["model"].items() if k not in ("fp16_type",)}

    with backend.device:
        if model_name == "llama":
            from cerebras.modelzoo.models.nlp.llama.model import LlamaModel, LlamaModelConfig
            model = LlamaModel(LlamaModelConfig(**model_kwargs))
        else:
            from cerebras.modelzoo.models.nlp.gpt2.model import Gpt2Model, GPT2ModelConfig
            model = Gpt2Model(GPT2ModelConfig(**model_kwargs))

    compiled_model = cstorch.compile(model, backend)
    compiled_model.eval()

    try:
        checkpoint_path = get_model_checkpoint(runconfig)
        if checkpoint_path:
            state_dict = load_from_checkpoint_file(checkpoint_path)
            model.load_state_dict(state_dict["model"], strict=True)
    except Exception as e:
        logging.warning(f"No checkpoint loaded ({e}); running inference latency benchmark with initialized parameters.")

    predictions_dir = artifact_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    dataloader = cstorch.utils.data.DataLoader(
        create_soda_inference_dataloader, params
    )

    executor = cstorch.utils.data.DataExecutor(
        dataloader,
        num_steps=runconfig.get("inference_steps"),
    )

    @cstorch.trace
    def inference_step(batch):
        return compiled_model(batch, autoregressive=True)

    @cstorch.step_closure
    def post_inference_step(predictions, start_time):
        end_time = time.time()
        host_turnaround_s = end_time - start_time
        
        rate = executor.profiler.rate_tracker.rate if executor.profiler else 0.0
        global_rate = executor.profiler.rate_tracker.global_rate if executor.profiler else 0.0

        tracer.log_step_metrics(
            executor.user_iteration,
            0.0,
            rate,
            global_rate,
            host_turnaround_s
        )

        is_log_step = executor.on_final_iteration or (
            log_steps and executor.user_iteration % log_steps == 0
        )

        if is_log_step:
            logging.info(
                f"| Inference Device={backend.device}, "
                f"Step={executor.user_iteration}, "
                f"Rate={format_rate(rate)} samples/sec, "
                f"GlobalRate={format_rate(global_rate)} samples/sec"
            )

        np.savez(
            predictions_dir / f"prediction_{executor.user_iteration}.npz",
            predictions=tree_map(cstorch.to_numpy, predictions),
            global_step=executor.user_iteration,
        )

    try:
        for batch in executor:
            start_time = time.time()
            predictions = inference_step(batch)
            post_inference_step(predictions, start_time)
    finally:
        tracer.export_metrics()
        if not (compile_only or validate_only) and executor.profiler:
            total_samples = int(executor.profiler.rate_tracker.total_samples)
            total_time = executor.profiler.rate_tracker.total_time
            logging.info(
                f"Processed {total_samples} sample(s) "
                f"in {total_time} seconds."
            )

if __name__ == "__main__":
    run_soda_gpt_inference()
