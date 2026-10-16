"""
SODA Wrapper around Cerebras ModelZoo run_gpt_inference.py.
Injects CerebrasTracer metrics collection into the GPT/Llama inference execution loop.
"""
import sys
import os
import time
import logging
import hashlib
from contextlib import ExitStack
from pathlib import Path
from typing import Any

# Append SODA src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from soda.cerebras.tracer import CerebrasTracer

REPO_ROOT = Path(__file__).resolve().parents[3]

class MockArgs:
    def __init__(self, output_dir, model_name, workload=None):
        self.output_dir = output_dir
        self.model = model_name
        self.workload = workload or {}


DEFAULT_BENCHMARK = {
    "mode": "decode",
    "prompt_tokens": 512,
    "output_tokens": 128,
    "warmup_steps": 5,
    "measured_steps": 30,
    "save_final_prediction": True,
}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _inspect_fixed_prompt_dataset(
    dataset_path: Path, prompt_tokens: int, vocab_size: int
) -> dict:
    """Verify that every benchmark sample contains the same valid prompt."""
    import h5py
    import numpy as np

    files = sorted(dataset_path.glob("*.h5"))
    if not files:
        raise ValueError(f"No HDF5 files found in {dataset_path}")

    digest = hashlib.sha256()
    reference = None
    sample_count = 0
    token_min = None
    token_max = None
    for path in files:
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        with h5py.File(path, "r") as stream:
            if "data" not in stream:
                raise ValueError(f"HDF5 benchmark file has no 'data' dataset: {path}")
            data = stream["data"]
            if data.ndim == 3 and data.shape[1] == 3:
                prompts = data[:, 0, :prompt_tokens]
                available_tokens = data.shape[2]
            elif data.ndim == 2:
                prompts = data[:, :prompt_tokens]
                available_tokens = data.shape[1]
            else:
                raise ValueError(f"Unsupported HDF5 benchmark shape {data.shape}: {path}")
            if available_tokens < prompt_tokens:
                raise ValueError(
                    f"Dataset provides {available_tokens} prompt tokens, "
                    f"but {prompt_tokens} were requested"
                )
            prompts = np.asarray(prompts)
            if prompts.shape[0] == 0:
                continue
            candidate = prompts[0]
            if reference is None:
                reference = candidate
            if not np.all(prompts == reference):
                raise ValueError("Benchmark HDF5 prompts are not identical")
            current_min = int(prompts.min())
            current_max = int(prompts.max())
            token_min = current_min if token_min is None else min(token_min, current_min)
            token_max = current_max if token_max is None else max(token_max, current_max)
            sample_count += prompts.shape[0]

    if sample_count == 0 or reference is None:
        raise ValueError(f"Benchmark dataset is empty: {dataset_path}")
    if token_min < 0 or token_max >= vocab_size:
        raise ValueError(
            f"Prompt token range [{token_min}, {token_max}] exceeds vocabulary "
            f"[0, {vocab_size - 1}]"
        )
    return {
        "dataset_sha256": digest.hexdigest(),
        "dataset_samples": sample_count,
        "prompt_token_min": token_min,
        "prompt_token_max": token_max,
        "fixed_prompts_verified": True,
    }


def configure_benchmark(params: dict) -> dict:
    """Validate and apply SODA's fixed-workload inference contract."""
    raw = {**DEFAULT_BENCHMARK, **(params.get("soda_benchmark") or {})}
    mode = str(raw["mode"]).lower()
    if mode not in {"prefill", "decode"}:
        raise ValueError(
            f"soda_benchmark.mode must be 'prefill' or 'decode', got {mode!r}"
        )

    prompt_tokens = _positive_int(raw["prompt_tokens"], "prompt_tokens")
    output_tokens = _positive_int(raw["output_tokens"], "output_tokens")
    warmup_steps = _positive_int(raw["warmup_steps"], "warmup_steps")
    measured_steps = _positive_int(raw["measured_steps"], "measured_steps")
    if mode == "prefill" and output_tokens != 1:
        raise ValueError("prefill mode requires output_tokens: 1")

    model = params.setdefault("model", {})
    vocab_size = _positive_int(model.get("vocab_size"), "model.vocab_size")
    start_token = vocab_size
    total_sequence_tokens = prompt_tokens + output_tokens
    inference_input = params.setdefault("inference_input", {})
    dataset_path = Path(inference_input.get("data_dir", "dummy_data_msl512"))
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_dir():
        raise ValueError(f"Inference dataset directory does not exist: {dataset_path}")
    inference_input["data_dir"] = str(dataset_path)
    batch_size = _positive_int(inference_input.get("batch_size", 1), "batch_size")
    dataset_metadata = _inspect_fixed_prompt_dataset(
        dataset_path, prompt_tokens, vocab_size
    )
    save_final_prediction = raw["save_final_prediction"]
    if not isinstance(save_final_prediction, bool):
        raise ValueError("save_final_prediction must be true or false")

    workload = {
        "mode": mode,
        "prompt_tokens_per_sample": prompt_tokens,
        "output_tokens_per_sample": output_tokens,
        "total_sequence_tokens": total_sequence_tokens,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "start_token": start_token,
        "early_stop_disabled": True,
        "save_final_prediction": save_final_prediction,
        "dataset_path": str(dataset_path),
        "batch_size": batch_size,
        **dataset_metadata,
    }

    # These are compiler-visible invariants, so SODA owns them rather than
    # allowing duplicate, inconsistent values in Model Zoo configuration.
    model["start_token"] = start_token
    model["max_tokens"] = output_tokens
    model["max_position_embeddings"] = total_sequence_tokens
    model["temperature"] = None
    model["top_k"] = None
    model["top_p"] = None
    params.setdefault("runconfig", {})["inference_steps"] = (
        warmup_steps + measured_steps
    )
    params["soda_benchmark"] = {
        "mode": mode,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "save_final_prediction": workload["save_final_prediction"],
    }
    return workload


def _extract_prediction_array(predictions: Any):
    """Return the sole generated sequence array from a pytree-like result."""
    import numpy as np

    if isinstance(predictions, np.ndarray):
        return predictions
    if isinstance(predictions, dict) and len(predictions) == 1:
        return _extract_prediction_array(next(iter(predictions.values())))
    if isinstance(predictions, (tuple, list)) and len(predictions) == 1:
        return _extract_prediction_array(predictions[0])
    raise ValueError(
        "Expected inference to return one prediction array; "
        f"got {type(predictions).__name__}"
    )


def verify_generated_output(predictions: Any, workload: dict) -> None:
    """Ensure every reserved output slot was filled by the implicit loop."""
    array = _extract_prediction_array(predictions)
    output_tokens = workload["output_tokens_per_sample"]
    if array.ndim < 2 or array.shape[-1] != workload["total_sequence_tokens"]:
        raise ValueError(
            "Unexpected prediction shape: expected final dimension "
            f"{workload['total_sequence_tokens']}, got {array.shape}"
        )
    if (array[..., -output_tokens:] == workload["start_token"]).any():
        raise RuntimeError(
            "Generation did not fill every reserved output slot; throughput is invalid"
        )


def _read_positive_int_env(name, default):
    """Read an optional positive integer without making a run fail on bad env."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed < 1:
            raise ValueError
        return parsed
    except ValueError:
        logging.warning("Ignoring invalid %s=%r; using %s", name, value, default)
        return default


def _op_profiler_window(inference_steps):
    """Return the opt-in OpProfiler window, or ``None`` when disabled."""
    if os.getenv("SODA_CSTORCH_OP_PROFILE", "").lower() not in {"1", "true", "yes"}:
        return None
    default_end = inference_steps if inference_steps else 1
    start_step = _read_positive_int_env("SODA_CSTORCH_PROFILE_START_STEP", 2)
    end_step = _read_positive_int_env("SODA_CSTORCH_PROFILE_END_STEP", default_end)
    if end_step < start_step:
        logging.warning(
            "Disabling CSTorch OpProfiler: end step %s precedes start step %s",
            end_step,
            start_step,
        )
        return None
    return start_step, end_step

def create_soda_inference_dataloader(params):
    from copy import deepcopy
    import torch
    from cerebras.modelzoo.data.nlp.gpt.GptHDF5MapDataProcessor import GptHDF5MapDataProcessor
    from cerebras.modelzoo.common.input_utils import PaddingSample

    params = deepcopy(params)
    inference_input = params["inference_input"]
    workload = params["_soda_workload"]

    def map_fn(x):
        if isinstance(x, PaddingSample):
            x.fill_(1)
            
        if hasattr(x, "shape") and len(x.shape) == 2 and x.shape[0] == 3:
            input_ids = x[0]
        else:
            input_ids = x

        prompt_tokens = workload["prompt_tokens_per_sample"]
        if input_ids.shape[-1] < prompt_tokens:
            raise ValueError(
                f"Dataset provides {input_ids.shape[-1]} prompt tokens, "
                f"but {prompt_tokens} were requested"
            )
        input_ids = torch.as_tensor(input_ids[:prompt_tokens], dtype=torch.int32)
        if torch.any(input_ids < 0) or torch.any(
            input_ids >= params["model"]["vocab_size"]
        ):
            raise ValueError("Synthetic prompt contains token IDs outside the model vocabulary")

        generation_buffer = torch.full(
            (workload["output_tokens_per_sample"],),
            workload["start_token"],
            dtype=torch.int32,
        )
        input_ids = torch.cat((input_ids, generation_buffer))
        # Greedy decoding ignores rand_uniform; retain the official [1] shape
        # with a deterministic value for SDK input compatibility.
        rand_uniform = torch.full((1,), 0.5, dtype=torch.float32)
        # The out-of-vocabulary start sentinel cannot be generated, so this
        # placeholder cannot terminate generation early.
        stop_sequences = torch.full(
            (1, 1), workload["start_token"], dtype=torch.int32
        )
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

    workload = configure_benchmark(params)
    params["_soda_workload"] = workload
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
    mock_args = MockArgs(
        output_dir=str(soda_output_dir),
        model_name=params["model"].get("name", "gpt"),
        workload=workload,
    )
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

    profile_window = (
        None
        if compile_only or validate_only
        else _op_profiler_window(runconfig.get("inference_steps"))
    )
    profiler = None
    profile_stack = ExitStack()
    if profile_window:
        start_step, end_step = profile_window
        try:
            schedule = cstorch.profiler.schedule(
                start_step=start_step,
                end_step=end_step,
            )
            profiler = cstorch.profiler.profile(schedule=schedule)
            profile_stack.enter_context(profiler)
            logging.info(
                "CSTorch OpProfiler enabled for steady-state steps %s-%s.",
                start_step,
                end_step,
            )
        except (AttributeError, TypeError, ValueError) as error:
            logging.warning("CSTorch OpProfiler unavailable; continuing without it: %s", error)
            profiler = None

    # DataExecutor reads the active profiler registry during construction, so
    # the profiler context must be entered before creating the executor.
    try:
        executor = cstorch.utils.data.DataExecutor(
            dataloader,
            num_steps=runconfig.get("inference_steps"),
        )
    except Exception:
        profile_stack.close()
        raise

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
        tracker_snapshot = (
            tracer.snapshot_rate_tracker(executor.profiler.rate_tracker)
            if executor.profiler
            else None
        )
        if executor.user_iteration == workload["warmup_steps"]:
            tracer.set_warmup_snapshot(tracker_snapshot)
        if executor.on_final_iteration:
            tracer.set_final_snapshot(tracker_snapshot)

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

        if (
            executor.on_final_iteration
            and workload["save_final_prediction"]
            and not (compile_only or validate_only)
        ):
            host_predictions = tree_map(cstorch.to_numpy, predictions)
            verify_generated_output(host_predictions, workload)
            tracer.generation_verified = True
            np.savez(
                predictions_dir / "prediction_final.npz",
                predictions=host_predictions,
                global_step=executor.user_iteration,
            )

    from cerebras.modelzoo.trainer.callbacks.summaries import _GLOBAL_SUMMARIES

    class _NoOpTrainer:
        def log_metrics(self, **kwargs):
            pass

    try:
        with _GLOBAL_SUMMARIES._cache_trainer(_NoOpTrainer()):
            for batch in executor:
                start_time = time.time()
                predictions = inference_step(batch)
                post_inference_step(predictions, start_time)
    finally:
        profile_stack.close()
        tracer.export_metrics()
        if executor.profiler:
            try:
                tracer.export_performance(executor.profiler.rate_tracker)
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                logging.warning("Could not export DataExecutor performance telemetry: %s", error)
        if profiler:
            try:
                profiler.export_chrome_trace(str(soda_output_dir / "chrome_trace.json"))
                tracer.export_appliance_summary(profiler)
            except Exception as error:
                logging.warning("Could not export CSTorch profiler artifacts: %s", error)
        if not (compile_only or validate_only) and executor.profiler:
            total_samples = int(executor.profiler.rate_tracker.total_samples)
            total_time = executor.profiler.rate_tracker.total_time
            logging.info(
                f"Processed {total_samples} sample(s) "
                f"in {total_time} seconds."
            )

if __name__ == "__main__":
    run_soda_gpt_inference()
