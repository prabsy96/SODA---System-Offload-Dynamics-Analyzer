"""Generate fixed-workload GPT-2/Llama CSX batch-scaling configurations."""

from copy import deepcopy
from pathlib import Path

import yaml

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "configs" / "exp1_batch_scaling"
DATA_DIR = "dummy_data_msl512"
WORKLOADS = {
    "decode": {"prompt_tokens": 512, "output_tokens": 10},
    "prefill": {"prompt_tokens": 512, "output_tokens": 1},
}


def _base_config(model: str, batch_size: int, mode: str) -> dict:
    if model == "llama3":
        model_config = {
            "name": "llama",
            "vocab_size": 128256,
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_heads": 32,
            "filter_size": 8192,
            "nonlinearity": "swiglu",
            "norm_type": "rmsnorm",
            "layer_norm_epsilon": 1.0e-5,
            "position_embedding_type": "rotary",
            "rotary_dim": 64,
            "rope_theta": 500000.0,
            "fp16_type": "cbfloat16",
            "pos_scaling_type": "llama3",
            "pos_scaling_factor": 32.0,
            "pos_scaling_extra_args": {
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
            },
        }
    elif model == "gpt2":
        model_config = {
            "name": "gpt2",
            "vocab_size": 50257,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_heads": 12,
            "nonlinearity": "gelu",
            "layer_norm_epsilon": 1.0e-5,
            "fp16_type": "cbfloat16",
        }
    elif model == "olmoe":
        model_config = {
            "name": "llama",
            "vocab_size": 50304,
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_heads": 16,
            "filter_size": 1024,
            "nonlinearity": "swiglu",
            "norm_type": "rmsnorm",
            "layer_norm_epsilon": 1.0e-5,
            "position_embedding_type": "rotary",
            "rotary_dim": 64,
            "fp16_type": "cbfloat16",
            "moe": {
                "num_experts": 64,
                "top_k": 8,
                "moe_implementation": "experimental",
                "routing_algorithm": "learned",
            },
        }
    else:
        raise ValueError(f"Unknown model {model!r}")

    workload = WORKLOADS[mode]
    model_config.update(
        {
            "start_token": model_config["vocab_size"],
            "max_tokens": workload["output_tokens"],
            "max_position_embeddings": (
                workload["prompt_tokens"] + workload["output_tokens"]
            ),
        }
    )
    return {
        "soda_benchmark": {
            "mode": mode,
            **workload,
            "warmup_steps": 2,
            "measured_steps": 8,
            "save_final_prediction": True,
        },
        "model": model_config,
        "runconfig": {
            "precision_opt_level": 1,
            "log_steps": 1,
            "inference_steps": 10,
        },
        "inference_input": {
            "data_processor": "GptHDF5MapDataProcessor",
            "data_dir": DATA_DIR,
            "batch_size": batch_size,
            "micro_batch_size": batch_size,
            "drop_last": False,
        },
        "optimizer": {
            "loss_scaling_factor": 1.0,
            "log_summaries": False,
        },
    }


def generate_model_configs(model: str) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_paths = []
    for batch_size in BATCH_SIZES:
        for mode in WORKLOADS:
            config = deepcopy(_base_config(model, batch_size, mode))
            # Existing names remain decode configs for backward compatibility.
            suffix = "" if mode == "decode" else "_prefill"
            out_path = OUTPUT_DIR / f"{model}_bs_{batch_size}{suffix}.yaml"
            with open(out_path, "w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, default_flow_style=False, sort_keys=False)
            generated_paths.append(out_path)
            print(
                f"Generated {model} {mode} config: {out_path} "
                f"(BS={batch_size})"
            )
    return generated_paths


def generate_llama_configs() -> list[Path]:
    return generate_model_configs("llama3")


def generate_gpt2_configs() -> list[Path]:
    return generate_model_configs("gpt2")


def generate_olmoe_configs() -> list[Path]:
    return generate_model_configs("olmoe")


if __name__ == "__main__":
    print("Generating Experiment 1 fixed-workload inference configurations...")
    paths = generate_llama_configs() + generate_gpt2_configs() + generate_olmoe_configs()
    print(f"Total configurations generated: {len(paths)}")
