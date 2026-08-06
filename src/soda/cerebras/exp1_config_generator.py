"""
Experiment 1 Configuration Generator for Inference SDK.
Generates derived YAML configs for batch size sweep BS in {1, 2, 4, 8, 16, 32, 64}
for Llama3 and GPT-2 models specifically for run_gpt_inference.py.
"""
import os
import yaml
from pathlib import Path

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
OUTPUT_DIR = Path("/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/configs/exp1_batch_scaling")

def generate_llama_configs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_paths = []
    for bs in BATCH_SIZES:
        config = {
            "model": {
                "name": "llama",
                "vocab_size": 128256,
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "num_heads": 32,
                "filter_size": 8192,
                "nonlinearity": "swiglu",
                "norm_type": "rmsnorm",
                "layer_norm_epsilon": 1.0e-05,
                "max_position_embeddings": 512,
                "position_embedding_type": "rotary",
                "rotary_dim": 64,
                "rope_theta": 500000.0,
                "fp16_type": "cbfloat16",
                "start_token": 1,
                "pos_scaling_type": "llama3",
                "pos_scaling_factor": 32.0,
                "pos_scaling_extra_args": {
                    "high_freq_factor": 4.0,
                    "low_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192,
                },
            },
            "runconfig": {
                "precision_opt_level": 1,
                "log_steps": 1,
                "inference_steps": 10,
            },
            "inference_input": {
                "data_processor": "GptHDF5MapDataProcessor",
                "data_dir": "/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/dummy_data_msl512",
                "batch_size": bs,
                "micro_batch_size": bs,
                "drop_last": False,
            },
            "optimizer": {
                "loss_scaling_factor": 1.0,
                "log_summaries": False,
            }
        }
        out_path = OUTPUT_DIR / f"llama3_bs_{bs}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        generated_paths.append(out_path)
        print(f"Generated Llama3 inference config: {out_path} (BS={bs})")
    return generated_paths

def generate_gpt2_configs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_paths = []
    for bs in BATCH_SIZES:
        config = {
            "model": {
                "name": "gpt2",
                "vocab_size": 50257,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_heads": 12,
                "max_position_embeddings": 512,
                "nonlinearity": "gelu",
                "layer_norm_epsilon": 1.0e-05,
                "fp16_type": "cbfloat16",
                "start_token": 1,
            },
            "runconfig": {
                "precision_opt_level": 1,
                "log_steps": 1,
                "inference_steps": 10,
            },
            "inference_input": {
                "data_processor": "GptHDF5MapDataProcessor",
                "data_dir": "/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/dummy_data_msl512",
                "batch_size": bs,
                "micro_batch_size": bs,
                "drop_last": False,
            },
            "optimizer": {
                "loss_scaling_factor": 1.0,
                "log_summaries": False,
            }
        }
        out_path = OUTPUT_DIR / f"gpt2_bs_{bs}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        generated_paths.append(out_path)
        print(f"Generated GPT-2 inference config: {out_path} (BS={bs})")
    return generated_paths

if __name__ == "__main__":
    print("Generating Experiment 1 Inference Configurations...")
    llama_cfgs = generate_llama_configs()
    gpt2_cfgs = generate_gpt2_configs()
    print(f"Total configurations generated: {len(llama_cfgs) + len(gpt2_cfgs)}")
