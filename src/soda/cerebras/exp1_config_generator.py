"""
Experiment 1 Configuration Generator.
Generates derived ModelZoo YAML configs for batch size sweep BS in {1, 2, 4, 8, 16, 32, 64}
for Llama3 and GPT-2 models.
"""
import os
import yaml
from pathlib import Path

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
BASE_LLAMA_CONFIG = "/cra-1684/tutorial/igormolybog/training_intro_bundle/3_training/nlp/llama3_tiny_msl_2k.yaml"
OUTPUT_DIR = Path("/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/configs/exp1_batch_scaling")

def generate_llama_configs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASE_LLAMA_CONFIG, "r") as f:
        base_yaml = yaml.safe_load(f)

    # Set loop max steps to 10 for fast execution per batch size
    base_yaml["trainer"]["init"]["loop"]["max_steps"] = 10
    base_yaml["trainer"]["init"]["checkpoint"]["steps"] = 10

    generated_paths = []
    for bs in BATCH_SIZES:
        cfg = yaml.safe_load(yaml.dump(base_yaml))
        cfg["trainer"]["fit"]["train_dataloader"]["batch_size"] = bs
        if "val_dataloader" in cfg["trainer"]["fit"]:
            if isinstance(cfg["trainer"]["fit"]["val_dataloader"], dict):
                cfg["trainer"]["fit"]["val_dataloader"]["batch_size"] = bs

        out_path = OUTPUT_DIR / f"llama3_bs_{bs}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        generated_paths.append(out_path)
        print(f"Generated Llama3 config: {out_path} (BS={bs})")

    return generated_paths

def generate_gpt2_configs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASE_LLAMA_CONFIG, "r") as f:
        base_yaml = yaml.safe_load(f)

    # Modify model settings for GPT-2
    base_yaml["trainer"]["init"]["loop"]["max_steps"] = 10
    base_yaml["trainer"]["init"]["checkpoint"]["steps"] = 10
    base_yaml["trainer"]["init"]["model"] = {
        "name": "gpt2",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 1024,
        "vocab_size": 50257,
        "dropout_rate": 0.1,
        "embedding_dropout_rate": 0.1,
        "attention_dropout_rate": 0.1,
        "nonlinearity": "gelu",
        "layer_norm_epsilon": 1.0e-05,
        "use_bias_in_output": False,
        "loss_scaling": "num_tokens",
        "loss_weight": 1.0
    }

    generated_paths = []
    for bs in BATCH_SIZES:
        cfg = yaml.safe_load(yaml.dump(base_yaml))
        cfg["trainer"]["fit"]["train_dataloader"]["batch_size"] = bs
        if "val_dataloader" in cfg["trainer"]["fit"]:
            if isinstance(cfg["trainer"]["fit"]["val_dataloader"], dict):
                cfg["trainer"]["fit"]["val_dataloader"]["batch_size"] = bs

        out_path = OUTPUT_DIR / f"gpt2_bs_{bs}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        generated_paths.append(out_path)
        print(f"Generated GPT-2 config: {out_path} (BS={bs})")

    return generated_paths

if __name__ == "__main__":
    print("Generating Experiment 1 Batch Scaling Configurations...")
    llama_cfgs = generate_llama_configs()
    gpt2_cfgs = generate_gpt2_configs()
    print(f"Total configurations generated: {len(llama_cfgs) + len(gpt2_cfgs)}")
