# Each config declares the model to test and the BS/SL sweeps to run.

# Soda Analyzer
INFERENCE_WARMUP = "5" # Default; inference time can be in seconds so 5x = 5 seconds

# Soda Microbench
MICROBENCH_WARMUP = "1000"
MICROBENCH_RUNS = "5000"

# Debug mode
DEBUG = True
if DEBUG:
    INFERENCE_WARMUP = "1"
    MICROBENCH_WARMUP = "1"
    MICROBENCH_RUNS = "1"

PARAMS = {
    "compile_type": "eager",
    "precision": "bfloat16",
    "device": "cuda",
    "inference_warmup": INFERENCE_WARMUP,
    "microbench_warmup": MICROBENCH_WARMUP,
    "microbench_runs": MICROBENCH_RUNS,
}

## PREFILL (max_new_tokens=1)
PREF_SWEEP_CONFIG = {
    "gpt2_short_ctx": {
        "model_name": "gpt2",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens":  sorted([128, 256, 512, 1024], reverse=True),
        "max_new_toks": [1],
    },
    "llama_3.2_1b_short_ctx": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096, 8192], reverse=True),
        "max_new_toks": [1],
    },
    "tinyllama_1.1b": {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([128, 256, 512, 1024, 2048], reverse=True),
        "max_new_toks": [1],
    },
    "olmoe_1b_7b": {
        "model_name": "allenai/OLMoE-1B-7B-0924",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096, 8192], reverse=True),
        "max_new_toks": [1],
    },
    "qwen1.5_moe_a2.7b": {
        "model_name": "Qwen/Qwen1.5-MoE-A2.7B",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096, 8192], reverse=True),
        "max_new_toks": [1],
    },
    "gpt_oss_20b": {
        "model_name": "openai/gpt-oss-20b",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096], reverse=True),
        "max_new_toks": [1],
    },
    "whisper_large_v3": {
        "model_name": "openai/whisper-large-v3",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([32000, 160000, 480000], reverse=True),
        "max_new_toks": [1],
    },
    "gemma_2b": {
        "model_name": "google/gemma-2b",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096, 8192], reverse=True),
        "max_new_toks": [1],
    },
}

## DECODE (max_new_tokens=10+)
DEC_SWEEP_CONFIG = {
    "gpt_oss_20b": {
        "model_name": "openai/gpt-oss-20b",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096], reverse=True),
        "max_new_toks": [10],
    },
    "llama_3.2_1b_short_ctx": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096, 8192], reverse=True),
        "max_new_toks": [10],
    },
    "tinyllama_1.1b": {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([128, 256, 512, 1024, 2048], reverse=True),
        "max_new_toks": [10],
    },
    "qwen1.5_moe_a2.7b": {
        "model_name": "Qwen/Qwen1.5-MoE-A2.7B",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096, 8192], reverse=True),
        "max_new_toks": [10],
    },
    "olmoe_1b_7b": {
        "model_name": "allenai/OLMoE-1B-7B-0924",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096, 8192], reverse=True),
        "max_new_toks": [10],
    },
    "whisper_large_v3": {
        "model_name": "openai/whisper-large-v3",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([32000, 160000, 480000], reverse=True),
        "max_new_toks": [50],
    },
}

## DEBUG (quick sanity check)
DEBUG_SWEEP_CONFIG = {
    "gpt2_short_ctx": {
        "model_name": "gpt2",
        "batch_sizes": sorted([1, 2], reverse=True),
        "seq_lens": sorted([128, 256], reverse=True),
        "max_new_toks": [1],
    },
}

## FP8 (H100/H200 only)
FP8_SWEEP_CONFIG = {
    "gpt_oss_20b_fp8": {
        "model_name": "openai/gpt-oss-20b",
        "batch_sizes": sorted([8, 16]),
        "seq_lens": sorted([512, 1024, 2048, 4096]),
        "max_new_toks": [1],
        "precision": "float8_e4m3fn",
    },
}
