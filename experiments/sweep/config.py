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

## PREFILL
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
    "olmoe_1b_7b_prefill": {
        "model_name": "allenai/OLMoE-1B-7B-0924",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096, 8192], reverse=True),
        "max_new_toks": [1],
    },
    "qwen1.5_moe_a2.7b": {
        "model_name": "Qwen/Qwen1.5-MoE-A2.7B",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096, 8192], reverse=True),
        # "max_new_toks": [10],
        "max_new_toks": [1],
    },
    "gpt_oss_20b": {
    "model_name": "openai/gpt-oss-20b",
    "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),  # lightweight MoE, smaller active params
    "seq_lens": sorted([512, 1024, 2048, 4096], reverse=True),  # supports up to 4k context
    "max_new_toks": [1],
    },
    # "mixtral_8x7b": {
    #     "model_name": "mistralai/Mixtral-8x7B-v0.1",
    #     "batch_sizes": sorted([1, 2, 4], reverse=True),  # Reduced due to model size
    #     "seq_lens": sorted([512, 1024, 2048, 4096], reverse=True),  # Max 4k to avoid OOM
    #     "max_new_toks": [10],
    # },
    "whisper_large_v3_prefill": {
        "model_name": "openai/whisper-large-v3",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        # Sweep Audio Durations: 30s (Standard), 10s, 2s
        # Note: Whisper pads to 30s internally, so these might have similar runtime
        "seq_lens": sorted([32000, 160000, 480000], reverse=True),
        "max_new_toks": [1], 
    },
}

# DECODE
DEC_SWEEP_CONFIG = {
    "gpt2_short_ctx": {
        "model_name": "gpt2",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens":  sorted([128, 256, 512, 1024], reverse=True),
        "max_new_toks": [1], # FIXME: @prabhu should this be 10? 
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
    "olmoe_1b_7b_prefill": {
        "model_name": "allenai/OLMoE-1B-7B-0924",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([512, 1024, 2048, 4096, 8192], reverse=True),
        "max_new_toks": [10],
    },
    "whisper_large_v3_prefill": {
        "model_name": "openai/whisper-large-v3",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        # Sweep Audio Durations: 30s (Standard), 10s, 2s
        # Note: Whisper pads to 30s internally, so these might have similar runtime
        "seq_lens": sorted([32000, 160000, 480000], reverse=True),
        "max_new_toks": [50], 
    },
}

DEBUG_SWEEP_CONFIG = {
    # GPT-2 short context
    "gpt2_short_ctx": {
        "model_name": "gpt2",
        "batch_sizes": sorted([1, 2], reverse=True),
        "seq_lens": sorted([128, 256], reverse=True),
        "max_new_toks": [1],
    },
    
    # # Llama-3.2-1B short context
    # "llama_3.2_1b_short_ctx": {
    #     "model_name": "meta-llama/Llama-3.2-1B",
    #     # "batch_sizes": sorted([1, 2, 4, 8], reverse=True),
    #     # "seq_lens": sorted([512, 1024, 2048], reverse=True),
    #     "max_new_toks": [1],
    # },

    # # TinyLlama-1.1B short context
    # "tinyllama_1.1b": {
    #     "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
    #     "seq_lens": sorted([128, 256, 512, 1024, 2048], reverse=True),
    #     # FIXME: DEBUG ONLY
    #     # "batch_sizes": sorted([1, 2, 4, 8], reverse=True),
    #     # "seq_lens": sorted([512, 1024, 2048], reverse=True),
    #     "max_new_toks": [1],
    # },
}
