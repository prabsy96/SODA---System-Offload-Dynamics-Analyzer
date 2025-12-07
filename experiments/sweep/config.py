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

SWEEP_CONFIGS = {
    # GPT-2 short context
    "gpt2_short_ctx": {
        "model_name": "gpt2",
        "batch_sizes": sorted([1, 2, 4, 8], reverse=True),
        "seq_lens": sorted([128, 256, 512, 1024], reverse=True),
        # FIXME: DEBUG ONLY
        # "batch_sizes": sorted([1, 2], reverse=True),
        # "seq_lens": sorted([128, 256], reverse=True),
        "max_new_toks": [1],
    },
    
    # # Llama-3.2-1B short context
    # "llama_3.2_1b_short_ctx": {
    #     "model_name": "meta-llama/Llama-3.2-1B",
    #     "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
    #     "seq_lens": sorted([128, 256, 512, 1024, 2048], reverse=True),
    #     # FIXME: DEBUG ONLY
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