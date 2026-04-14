# ==============================================================================
# SODA: System Offload Dynamics Analyzer
# Docker image for GPU profiling of LLM inference workloads
# ==============================================================================
#
# Build:
#   docker build -t soda .
#   docker build -t soda --build-arg CUDA_ARCHS="90" .       # H100 only
#   docker build -t soda --build-arg INSTALL_FLASH_ATTN=1 .  # with flash-attn
#
# Run:
#   docker run --gpus all -v ./output:/app/output soda \
#       soda-cli --model gpt2 --seq-len 512 --batch-size 1
#
# ==============================================================================

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# ---------- Build arguments ----------
ARG PYTHON_VERSION=3.11
ARG INSTALL_FLASH_ATTN=0
ARG INSTALL_TRANSFORMER_ENGINE=0
ARG CUDA_ARCHS="70;75;80;86;89;90"

# ---------- Prevent interactive prompts during apt ----------
ENV DEBIAN_FRONTEND=noninteractive

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        make \
        git \
        curl \
        ninja-build \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------- Upgrade pip ----------
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ---------- Install PyTorch with CUDA 12.6 support ----------
# Largest layer — placed early for cache optimization
RUN python -m pip install --no-cache-dir \
    "torch>=2.6.0" \
    --index-url https://download.pytorch.org/whl/cu126

# ---------- Set up working directory ----------
WORKDIR /app

# ---------- Copy package metadata first for layer caching ----------
COPY pyproject.toml LICENSE README.md ./

# ---------- Copy source code ----------
COPY src/ ./src/

# ---------- Install SODA and dependencies ----------
RUN python -m pip install --no-cache-dir -e .

# ---------- Build baremetal GEMM binary ----------
RUN mkdir -p /app/src/soda/microbench/baremetal/build \
    && cmake \
        -S /app/src/soda/microbench/baremetal \
        -B /app/src/soda/microbench/baremetal/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && cmake --build /app/src/soda/microbench/baremetal/build -- -j$(nproc)

# ---------- Optional: flash-attn ----------
RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
        python -m pip install --no-cache-dir --no-build-isolation flash-attn; \
    fi

# ---------- Optional: transformer-engine ----------
RUN if [ "${INSTALL_TRANSFORMER_ENGINE}" = "1" ]; then \
        python -m pip install --no-cache-dir transformer-engine; \
    fi

# ---------- Copy remaining project files ----------
COPY env.sh ./
COPY examples/ ./examples/

# ---------- Environment variables (replicates env.sh for /app) ----------

# Core SODA paths
ENV SODA_ROOT=/app
ENV SODA_ENV_LOADED=1
ENV SODA_SRC=/app/src
ENV SODA_EXAMPLES=/app/examples
ENV SODA_OUTPUT=/app/output

# Microbenchmark directories
ENV MICROBENCH_DIR=/app/src/soda/microbench
ENV BAREMETAL_MICROBENCH_DIR=/app/src/soda/microbench/baremetal
ENV FRAMEWORK_DIR=/app/src/soda/microbench/framework
ENV PYTORCH_MICROBENCH_DIR=/app/src/soda/microbench/framework/pytorch

# Microbenchmark output subdirectories (relative — resolved against EXPERIMENT_DIR)
ENV BAREMETAL_OUTPUT_DIR=microbench/baremetal
ENV PYTORCH_OUTPUT_DIR=microbench/framework/pytorch

# Microbenchmark script directories
ENV BAREMETAL_SCRIPTS=/app/src/soda/microbench/baremetal/scripts
ENV PYTORCH_SCRIPTS=/app/src/soda/microbench/framework/pytorch/scripts

# Build directories
ENV BAREMETAL_BUILD=/app/src/soda/microbench/baremetal/build
ENV BAREMETAL_BINARY=/app/src/soda/microbench/baremetal/build/main_gemm_bm

# Virtual environment (not used in Docker, but set for compatibility)
ENV PYTHON_VENV=/app/.venv

# Environment metadata file
ENV ENV_METADATA=env_metadata.json

# Experiment directory (set by tracer at runtime)
ENV EXPERIMENT_DIR=""

# Microbench data paths (relative — resolved against EXPERIMENT_DIR at runtime)
ENV ALL_SEQUENCES=microbench/all_sequences.json
ENV ALL_KERNEL_SEQUENCES=microbench/all_kernel_sequences.json
ENV UNIQUE_ALL_SEQUENCES=microbench/unique_all_sequences.json
ENV ALL_GEMM_SEQUENCES=microbench/all_gemm_sequences.json
ENV UNIQUE_GEMM_SEQUENCES=microbench/unique_gemm_sequences.json
ENV PYTORCH_TRACES=microbench/framework/pytorch/traces
ENV PYTORCH_GEMM_SEQUENCES=microbench/framework/pytorch/output/pytorch_gemm_sequences.json
ENV PYTORCH_ALL_SEQUENCES=microbench/framework/pytorch/output/pytorch_all_sequences.json
ENV BAREMETAL_TRACES=microbench/baremetal/traces
ENV BAREMETAL_JOBS=microbench/baremetal/output/jobs.json
ENV BAREMETAL_GEMM_KERNELS=microbench/baremetal/output/baremetal_gemm_kernels.json
ENV TAX_BREAK_SUMMARY=microbench/taxbreak.json
ENV TAX_BREAK_PLOT=microbench/taxbreak_plot.png
ENV ASSERT_LOG=microbench/assert.log

# Enhanced TaxBreak pipeline outputs (informational — paths constructed by pipeline)
ENV KERNEL_DATABASE=kernel_database.json
ENV NCU_OUTPUT_DIR=taxbreak/ncu
ENV ENHANCED_TAXBREAK_SUMMARY=taxbreak/enhanced_taxbreak.json

# PyTorch memory allocator config
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

# HuggingFace — override at runtime via -e HF_HOME=... -e HF_TOKEN=...
ENV HF_HOME=/app/hf_cache

# Python path
ENV PYTHONPATH=/app/src

# ---------- Create output and cache directories ----------
RUN mkdir -p /app/output /app/hf_cache

# ---------- Entrypoint ----------
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["soda-cli", "--help"]
