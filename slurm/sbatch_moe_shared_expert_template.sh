#!/bin/bash
# =============================================================================
# SODA SBATCH Template: MoE CUPTI Memory Profiling Pipeline
# =============================================================================
# Standalone MoE per-operator memory profiling via PyTorch Profiler CUPTI
# hardware counters.  No dependency on kernel_database.json or the main SODA
# pipeline.
#
# Pipeline:
#   1. Load model + tokenizer via HuggingFace
#   2. Run curated benchmark prompts from soda.moe.prompts
#   3. Measure per-operator HBM (dram bytes) and L2 traffic via CUPTI
#   4. Classify operators using nn.Module hierarchy (deterministic)
#   5. Output per-prompt + aggregated op_profile.json
#
# Each prompt is tokenized at its natural length (no padding).  Prompts in
# soda.moe.prompts are length-stratified (_short ~50-80tok, base ~120-200tok,
# _long 300+tok) to test routing behavior across context lengths.
# --max-seq-len serves only as a safety truncation cap.
#
# Classification method:
#   nn.Module scope from torch.profiler(with_modules=True).  Regex patterns
#   match Qwen2-MoE, Mixtral, DeepSeek-V2 module naming deterministically.
#   No shape heuristics or kernel DB dependency.
#
# This template intentionally does NOT embed HF tokens or read token files.
# For gated models, authenticate before submit:
#   huggingface-cli login
#
# Usage:
#   cp slurm/sbatch_moe_shared_expert_template.sh slurm/my_moe_job.sh
#   # Edit USER CONFIGURATION section
#   sbatch slurm/my_moe_job.sh
# =============================================================================

#SBATCH --job-name=soda_moe_cupti
#SBATCH --output=soda_moe_%j.out
#SBATCH --error=soda_moe_%j.err
#SBATCH -t 0-06:00:00

# SBATCH -p partition_name_here
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=32G

set -euo pipefail

# =============================================================================
# USER CONFIGURATION — Edit these values for your setup
# =============================================================================

# Path to your SODA repository clone
SODA_PROJECT_ROOT="$HOME/SODA---System-Offload-Dynamics-Analyzer"

# Conda environment name (set to "" to skip conda activation)
CONDA_ENV=""

# CUDA toolkit module to load (set to "" to skip)
CUDA_MODULE="cuda12.8/toolkit/12.8.1"

# Nsight Compute module — provides the `ncu` binary for the NCU bridge
# (used when CUPTI returns zero bytes; set to "" to skip)
NSIGHT_MODULE="cuda12.8/nsight/12.8.1"

# Fail early when torch CUDA tag and loaded CUDA module major.minor mismatch.
# Example: torch 2.9.0+cu128 expects CUDA module 12.8.
ENFORCE_CUDA_TAG_MATCH=1

# Run a tiny CUPTI profiler smoke test before expensive model load.
# It verifies non-zero dram/lts counters on a CUDA matmul.
RUN_CUPTI_SMOKE_TEST=1

# Model config
MODEL="Qwen/Qwen1.5-MoE-A2.7B"
MAX_NEW_TOKENS=1
PRECISION="bfloat16"
WARMUP=2

# Safety truncation cap.  Prompts are tokenized at natural length;
# this only truncates prompts exceeding this limit.
MAX_SEQ_LEN=4096

# Prompt selection (leave both empty for default 7 representative prompts):
#   MOE_PROMPTS  — specific prompt names, space-separated
#   MOE_CATEGORIES — domain categories (code, science, legal, etc.)
# See soda.moe.prompts for available names and categories.
MOE_PROMPTS=""
MOE_CATEGORIES=""

# Output tag used by soda_make_output_root
OUTPUT_TAG="moe-cupti"

# Optional HuggingFace cache location
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"

# =============================================================================
# ENVIRONMENT SETUP — Usually no changes needed below
# =============================================================================

echo "============================================================================"
echo "SODA MoE CUPTI Memory Profiling Pipeline"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Config: model=$MODEL mt=$MAX_NEW_TOKENS p=$PRECISION warmup=$WARMUP"
echo "        max_seq_len=$MAX_SEQ_LEN (truncation cap only)"
if [ -n "$MOE_PROMPTS" ]; then
	echo "Prompts: $MOE_PROMPTS"
elif [ -n "$MOE_CATEGORIES" ]; then
	echo "Categories: $MOE_CATEGORIES"
else
	echo "Prompts: default (7 representative, one per domain)"
fi
echo "============================================================================"

if [ -n "$CUDA_MODULE" ]; then
	module load "$CUDA_MODULE" 2>/dev/null || true
fi

if [ -n "$NSIGHT_MODULE" ]; then
	module load "$NSIGHT_MODULE" 2>/dev/null || true
fi

# Verify ncu is available for the NCU bridge
which ncu >/dev/null 2>&1 && echo "[setup] ncu: $(which ncu) ($(ncu --version 2>/dev/null | head -1))" || echo "[setup] WARNING: ncu not found in PATH — NCU bridge will be skipped"

if [ -n "$CONDA_ENV" ]; then
	export PATH=~/miniconda3/bin:$PATH
	source ~/miniconda3/etc/profile.d/conda.sh
	conda activate "$CONDA_ENV"
fi

if [ -f "$SODA_PROJECT_ROOT/env.sh" ]; then
	source "$SODA_PROJECT_ROOT/env.sh"
else
	echo "Error: env.sh not found at $SODA_PROJECT_ROOT/env.sh"
	exit 1
fi

cd "$SODA_ROOT"
source "$SODA_ROOT/slurm/output_paths.sh"

export HF_HOME="$HF_HOME"
export CUDA_MODULE

pip install -e "$SODA_PROJECT_ROOT" --quiet 2>/dev/null || true

echo ""
echo "Environment:"
echo "  SODA_ROOT: $SODA_ROOT"
echo "  HF_HOME:   $HF_HOME"
python -c "
import torch
print(f'  PyTorch:   {torch.__version__}')
print(f'  CUDA:      {torch.cuda.is_available()}')
if torch.cuda.is_available():
	print(f'  GPU:       {torch.cuda.get_device_name(0)}')
"
echo ""

if [ "$ENFORCE_CUDA_TAG_MATCH" -eq 1 ]; then
	python - <<'PY'
import os
import re
import sys
import torch

cuda_module = os.environ.get("CUDA_MODULE", "")
torch_ver = torch.__version__

m_torch = re.search(r"\+cu(\d{2,3})", torch_ver)
m_mod = re.search(r"cuda(\d+)\.(\d+)", cuda_module)

if not m_torch or not m_mod:
	print("[preflight] CUDA tag check skipped (unable to parse torch/module versions)")
	raise SystemExit(0)

cu_tag = m_torch.group(1)
if len(cu_tag) == 3:
	torch_cuda = f"{int(cu_tag[:2])}.{int(cu_tag[2])}"
else:
	torch_cuda = f"{int(cu_tag[0])}.{int(cu_tag[1])}"

module_cuda = f"{int(m_mod.group(1))}.{int(m_mod.group(2))}"

if torch_cuda != module_cuda:
	print(
		"[preflight] ERROR: torch/CUDA mismatch: "
		f"torch={torch_ver} (expects CUDA {torch_cuda}), module={cuda_module}"
	)
	print("[preflight] Load a matching CUDA module or use a matching torch build.")
	raise SystemExit(2)

print(f"[preflight] torch/CUDA match OK: torch {torch_ver}, module {cuda_module}")
PY
fi

if [ "$RUN_CUPTI_SMOKE_TEST" -eq 1 ]; then
	python - <<'PY'
import torch

if not torch.cuda.is_available():
	print("[preflight] ERROR: CUDA unavailable; cannot run CUPTI smoke test")
	raise SystemExit(2)

from torch.profiler import profile, ProfilerActivity
try:
	from torch.profiler import _ExperimentalConfig
except Exception as exc:
	print(f"[preflight] ERROR: _ExperimentalConfig unavailable: {exc}")
	raise SystemExit(2)

from soda.moe.cupti_profiler import _extract_cupti_from_kernel, _is_cuda_device_type

exp = _ExperimentalConfig(
	profiler_metrics=[
		"dram__bytes_read.sum",
		"dram__bytes_write.sum",
		"lts__t_bytes.sum",
	],
	profiler_measure_per_kernel=True,
)

x = torch.randn((512, 512), device="cuda", dtype=torch.float16)
y = torch.randn((512, 512), device="cuda", dtype=torch.float16)

total_r = total_w = total_l2 = 0.0
cuda_events = 0
nonzero_metric_events = 0
sample = []
with profile(
	activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
	record_shapes=False,
	with_modules=False,
	experimental_config=exp,
) as prof:
	z = x @ y
	torch.cuda.synchronize()

for evt in prof.events():
	if not _is_cuda_device_type(getattr(evt, "device_type", None)):
		continue
	cuda_events += 1
	m = _extract_cupti_from_kernel(evt)
	r = float(m.get("hbm_read", 0.0) or 0.0)
	w = float(m.get("hbm_write", 0.0) or 0.0)
	l2 = float(m.get("l2_bytes", 0.0) or 0.0)
	total_r += r
	total_w += w
	total_l2 += l2
	if r > 0.0 or w > 0.0 or l2 > 0.0:
		nonzero_metric_events += 1
	if len(sample) < 5:
		sample.append({
			"name": getattr(evt, "name", ""),
			"has_cuda_metrics": getattr(evt, "cuda_metrics", None) is not None,
			"has_metrics": getattr(evt, "metrics", None) is not None,
		})

if (total_r + total_w) <= 0.0 and total_l2 <= 0.0:
	print("[preflight] ERROR: CUPTI smoke test found zero dram/lts counters")
	print(f"[preflight] CUDA events seen: {cuda_events}, nonzero-metric events: {nonzero_metric_events}")
	for i, s in enumerate(sample, 1):
		print(
			f"[preflight] sample[{i}] name={s['name']} "
			f"cuda_metrics={s['has_cuda_metrics']} metrics={s['has_metrics']}"
		)
	raise SystemExit(2)

print(
	"[preflight] CUPTI smoke test OK: "
	f"dram_read={total_r:.0f}, dram_write={total_w:.0f}, l2={total_l2:.0f}, "
	f"cuda_events={cuda_events}, nonzero_metric_events={nonzero_metric_events}"
)
PY
fi

OUTPUT_ROOT="$(soda_make_output_root "$OUTPUT_TAG")"
mkdir -p "$OUTPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"

# =============================================================================
# Run CUPTI profiling pipeline
# =============================================================================

echo ""
echo "============================================"
echo "MoE CUPTI Memory Profiling"
echo "============================================"

# Build the command
MOE_CMD="soda-cli --moe-profile"
MOE_CMD="$MOE_CMD --model $MODEL"
MOE_CMD="$MOE_CMD --output-dir $OUTPUT_ROOT"
MOE_CMD="$MOE_CMD --max-new-tokens $MAX_NEW_TOKENS"
MOE_CMD="$MOE_CMD --precision $PRECISION"
MOE_CMD="$MOE_CMD --moe-warmup $WARMUP"
MOE_CMD="$MOE_CMD --max-seq-len $MAX_SEQ_LEN"

if [ -n "$MOE_PROMPTS" ]; then
	MOE_CMD="$MOE_CMD --moe-prompts $MOE_PROMPTS"
fi

if [ -n "$MOE_CATEGORIES" ]; then
	MOE_CMD="$MOE_CMD --moe-categories $MOE_CATEGORIES"
fi

echo "Running: $MOE_CMD"
eval "$MOE_CMD" 2>&1 | tee "$OUTPUT_ROOT/moe_cupti.log"

echo ""
echo "============================================"
echo "Done"
echo "Finished: $(date)"
echo "============================================"

# Locate output directory
MODEL_SLUG=$(echo "$MODEL" | tr '/' '_')
CUPTI_DIR="$OUTPUT_ROOT/${MODEL_SLUG}/moe_cupti_profile"

echo ""
echo "Outputs:"
echo "  Aggregated op_profile: $CUPTI_DIR/op_profile.json"
echo "  Metadata:              $CUPTI_DIR/metadata.json"
echo "  Per-prompt profiles:   $CUPTI_DIR/per_prompt/"

if [ -f "$CUPTI_DIR/op_profile.json" ]; then
	python -c "
import json
d = json.load(open('$CUPTI_DIR/op_profile.json'))
shared = sum(1 for r in d if r.get('is_shared_expert'))
routed = sum(1 for r in d if r.get('expert_type') == 'routed_expert')
attn = sum(1 for r in d if r.get('expert_type') == 'attention')
gate = sum(1 for r in d if r.get('expert_type') == 'gate')
print(f'  Records: {len(d)} total')
print(f'    shared_expert:  {shared}')
print(f'    routed_expert:  {routed}')
print(f'    attention:      {attn}')
print(f'    gate:           {gate}')
print(f'    other:          {len(d) - shared - routed - attn - gate}')
"
fi

if [ -f "$CUPTI_DIR/metadata.json" ]; then
	python -c "
import json
m = json.load(open('$CUPTI_DIR/metadata.json'))
print(f'  CUPTI available: {m.get(\"cupti_available\", \"unknown\")}')
print(f'  Prompts profiled: {m.get(\"num_prompts\", 0)}')
for name, count in m.get('events_per_prompt', {}).items():
    print(f'    {name}: {count} events')
"
fi

echo ""
echo "Per-prompt profiles:"
if [ -d "$CUPTI_DIR/per_prompt" ]; then
	for f in "$CUPTI_DIR/per_prompt"/op_profile_*.json; do
		if [ -f "$f" ]; then
			count=$(python -c "import json; print(len(json.load(open('$f'))))")
			echo "  $(basename "$f"): $count records"
		fi
	done
fi
