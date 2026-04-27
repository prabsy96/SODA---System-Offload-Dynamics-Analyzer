#!/bin/bash
# verify_parallelism.sh

source env.sh
activate_env

MODEL="gpt2"  # Using gpt2 as it's small and quick to load

echo "=== Testing TP (Default) ==="
soda-cli --model $MODEL --num-gpus 2 --parallelism tp --batch-size 1 --seq-len 128 --output-dir output/test_tp

echo "=== Testing DP ==="
soda-cli --model $MODEL --num-gpus 2 --parallelism dp --batch-size 1 --seq-len 128 --output-dir output/test_dp

echo "=== Testing FSDP ==="
soda-cli --model $MODEL --num-gpus 2 --parallelism fsdp --batch-size 1 --seq-len 128 --output-dir output/test_fsdp

echo "=== Testing EP (on an MoE model) ==="
# Using a small MoE model if possible, or just gpt2 to check it doesn't crash
soda-cli --model $MODEL --num-gpus 2 --parallelism ep --batch-size 1 --seq-len 128 --output-dir output/test_ep
