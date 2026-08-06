import os
import shutil

base_dir = "/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/model_dir/soda_output"

for bs in [1, 2]:
    target_dir = os.path.join(base_dir, f"exp1_llama3_bs_{bs}")
    if os.path.exists(target_dir):
        print(f"Cleaning stale checkpoint directory: {target_dir}")
        shutil.rmtree(target_dir)
