import os
import glob
import re

config_dir = '/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/configs/exp1_batch_scaling'
llama_files = glob.glob(os.path.join(config_dir, 'llama3_bs_*.yaml'))

for file_path in llama_files:
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Set rotary_dim: 64 for Llama-3.2-1B (since head_dim = 2048 // 32 = 64)
    content = re.sub(r'rotary_dim:\s*128', 'rotary_dim: 64', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"Fixed rotary_dim: 64 in {os.path.basename(file_path)}")
