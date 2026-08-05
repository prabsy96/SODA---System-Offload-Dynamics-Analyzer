import os
import glob
import re

config_dir = '/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/configs/exp1_batch_scaling'
llama_files = glob.glob(os.path.join(config_dir, 'llama3_bs_*.yaml'))

for file_path in llama_files:
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace hidden_size: 4096 -> 2048
    content = re.sub(r'hidden_size:\s*4096', 'hidden_size: 2048', content)
    # Replace filter_size: 14336 -> 8192
    content = re.sub(r'filter_size:\s*14336', 'filter_size: 8192', content)
    # Replace num_hidden_layers: 2 -> 16
    content = re.sub(r'num_hidden_layers:\s*2', 'num_hidden_layers: 16', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"Updated {os.path.basename(file_path)} to exact Llama-3.2-1B parameters.")
