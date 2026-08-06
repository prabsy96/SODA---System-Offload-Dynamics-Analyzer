import os
import glob
import re

config_dir = '/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/configs/exp1_batch_scaling'
yaml_files = glob.glob(os.path.join(config_dir, '*.yaml'))

for file_path in yaml_files:
    # Extract batch size from filename: e.g., llama3_bs_2.yaml -> 2
    match = re.search(r'_bs_(\d+)\.yaml', file_path)
    if not match:
        continue
    batch_size = match.group(1)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace micro_batch_size: 16 with micro_batch_size: {batch_size}
    # Be careful to only replace it if it exists
    new_content = re.sub(
        r'csx\.performance\.micro_batch_size:\s*\d+',
        f'csx.performance.micro_batch_size: {batch_size}',
        content
    )
    
    # Write back if changed
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Updated {file_path} with micro_batch_size: {batch_size}")
