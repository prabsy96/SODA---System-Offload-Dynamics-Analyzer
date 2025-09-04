#!/bin/bash
# see https://slurm.schedmd.com/sbatch.html for options

# Jobname
#SBATCH -J run_tool # Updated job name for clarity
# Output and error files
#SBATCH -o run_tool.out
#SBATCH -e run_tool.err

# Job time limit (currently 15mins) - WARNING: This is very short for a training job.
# You may need to increase this significantly (e.g., #SBATCH -t 1-02:00:00 for 1 day 2 hours).
#SBATCH -t 00:15

# Node configuration
#SBATCH --nodes 1                 # Request 1 full node

# CPU configuration
#SBATCH --ntasks 1                # Run a single task (your main script process)
#SBATCH --cpus-per-task=96        # Request all 96 CPU cores on the node (2x48-core CPUs)
                                  # for your single task. Your script/application
                                  # needs to be able to utilize these cores.

# Memory configuration
#SBATCH --mem=512G                # Request all 512GB of system RAM on the node.
                                  # Ensure your application can manage/utilize this amount,
                                  # and be mindful of potential OS overhead if issues arise.
                                  # (Alternatively, --mem=0 with --exclusive might be used
                                  # on some systems to request all available memory).

# GPU allocation
#SBATCH --gres=gpu:H100:1         # Request 1 H100 GPU.

# --- Execution Block ---

# Load necessary modules
 echo "Loading modules..."
 module load nvhpc/24.9 cuda12.2/nsight/12.2.2 cuda12.2/blas/12.2.2 cuda12.2/fft/12.2.2 cuda12.2/toolkit/12.2.2 cuda12.2/profiler/12.2.2 gcc11/11.3.0
 module list # Optional: lists loaded modules for debugging

# Activate your environment
 echo "Activating environment..."
 # source ~/Samsung/Tool/env/bin/activate # Ensure this path is correct relative to your job's working directory or use an absolute path.

# Run your script
cd ../
echo "Running script..."
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the root directory of the SODA repository."
    exit 1
fi
python -m soda.main \
  --model gpt2 \
  --output-dir ./soda_results \
  --batch_size 1 \
  --seq_len 128 \
  --fusion 2 3 \
  --prox_score 1.0
echo "Job finished."
