#!/bin/bash
#SBATCH --job-name=simple_ui
#SBATCH --output=logs/simple_ui_%j.out
#SBATCH --error=logs/simple_ui_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --account=carpena

# Load required modules
module purge
module load python
module load cuda/12.2.2

# Activate virtual environment
source $SLURM_SUBMIT_DIR/venv/bin/activate

# Print environment info
echo "=== Environment Information ==="
echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run the UI
python src/simple_ui.py
