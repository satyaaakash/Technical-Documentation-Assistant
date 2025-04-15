#!/bin/bash
#SBATCH --job-name=tech_doc_test
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --account=carpena

# Load required modules
module purge
module load python
module load cuda/12.2.2

# Print environment info
echo "== Environment Information =="
echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
which nvidia-smi && nvidia-smi || echo "nvidia-smi not found"

# Activate virtual environment
source $HOME/tech-documentation-assistant/venv/bin/activate

# Run test script
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Torch version:', torch.__version__); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
