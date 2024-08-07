#BSUB -J anurag_torch_run          # Job name
#BSUB -n 2                        # Number of cores
#BSUB -P acc_rg_HPIMS             # Project name
#BSUB -q gpu                      # Queue name
#BSUB -R "rusage[mem=5000]"       # Memory requirement
#BSUB -R "h10080g"                # GPU resource requirement
#BSUB -gpu "num=2"                # Number of GPUs
#BSUB -o anurag_torch_run.out     # Standard output file
#BSUB -e anurag_torch_run.err     # Standard error file
#BSUB -W 30:00                           # Time limit (12 hours)

# Load the necessary modules
module load python/3.10.4

# Set the PYTHONPATH environment variable
export PYTHONPATH="/sc/arion/work/patila06/VirtualEnvs/nanogpt_venv_3104/lib/python3.10/site-packages:$PYTHONPATH"

# Activate your Python virtual environment
source /sc/arion/work/patila06/VirtualEnvs/nanogpt_venv_3104/bin/activate

# Change directory to build-nanogpt
cd /sc/arion/work/patila06/Projects/build-nanogpt/

# Run your Python script
# python your_script.py
# or, for PyTorch with distributed training
torchrun --nproc-per-node=2 build_nanogpt/train_gpt2.py
