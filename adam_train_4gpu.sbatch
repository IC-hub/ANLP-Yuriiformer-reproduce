#!/bin/bash
#SBATCH --job-name=adamformer-4gpu
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-16,babel-m9-20,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28
#SBATCH --output=logs/adamformer_4gpu_%j.out
#SBATCH --error=logs/adamformer_4gpu_%j.err

mkdir -p logs checkpoints_adam

cd "$SLURM_SUBMIT_DIR"

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo "Start:     $(date)"
echo "================"

source activate jspo
export PYTHONUNBUFFERED=1

# Resume from checkpoint if available, otherwise train from scratch
if [ -f checkpoints_adam/best.pt ]; then
    echo "Resuming from checkpoints_adam/best.pt"
    torchrun --nproc_per_node=4 adam_train_ddp.py --resume checkpoints_adam/best.pt
else
    echo "Training from scratch"
    torchrun --nproc_per_node=4 adam_train_ddp.py
fi

echo "=== Training finished: $(date) ==="
