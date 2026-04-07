#!/bin/bash
#SBATCH --job-name=adamwformer-4gpu
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/adamwformer_4gpu_%j.out
#SBATCH --error=logs/adamwformer_4gpu_%j.err

mkdir -p logs checkpoints_adamw

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
if [ -f checkpoints_adamw/best.pt ]; then
    echo "Resuming from checkpoints_adamw/best.pt"
    torchrun --nproc_per_node=4 adamw_train_ddp.py --resume checkpoints_adamw/best.pt
else
    echo "Training from scratch"
    torchrun --nproc_per_node=4 adamw_train_ddp.py
fi

echo "=== Training finished: $(date) ==="
