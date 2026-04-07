#!/bin/bash
#SBATCH --job-name=adamformer-2gpu
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/adamformer_2gpu_%j.out
#SBATCH --error=logs/adamformer_2gpu_%j.err

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
torchrun --nproc_per_node=2 adam_train_ddp.py

echo "=== Training finished: $(date) ==="
