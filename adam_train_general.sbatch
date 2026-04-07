#!/bin/bash
#SBATCH --job-name=adamformer
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/adamformer_train_%j.out
#SBATCH --error=logs/adamformer_train_%j.err

mkdir -p logs checkpoints_adam

cd "$SLURM_SUBMIT_DIR"

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:     $(date)"
echo "================"

source activate jspo
python adam_train.py

echo "=== Training finished: $(date) ==="
