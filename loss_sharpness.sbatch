#!/bin/bash
#SBATCH --job-name=sharp
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-t9-24,babel-w5-28
#SBATCH --output=logs/sharp_%x_%j.out
#SBATCH --error=logs/sharp_%x_%j.err

set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

echo "=== Job: $SLURM_JOB_ID  variant=$VARIANT  node=$SLURM_NODELIST  $(date) ==="

source activate jspo
export PYTHONUNBUFFERED=1
export HF_HOME=${CACHE}/huggingface
export HF_DATASETS_CACHE=${CACHE}/huggingface/datasets

python loss_sharpness.py --variant "$VARIANT"

echo "=== Done: $(date) ==="
