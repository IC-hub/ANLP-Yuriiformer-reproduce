#!/bin/bash
#SBATCH --job-name=hf-upload-ckpts
#SBATCH --partition=debug
#SBATCH --time=11:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/hf_upload_%j.out
#SBATCH --error=logs/hf_upload_%j.err

set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

echo "=== HF upload checkpoints  job=$SLURM_JOB_ID  node=$SLURM_NODELIST  $(date) ==="
source activate jspo
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=0

python hf_upload_checkpoints.py
echo "=== Done: $(date) ==="
