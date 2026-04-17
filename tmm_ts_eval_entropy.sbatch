#!/bin/bash
#SBATCH --job-name=tmm-ts-entropy
#SBATCH --partition=preempt
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40S:1
#SBATCH --output=logs/tmm_%x_%j.out
#SBATCH --error=logs/tmm_%x_%j.err

# Usage: sbatch --export=ALL,VARIANT=wsd tmm_ts_eval_entropy.sbatch
set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

echo "=== TMM-${VARIANT}-TS entropy  job=$SLURM_JOB_ID  node=$SLURM_NODELIST  $(date) ==="
source activate jspo
export PYTHONUNBUFFERED=1
export HF_HOME=${CACHE}/huggingface
export HF_DATASETS_CACHE=${CACHE}/huggingface/datasets

python attention_entropy.py --variant "ts-tmm-${VARIANT}"
echo "=== Done: $(date) ==="
