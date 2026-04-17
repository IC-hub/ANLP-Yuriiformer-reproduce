#!/bin/bash
#SBATCH --job-name=tmm-owt-downstream
#SBATCH --partition=preempt
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40S:1
#SBATCH --output=logs/tmm_%x_%j.out
#SBATCH --error=logs/tmm_%x_%j.err

# Usage: sbatch --export=ALL,VARIANT=wsd tmm_owt_eval_downstream.sbatch
set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

echo "=== TMM-${VARIANT}-OWT downstream  job=$SLURM_JOB_ID  node=$SLURM_NODELIST  $(date) ==="
source activate jspo
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export HF_HOME=${CACHE}/huggingface
export HF_DATASETS_CACHE=${CACHE}/huggingface/datasets

CKPT=${CACHE}/checkpoints_tmm_${VARIANT}_owt/best.pt
OUT_DIR=eval_results_tmm_${VARIANT}_owt
mkdir -p "$OUT_DIR"
python tmm_eval_run.py --checkpoint "$CKPT" --output-dir "$OUT_DIR"
echo "=== Done: $(date) ==="
