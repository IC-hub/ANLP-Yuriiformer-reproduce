#!/bin/bash
#SBATCH --job-name=tmm-ts-eval-all
#SBATCH --partition=preempt
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40S:1
#SBATCH --output=logs/tmm_ts_eval_all_%j.out
#SBATCH --error=logs/tmm_ts_eval_all_%j.err

# Evaluate completed TMM TS variants: downstream tasks, attention entropy, sharpness
# Usage: sbatch --export=ALL,VARIANT=wsd tmm_ts_eval_all.sbatch
#   VARIANT = wsd | sam | sawd

set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

echo "=== TMM-${VARIANT}-TS full eval  job=$SLURM_JOB_ID  node=$SLURM_NODELIST  $(date) ==="

source activate jspo
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export HF_HOME=${CACHE}/huggingface
export HF_DATASETS_CACHE=${CACHE}/huggingface/datasets

CKPT=${CACHE}/checkpoints_tmm_${VARIANT}_ts/best.pt
OUT_DIR=eval_results_tmm_${VARIANT}_ts

echo "=== Checkpoint: $CKPT ==="
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

# 1) Downstream eval (HellaSwag + ARC-Easy)
echo ""
echo "=== [1/3] Downstream evaluation ==="
mkdir -p "$OUT_DIR"
python tmm_eval_run.py --checkpoint "$CKPT" --output-dir "$OUT_DIR"

# 2) Attention entropy
echo ""
echo "=== [2/3] Attention entropy ==="
python attention_entropy.py --variant "ts-tmm-${VARIANT}"

# 3) Loss-landscape sharpness
echo ""
echo "=== [3/3] Loss-landscape sharpness ==="
python loss_sharpness.py --variant "ts-tmm-${VARIANT}"

echo ""
echo "=== All evaluations done: $(date) ==="
