#!/bin/bash
#SBATCH --job-name=attn-ent
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-t9-24,babel-w5-28
#SBATCH --output=logs/attn_entropy_%x_%j.out
#SBATCH --error=logs/attn_entropy_%x_%j.err

# Submit per variant via:
#   sbatch -J attn-vanilla --nodelist=babel-t9-24 --export=ALL,VARIANT=vanilla attention_entropy.sbatch
#   sbatch -J attn-yurii   --nodelist=babel-u5-28 --export=ALL,VARIANT=yurii   attention_entropy.sbatch
#   sbatch -J attn-tmm     --nodelist=babel-p9-28 --export=ALL,VARIANT=tmm     attention_entropy.sbatch
#   sbatch -J attn-adam    --nodelist=babel-w5-20 --export=ALL,VARIANT=adam    attention_entropy.sbatch
#   sbatch -J attn-adamw   --nodelist=babel-w5-16 --export=ALL,VARIANT=adamw   attention_entropy.sbatch

set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

echo "=== Job Info ==="
echo "Job:    $SLURM_JOB_ID  variant=$VARIANT"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "================"

source activate jspo
export PYTHONUNBUFFERED=1
export HF_HOME=${CACHE}/huggingface
export HF_DATASETS_CACHE=${CACHE}/huggingface/datasets

python attention_entropy.py --variant "$VARIANT"

echo "=== Done: $(date) ==="
