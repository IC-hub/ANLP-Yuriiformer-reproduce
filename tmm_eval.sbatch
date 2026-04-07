#!/bin/bash
#SBATCH --job-name=tmm-eval
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-16,babel-m9-20,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28
#SBATCH --output=logs/tmm_eval_%j.out
#SBATCH --error=logs/tmm_eval_%j.err

mkdir -p logs eval_results_tmm

cd "$SLURM_SUBMIT_DIR"

echo "=== TMMFormer Evaluation ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"

source activate jspo
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_FLASH_ATTENTION=1

pip uninstall flash_attn vllm -y 2>/dev/null || true

python tmm_eval_run.py --checkpoint checkpoints_tmm/best.pt --output-dir eval_results_tmm

echo "=== Evaluation finished: $(date) ==="
