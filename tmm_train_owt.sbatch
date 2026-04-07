#!/bin/bash
#SBATCH --job-name=tmm-owt
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-16,babel-m9-20,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/tmm_owt_%j.out
#SBATCH --error=logs/tmm_owt_%j.err

mkdir -p logs
mkdir -p ${CACHE}/checkpoints_tmm_owt

cd "$SLURM_SUBMIT_DIR"

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo "Start:     $(date)"
echo "CACHE:     $CACHE"
echo "================"

source activate jspo
export PYTHONUNBUFFERED=1
export HF_HOME=${CACHE}/huggingface
export HF_DATASETS_CACHE=${CACHE}/huggingface/datasets

cleanup() {
    echo "=== Caught preemption signal at $(date) ==="
    kill -TERM "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID" 2>/dev/null
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap cleanup USR1

CKPT=${CACHE}/checkpoints_tmm_owt/best.pt
if [ -f "$CKPT" ]; then
    echo "Resuming from $CKPT"
    srun torchrun --nproc_per_node=2 tmm_train_owt.py --resume "$CKPT" &
else
    echo "Training from scratch"
    srun torchrun --nproc_per_node=2 tmm_train_owt.py &
fi
TRAIN_PID=$!

wait "$TRAIN_PID"
echo "=== Training finished: $(date) ==="
