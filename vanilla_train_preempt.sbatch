#!/bin/bash
#SBATCH --job-name=vanilla-2gpu
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-16,babel-m9-20,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/vanilla_debug_%j.out
#SBATCH --error=logs/vanilla_debug_%j.err

mkdir -p logs checkpoints_vanilla

cd "$SLURM_SUBMIT_DIR"

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo "Start:     $(date)"
echo "================"

source activate jspo
export PYTHONUNBUFFERED=1

cleanup() {
    echo "=== Caught preemption signal at $(date) ==="
    kill -TERM "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID" 2>/dev/null
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap cleanup USR1

if [ -f checkpoints_vanilla/best.pt ]; then
    echo "Resuming from checkpoints_vanilla/best.pt"
    srun torchrun --nproc_per_node=2 vanilla_train_ddp.py --resume checkpoints_vanilla/best.pt &
else
    echo "Training from scratch"
    srun torchrun --nproc_per_node=2 vanilla_train_ddp.py &
fi
TRAIN_PID=$!

wait "$TRAIN_PID"
echo "=== Training finished: $(date) ==="
