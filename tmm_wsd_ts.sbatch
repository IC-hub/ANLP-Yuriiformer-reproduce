#!/bin/bash
#SBATCH --job-name=tmm-wsd-ts
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-16,babel-m9-20,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-p5-24,babel-x5-16,babel-t9-24,babel-w5-28,babel-w9-16,babel-o9-16
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/tmm_wsd_ts_%j.out
#SBATCH --error=logs/tmm_wsd_ts_%j.err

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
mkdir -p ${CACHE}/checkpoints_tmm_wsd_ts

echo "=== Job: $SLURM_JOB_ID  mode=wsd  model=TMM  node=$SLURM_NODELIST  $(date) ==="

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

CKPT=${CACHE}/checkpoints_tmm_wsd_ts/best.pt
if [ -f "$CKPT" ]; then
    echo "Resuming from $CKPT"
    srun torchrun --nproc_per_node=2 tmm_train_flat.py --mode wsd --resume "$CKPT" &
else
    echo "Training from scratch"
    srun torchrun --nproc_per_node=2 tmm_train_flat.py --mode wsd &
fi
TRAIN_PID=$!
wait "$TRAIN_PID"
echo "=== Done: $(date) ==="
