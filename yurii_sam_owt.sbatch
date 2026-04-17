#!/bin/bash
#SBATCH --job-name=yurii-sam
#SBATCH --partition=general
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-16,babel-m9-20,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-p5-24,babel-x5-16
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/yurii_sam_owt_%j.out
#SBATCH --error=logs/yurii_sam_owt_%j.err

# Yurii + cosine + full SAM (~2x compute vs baseline; reserve 4 days).

mkdir -p logs
mkdir -p ${CACHE}/checkpoints_yurii_sam_owt
cd "$SLURM_SUBMIT_DIR"

echo "=== Job: $SLURM_JOB_ID  mode=sam  node=$SLURM_NODELIST  $(date) ==="

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

CKPT=${CACHE}/checkpoints_yurii_sam_owt/best.pt
if [ -f "$CKPT" ]; then
    echo "Resuming from $CKPT"
    srun torchrun --nproc_per_node=2 yurii_train_owt_flat.py --mode sam --resume "$CKPT" &
else
    echo "Training from scratch"
    srun torchrun --nproc_per_node=2 yurii_train_owt_flat.py --mode sam &
fi
TRAIN_PID=$!
wait "$TRAIN_PID"
echo "=== Done: $(date) ==="
