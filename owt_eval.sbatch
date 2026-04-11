#!/bin/bash
#SBATCH --job-name=owt-eval
#SBATCH --partition=general
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-t9-24
#SBATCH --output=logs/owt_eval_%x_%j.out
#SBATCH --error=logs/owt_eval_%x_%j.err

# Submit per variant via:
#   sbatch -J owt-eval-<v> --export=ALL,VARIANT=<v> owt_eval.sbatch
# where <v> in {vanilla, yurii, tmm, adam, adamw}

set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

echo "=== owt-eval $VARIANT  job=$SLURM_JOB_ID  node=$SLURM_NODELIST  $(date) ==="

source activate jspo
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export HF_HOME=${CACHE}/huggingface
export HF_DATASETS_CACHE=${CACHE}/huggingface/datasets

case "$VARIANT" in
  yurii)   SCRIPT=eval_run.py;        CKPT="${CACHE}/checkpoints_yurii_owt/best.pt" ;;
  tmm)     SCRIPT=tmm_eval_run.py;    CKPT="${CACHE}/checkpoints_tmm_owt/best.pt" ;;
  vanilla) SCRIPT=vanilla_eval_run.py;CKPT="${CACHE}/checkpoints_vanilla_owt/best.pt" ;;
  adam)    SCRIPT=adam_eval_run.py;   CKPT="${CACHE}/checkpoints_adam_owt/best.pt" ;;
  adamw)   SCRIPT=adamw_eval_run.py;  CKPT="${CACHE}/checkpoints_adamw_owt/best.pt" ;;
  wsd)     SCRIPT=eval_run.py;        CKPT="${CACHE}/checkpoints_yurii_wsd_owt/best.pt" ;;
  sam)     SCRIPT=eval_run.py;        CKPT="${CACHE}/checkpoints_yurii_sam_owt/best.pt" ;;
  sawd)    SCRIPT=eval_run.py;        CKPT="${CACHE}/checkpoints_yurii_sawd_owt/best.pt" ;;
  *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac

OUT="eval_results_${VARIANT}_owt"
mkdir -p "$OUT"

python "$SCRIPT" --checkpoint "$CKPT" --output-dir "$OUT"

echo "=== Done: $(date) ==="
