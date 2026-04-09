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

CKPT="${CACHE}/checkpoints_${VARIANT}_owt/best.pt"
OUT="eval_results_${VARIANT}_owt"
mkdir -p "$OUT"

case "$VARIANT" in
  yurii)   SCRIPT=eval_run.py ;;
  tmm)     SCRIPT=tmm_eval_run.py ;;
  vanilla) SCRIPT=vanilla_eval_run.py ;;
  adam)    SCRIPT=adam_eval_run.py ;;
  adamw)   SCRIPT=adamw_eval_run.py ;;
  *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac

python "$SCRIPT" --checkpoint "$CKPT" --output-dir "$OUT"

echo "=== Done: $(date) ==="
