#!/bin/bash
#SBATCH --job-name=hybrid-eval
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-t9-24,babel-w5-28
#SBATCH --output=logs/hybrid_eval_%x_%j.out
#SBATCH --error=logs/hybrid_eval_%x_%j.err

# Submit per variant via (cutoff mode):
#   sbatch -J hybrid-yurii6-owt --export=ALL,BASE=yurii,CUTOFF=6,CKPT=$CACHE/checkpoints_yurii_owt/best.pt,OUT=eval_results_hybrid_yurii6_owt hybrid_eval.sbatch
# Or (sandwich mode — layer set overrides CUTOFF):
#   sbatch -J hybrid-yurii-0-11-owt  --export=ALL,BASE=yurii,NLAYERS=0,11,CKPT=$CACHE/checkpoints_yurii_owt/best.pt,OUT=eval_results_hybrid_yurii_0_11_owt  hybrid_eval.sbatch

set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

echo "=== hybrid-eval  base=$BASE  cutoff=${CUTOFF:-<unset>}  nlayers=${NLAYERS:-<unset>}  job=$SLURM_JOB_ID  node=$SLURM_NODELIST  $(date) ==="

source activate jspo
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export HF_HOME=${CACHE}/huggingface
export HF_DATASETS_CACHE=${CACHE}/huggingface/datasets

mkdir -p "$OUT"

EXTRA_ARGS=""
if [ -n "${NLAYERS:-}" ]; then
    EXTRA_ARGS="--nesterov-layers ${NLAYERS}"
fi

python hybrid_eval_run.py \
    --base-model "$BASE" \
    --cutoff-layer "${CUTOFF:-6}" \
    $EXTRA_ARGS \
    --checkpoint "$CKPT" \
    --output-dir "$OUT"

echo "=== Done: $(date) ==="
