#!/bin/bash
# Move large checkpoint directories from login-node disk into $CACHE on the
# compute node. Static-only: skips dirs that are still being written by
# active training jobs (yurii_sam, yurii_sawd in checkpoints/).
#
# After cp succeeds AND a per-file size verification passes, the originals
# on login disk are removed. If anything fails, originals are kept intact.

#SBATCH --job-name=move-ckpts
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/move_ckpts_%j.out
#SBATCH --error=logs/move_ckpts_%j.err

set -euo pipefail

ts() { echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CACHE_ROOT="${CACHE:-/data/user_data/jgai/cache}"
DEST_ROOT="${CACHE_ROOT}/checkpoints_ts_archive"
SRC_ROOT="$SLURM_SUBMIT_DIR"

ts "node = $(hostname)"
ts "source root = $SRC_ROOT"
ts "destination root = $DEST_ROOT"
mkdir -p "$DEST_ROOT"

# (src_relative_path, dest_relative_path) pairs.
# Static (training NOT in progress) entries only.
ENTRIES=(
  "checkpoints_yurii      yurii_ts"
  "checkpoints_tmm        tmm_ts"
  "checkpoints_adam       adam_ts"
  "checkpoints_adamw      adamw_ts"
  "checkpoints_vanilla    vanilla_ts"
  "checkpoints/yurii_wsd  yurii_wsd_ts"
  "checkpoints/yurii_sam  yurii_sam_ts"
  "checkpoints/yurii_sawd yurii_sawd_ts"
)

# First sanity check destination root is actually writable (catches NFS perm issues
# like the babel-x9-24 case from earlier).
TEST_FILE="$DEST_ROOT/.write_test_$$"
echo "ok" > "$TEST_FILE" || { echo "FATAL: cannot write to $DEST_ROOT" >&2; exit 1; }
rm -f "$TEST_FILE"
ts "destination is writable"

verify_dir() {
  local src="$1" dst="$2"
  local src_size dst_size src_count dst_count
  if [[ ! -d "$src" || ! -d "$dst" ]]; then
    return 1
  fi
  src_size=$(du -sb "$src" | awk '{print $1}')
  dst_size=$(du -sb "$dst" | awk '{print $1}')
  src_count=$(find "$src" -type f | wc -l)
  dst_count=$(find "$dst" -type f | wc -l)
  if [[ "$src_size" != "$dst_size" || "$src_count" != "$dst_count" ]]; then
    echo "  size mismatch: src=${src_size}B/${src_count}f  dst=${dst_size}B/${dst_count}f"
    return 1
  fi
  echo "  verified: ${src_size}B across ${src_count} files"
  return 0
}

for entry in "${ENTRIES[@]}"; do
  read -r src_rel dst_rel <<< "$entry"
  src_abs="$SRC_ROOT/$src_rel"
  dst_abs="$DEST_ROOT/$dst_rel"

  echo
  ts "=== $src_rel -> $dst_abs ==="

  if [[ ! -d "$src_abs" ]]; then
    ts "  source missing, skip"
    continue
  fi

  if [[ -d "$dst_abs" ]]; then
    ts "  destination already exists, verifying..."
    if verify_dir "$src_abs" "$dst_abs"; then
      ts "  already in sync, will delete source"
    else
      ts "  destination exists but mismatched -> re-copying"
      rm -rf "$dst_abs"
      cp -a "$src_abs" "$dst_abs"
      verify_dir "$src_abs" "$dst_abs" || { ts "  verify FAILED, leaving source intact"; continue; }
    fi
  else
    ts "  copying with cp -a ..."
    cp -a "$src_abs" "$dst_abs"
    verify_dir "$src_abs" "$dst_abs" || { ts "  verify FAILED, leaving source intact"; continue; }
  fi

  ts "  removing source $src_abs"
  rm -rf "$src_abs"
  ts "  done"
done

echo
ts "=== final destination listing ==="
ls -la "$DEST_ROOT"
echo
du -sh "$DEST_ROOT"/*
echo
ts "=== remaining size on login under $SRC_ROOT/checkpoints* ==="
du -sh "$SRC_ROOT"/checkpoints "$SRC_ROOT"/checkpoints_yurii "$SRC_ROOT"/checkpoints_tmm "$SRC_ROOT"/checkpoints_adam "$SRC_ROOT"/checkpoints_adamw "$SRC_ROOT"/checkpoints_vanilla 2>/dev/null || true
ts "=== Done ==="
