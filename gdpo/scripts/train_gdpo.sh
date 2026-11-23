#!/bin/bash
# ============================================================
# TiTok stage 2: GDPO training (paper setting, 1x A100 80GB).
#
#   bash gdpo/scripts/train_gdpo.sh
#   RESUME=True bash gdpo/scripts/train_gdpo.sh            # resume latest ckpt
#   RESUME=<ckpt dir> bash gdpo/scripts/train_gdpo.sh      # resume a given ckpt
#
# Starts from the merged SFT model (checkpoints/base/salmonn2p_7b_unav_v8).
# The paper reports checkpoint-2000. Extra arguments are forwarded to the trainer.
# ============================================================
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/env.sh"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-29522}"

SFT_CKPT="${SFT_CKPT:-$CKPT_DIR/base/salmonn2p_7b_unav_v8}"
DATASET="${DATASET:-$DATA_DIR/gdpo/train.json}"
CONFIG="${CONFIG:-gdpo/configs/titok_gdpo.yaml}"
RUN="${RUN:-titok_gdpo}"
OUT="${OUT:-$CKPT_DIR/gdpo/$RUN}"

# Fail fast instead of after the model has been loaded.
for p in "$SFT_CKPT/config.json" "$DATASET" "$CONFIG" \
         "$DATA_DIR/gdpo/val_unav100.json" "$DATA_DIR/gdpo/val_charades.json"; do
    [ -e "$p" ] || { echo "[error] not found: $p" >&2; exit 1; }
done

mkdir -p "$OUT"
export WANDB_DIR="$OUT"

RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS=(--resume_from_checkpoint "$RESUME")

torchrun --standalone --nproc_per_node=1 --master_port="$MASTER_PORT" \
    gdpo/gdpo_trainer.py \
    --config        "$CONFIG" \
    --model_path    "$SFT_CKPT" \
    --model_base    "$SFT_CKPT" \
    --dataset_path  "$DATASET" \
    --reward_module rewards \
    --tti_mode on \
    --output_dir    "$OUT" \
    --run_name      "$RUN" \
    "${RESUME_ARGS[@]}" \
    "$@" \
    2>&1 | tee -a "$OUT/train.log"
