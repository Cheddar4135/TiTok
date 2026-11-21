#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling
#   = run_rMsep3_noscaling.sh (sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling) 와 동일 설정.
#   차이점: RL 학습 데이터 unpucha_v2.json → uncha_v3.json, GPU 2장 → 1장(A100 80GB).
#   ⚠️ eff batch = grad_accum(4) × GPU(1) = 4 (config_sep3_uncha_1gpu.yaml) → max_steps 5179 불변.
#   ⚠️ 신규 run — SFT(unav_v8) 에서 fresh 시작. 재개는 RESUME=True bash <this>.
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/team404/workspace/setup.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# wandb 키 미설정이면 offline 으로 기록 (나중에 `wandb sync $OUT/wandb/offline-run-*`)
if [[ -z "$WANDB_API_KEY" || "$WANDB_API_KEY" == "<"* ]]; then
  echo "⚠️ WANDB_API_KEY 미설정 → WANDB_MODE=offline"
  unset WANDB_API_KEY
  export WANDB_MODE=${WANDB_MODE:-offline}
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SFT_CKPT=$CKPT_DIR/base/salmonn2p_7b_unav_v8
DATASET=$TRAIN_DIR/uncha_v3.json
CONFIG=gdpo/config_sep3_uncha_1gpu.yaml

# ── 사전 확인 — 없으면 여기서 죽는다 (모델 로드 한참 뒤에 실패하지 않도록) ──
for p in "$SFT_CKPT/config.json" "$DATASET" "$CONFIG" \
         "$BASE_ROOT/data/val/unav100_tail_val100.json" \
         "$BASE_ROOT/data/val/charades_tail_val100_noaudio.json"; do
  [ -e "$p" ] || { echo "❌ 없음: $p"; exit 1; }
done
echo "✅ 모델/데이터 확인 완료"

RUN=${RUN:-sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling}
OUT=${OUT:-$CKPT_DIR/gdpo/$RUN}
mkdir -p "$OUT"
export WANDB_DIR="$OUT"

# 재개: RESUME=True (최신 ckpt 자동) 또는 RESUME=<ckpt 경로>
RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS=(--resume_from_checkpoint "$RESUME")

torchrun --standalone --nproc_per_node=1 --master_port=29522 \
  gdpo/gdpo_trainer_batch.py \
  --config       "$CONFIG" \
  --model_path   "$SFT_CKPT" \
  --model_base   "$SFT_CKPT" \
  --dataset_path "$DATASET" \
  --reward_module reward_functions_rM_sep3 \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  "${RESUME_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$OUT/train.log"
