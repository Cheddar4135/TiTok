#!/bin/bash
# ============================================================
# sft_7b_unav_v8_chronus_rl_rMsep3_uncha_batch4_natural
#   = run_rMsep3_natural.sh (sft_7b_unav_v8_chronus_rl_rMsep3_unpucha_batch4_natural) 재실험.
#   입력마커 = natural_text('second{XXX.Y}', 8토큰), 출력/GT = chronus('second{s}-second{e}. ...').
#   차이점:
#     - RL 데이터 : unpucha_chronus.json → uncha_v3_chronus.json (uncha_v3_local.json chronus 변환)
#     - GPU 2장(H100) → 1장(A100 80GB). eff batch = grad_accum(4) × GPU(1) = 4 → max_steps 5179 불변.
#     - val charades → noaudio 판 (서버에 charades 오디오 없음)
#   config: gdpo/config_sep3_uncha_natural_1gpu.yaml
#   재개: RESUME=<ckpt 경로> bash <this>
#   스모크: OUT=... RUN=... bash <this> --max_steps 3
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /workspace/setup.sh
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-train_blackwell}   # Blackwell(RTX PRO 6000) 서버용 env
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISABLE_TORCHCODEC=${DISABLE_TORCHCODEC:-1}   # torchcodec 끄고 decord 사용

SFT_CKPT=$CKPT_DIR/base/salmonn2p_7b_unav_v8_chronus
DATASET=$TRAIN_DIR/uncha_v3_chronus.json
CONFIG=gdpo/config_sep3_uncha_natural_1gpu.yaml

# ── 사전 확인 — 없으면 여기서 죽는다 (모델 로드 한참 뒤에 실패하지 않도록) ──
for p in "$SFT_CKPT/config.json" "$SFT_CKPT/model.safetensors.index.json" "$DATASET" "$CONFIG" \
         "$BASE_ROOT/data/val/unav100_tail_val100_chronus.json" \
         "$BASE_ROOT/data/val/charades_tail_val100_noaudio_chronus.json"; do
  [ -e "$p" ] || { echo "❌ 없음: $p"; exit 1; }
done
echo "✅ 모델/데이터 확인 완료"

RUN=${RUN:-sft_7b_unav_v8_chronus_rl_rMsep3_uncha_batch4_natural}
OUT=${OUT:-$CKPT_DIR/gdpo/$RUN}
mkdir -p "$OUT"
export WANDB_DIR="$OUT"

RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS=(--resume_from_checkpoint "$RESUME")

torchrun --standalone --nproc_per_node=1 --master_port=29523 \
  gdpo/gdpo_trainer_batch_natural.py \
  --config       "$CONFIG" \
  --model_path   "$SFT_CKPT" \
  --model_base   "$SFT_CKPT" \
  --dataset_path "$DATASET" \
  --reward_module reward_functions_rM_sep3_natural \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  "${RESUME_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$OUT/train.log"
