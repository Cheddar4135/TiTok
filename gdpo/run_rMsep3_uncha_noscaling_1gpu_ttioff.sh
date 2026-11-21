#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling_ttioff   (w/o TTI ablation)
#   = run_rMsep3_uncha_noscaling_1gpu.sh (sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling) 와 동일.
#   유일한 차이: **tti_mode on → off**
#     --tti_mode off → model.config.time_token_id_range=None (rope OFF 분기),
#                      data tti_time_format=off (vision span 에 time marker 미삽입).
#     출력 포맷(<tX> 타임토큰 GT)·reward·lora·clip·lr·steps·seed(2024) 전부 baseline 그대로.
#   서버 차이: /home/team404/workspace → /workspace
#     dataset = uncha_v3_local.json (uncha_v3.json 과 경로만 다른 동일 데이터, 10358개)
#   학습 전 gdpo/preflight_tti_off.py 로 off 정합성 검증 (실패 시 중단). 건너뛰기: PREFLIGHT=0
#   TTI_DEBUG=1: rank-0 첫 3 step TTI 정합성 로그 (학습 무영향).
#   ⚠️ 신규 run — SFT(unav_v8) 에서 fresh 시작. 재개는 RESUME=True bash <this>.
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /workspace/setup.sh
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-salmonn2plus}"
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# wandb 키 미설정이면 offline 으로 기록 (나중에 `wandb sync $OUT/wandb/offline-run-*`)
if [[ -z "$WANDB_API_KEY" || "$WANDB_API_KEY" == "<"* ]]; then
  echo "⚠️ WANDB_API_KEY 미설정 → WANDB_MODE=offline"
  unset WANDB_API_KEY
  export WANDB_MODE=${WANDB_MODE:-offline}
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# TTI 정합성 진단 (rank-0, 첫 3 step 만; 학습에는 영향 없음)
export TTI_DEBUG=${TTI_DEBUG:-1}
export TTI_DEBUG_STEPS=${TTI_DEBUG_STEPS:-3}

SFT_CKPT=$CKPT_DIR/base/salmonn2p_7b_unav_v8
DATASET=$TRAIN_DIR/uncha_v3_local.json
CONFIG=gdpo/config_sep3_uncha_1gpu_ttioff.yaml

# ── 사전 확인 — 없으면 여기서 죽는다 (모델 로드 한참 뒤에 실패하지 않도록) ──
for p in "$SFT_CKPT/config.json" "$DATASET" "$CONFIG" \
         "$BASE_ROOT/data/val/unav100_tail_val100.json" \
         "$BASE_ROOT/data/val/charades_tail_val100_noaudio.json"; do
  [ -e "$p" ] || { echo "❌ 없음: $p"; exit 1; }
done
echo "✅ 모델/데이터 확인 완료"

if [ "${PREFLIGHT:-1}" != "0" ]; then
  SFT_CKPT="$SFT_CKPT" DATASET="$DATASET" python gdpo/preflight_tti_off.py \
    || { echo "❌ preflight 실패 → 학습 중단"; exit 1; }
fi

RUN=${RUN:-sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling_ttioff}
OUT=${OUT:-$CKPT_DIR/gdpo/$RUN}
mkdir -p "$OUT"
export WANDB_DIR="$OUT"

# 재개: RESUME=True (최신 ckpt 자동) 또는 RESUME=<ckpt 경로>
RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS=(--resume_from_checkpoint "$RESUME")

torchrun --standalone --nproc_per_node=1 --master_port=29523 \
  gdpo/gdpo_trainer_batch.py \
  --config       "$CONFIG" \
  --model_path   "$SFT_CKPT" \
  --model_base   "$SFT_CKPT" \
  --dataset_path "$DATASET" \
  --reward_module reward_functions_rM_sep3 \
  --tti_mode off \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  "${RESUME_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$OUT/train.log"
