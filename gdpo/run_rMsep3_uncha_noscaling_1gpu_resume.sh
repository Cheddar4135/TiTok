#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling  ── RESUME (/workspace 서버)
#   = run_rMsep3_uncha_noscaling_1gpu.sh 와 동일 설정, checkpoint-1800(global_step 1800/5179)부터 재개.
#   서버 차이: /home/team404/workspace → /workspace
#     conda   = /workspace/miniconda3, env 는 CONDA_ENV (기본 train_blackwell)
#     dataset = uncha_v3_local.json (uncha_v3.json 과 경로만 다른 동일 데이터, 10358개)
#     config  = config_sep3_uncha_1gpu_local.yaml (val 경로만 /workspace 로 치환)
#   wandb: 원 run 은 offline(id 7q90mzyj) → 같은 id 로 이어서 기록 (WANDB_RESUME=allow).
#     ⚠️ online 으로 이어 쓰기 전에 기존 offline 기록을 먼저 올려두는 게 안전:
#        wandb sync $OUT/wandb/offline-run-20260929_195352-7q90mzyj
#   실행 예: CUDA_VISIBLE_DEVICES=0 bash gdpo/run_rMsep3_uncha_noscaling_1gpu_resume.sh
#            RESUME=True ... → output_dir 내 최신 ckpt 자동 선택
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-train_blackwell}"
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# wandb 키 미설정이면 offline 으로 기록 (나중에 `wandb sync $OUT/wandb/offline-run-*`)
if [[ -z "$WANDB_API_KEY" || "$WANDB_API_KEY" == "<"* ]]; then
  echo "⚠️ WANDB_API_KEY 미설정 → WANDB_MODE=offline"
  unset WANDB_API_KEY
  export WANDB_MODE=${WANDB_MODE:-offline}
fi

# ── wandb 이어서 기록 ──
export WANDB_RUN_ID=${WANDB_RUN_ID:-7q90mzyj}
export WANDB_RESUME=${WANDB_RESUME:-allow}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DISABLE_TORCHCODEC=${DISABLE_TORCHCODEC:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

SFT_CKPT=$CKPT_DIR/base/salmonn2p_7b_unav_v8
DATASET=$TRAIN_DIR/uncha_v3_local.json
CONFIG=${CONFIG:-gdpo/config_sep3_uncha_1gpu_local.yaml}

RUN=${RUN:-sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling}
OUT=${OUT:-$CKPT_DIR/gdpo/$RUN}
RESUME=${RESUME:-$OUT/checkpoint-1800}

# ── 사전 확인 — 없으면 여기서 죽는다 (모델 로드 한참 뒤에 실패하지 않도록) ──
for p in "$SFT_CKPT/config.json" "$DATASET" "$CONFIG" \
         "$BASE_ROOT/data/val/unav100_tail_val100.json" \
         "$BASE_ROOT/data/val/charades_tail_val100_noaudio.json"; do
  [ -e "$p" ] || { echo "❌ 없음: $p"; exit 1; }
done
if [[ "$RESUME" != "True" ]]; then
  for f in adapter_model.safetensors optimizer.pt scheduler.pt rng_state.pth trainer_state.json; do
    [ -e "$RESUME/$f" ] || { echo "❌ 없음: $RESUME/$f"; exit 1; }
  done
fi
echo "✅ 모델/데이터/체크포인트 확인 완료 (resume=$RESUME, wandb id=$WANDB_RUN_ID)"

mkdir -p "$OUT"
export WANDB_DIR="$OUT"

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
  --resume_from_checkpoint "$RESUME" \
  "$@" \
  2>&1 | tee -a "$OUT/train.log"
