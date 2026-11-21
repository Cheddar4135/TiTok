#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling_noaudio  ── RESUME
#   = run_rMsep3_uncha_noscaling_1gpu_noaudio.sh 와 동일 설정, checkpoint-200(global_step 200/5179)부터 재개.
#   차이: conda env = CONDA_ENV (기본 train_blackwell)
#   wandb: online, $OUT/wandb_run_id (tjecz3hi) 로 같은 run 에 이어서 기록 (WANDB_RESUME=allow).
#   실행:  tmux new -s uncha_noaudio  →  bash gdpo/run_rMsep3_uncha_noscaling_1gpu_noaudio_resume.sh
#          RESUME=True ... → output_dir 내 최신 ckpt 자동 선택
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /workspace/setup.sh
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-train_blackwell}"
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SFT_CKPT=$CKPT_DIR/base/salmonn2p_7b_unav_v8
DATASET=$TRAIN_DIR/uncha_v3_local_noaudio.json
CONFIG=gdpo/config_sep3_uncha_1gpu_noaudio.yaml

RUN=${RUN:-sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling_noaudio}
OUT=${OUT:-$CKPT_DIR/gdpo/$RUN}
RESUME=${RESUME:-$OUT/checkpoint-200}

# ── 사전 확인 — 없으면 여기서 죽는다 (모델 로드 한참 뒤에 실패하지 않도록) ──
for p in "$SFT_CKPT/config.json" "$SFT_CKPT/model.safetensors.index.json" "$DATASET" "$CONFIG" \
         "$BASE_ROOT/data/val/unav100_tail_val100_noaudio.json" \
         "$BASE_ROOT/data/val/charades_tail_val100_noaudio.json"; do
  [ -e "$p" ] || { echo "❌ 없음: $p"; exit 1; }
done
if [[ "$RESUME" != "True" ]]; then
  for f in adapter_model.safetensors optimizer.pt scheduler.pt rng_state.pth trainer_state.json; do
    [ -e "$RESUME/$f" ] || { echo "❌ 없음: $RESUME/$f"; exit 1; }
  done
fi

# ── wandb (online, 기존 run 이어서) ──
if [[ -z "$WANDB_API_KEY" || "$WANDB_API_KEY" == "<"* ]]; then
  echo "❌ WANDB_API_KEY 미설정 (paths.env)"; exit 1
fi
[ -s "$OUT/wandb_run_id" ] || { echo "❌ 없음: $OUT/wandb_run_id"; exit 1; }
export WANDB_MODE=online
export WANDB_ENTITY
export WANDB_DIR="$OUT"
export WANDB_RUN_ID=$(cat "$OUT/wandb_run_id")
export WANDB_RESUME=allow
echo "✅ 확인 완료 (resume=$RESUME, wandb mode=$WANDB_MODE entity=$WANDB_ENTITY project=${WANDB_PROJECT:-huggingface} id=$WANDB_RUN_ID)"

torchrun --standalone --nproc_per_node=1 --master_port=29541 \
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
