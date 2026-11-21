#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling_GRPO   (ablation: w/o GDPO → GRPO)
#   = run_rMsep3_uncha_noscaling_1gpu.sh (sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling) 와 동일.
#     유일한 차이: advantage 계산 GDPO → GRPO  (참고: run_rMsep3_unpucha_v8_GRPO.sh — unpucha 판 GRPO)
#       trainer: gdpo_trainer_batch.py → gdpo_trainer_batch_GRPO.py
#         (두 파일 diff = _compute_gdpo_advantages → _compute_grpo_advantages 뿐.
#          GRPO 판엔 --seed/--gradient_accumulation_steps CLI 만 없음 → baseline 도 둘 다 null(config 값) 이라 무관)
#     dataset → uncha_v3_local.json (uncha_v3.json 경로만 /workspace 로 치환, 10358개 순서 동일
#                                    → seed 2024 데이터 순서 baseline 과 동일)
#     그 외 baseline 그대로: SFT salmonn2p_7b_unav_v8 / reward_functions_rM_sep3 (5채널, weight 1) /
#              multi_seg_weight=1.0 / lora r32 MLP포함 / clip ε0.2-0.28 μ=2 / lr 1e-5 / warmup 0.05 /
#              max_steps 5179 / tti_mode on / seed 2024 / 1 GPU, grad_accum 4 (eff batch 4).
#   config: gdpo/config_sep3_uncha_1gpu_GRPO.yaml
#
#   wandb: online, entity=$WANDB_ENTITY, project=huggingface(기존 런들과 동일).
#          run id 를 $OUT/wandb_run_id 에 고정 → RESUME 시 같은 wandb run 에 이어서 기록.
#   실행:  tmux new -s uncha_grpo  →  bash gdpo/run_rMsep3_uncha_noscaling_1gpu_GRPO.sh
#   재개:  RESUME=True bash gdpo/run_rMsep3_uncha_noscaling_1gpu_GRPO.sh
#   스모크: OUT=/tmp/x RUN=smoke WANDB_MODE=disabled bash <this> --max_steps 2
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /workspace/setup.sh
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-train}"
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SFT_CKPT=$CKPT_DIR/base/salmonn2p_7b_unav_v8
DATASET=$TRAIN_DIR/uncha_v3_local.json
CONFIG=gdpo/config_sep3_uncha_1gpu_GRPO.yaml
TRAINER=gdpo/gdpo_trainer_batch_GRPO.py

# ── 사전 확인 — 없으면 여기서 죽는다 (모델 로드 한참 뒤에 실패하지 않도록) ──
for p in "$SFT_CKPT/config.json" "$SFT_CKPT/model.safetensors.index.json" "$DATASET" "$CONFIG" "$TRAINER" \
         "$BASE_ROOT/data/val/unav100_tail_val100.json" \
         "$BASE_ROOT/data/val/charades_tail_val100_noaudio.json"; do
  [ -e "$p" ] || { echo "❌ 없음: $p"; exit 1; }
done
grep -q "_compute_grpo_advantages(rewards_per_func" "$TRAINER" || { echo "❌ $TRAINER 가 GRPO advantage 를 안 씀"; exit 1; }
echo "✅ 모델/데이터/trainer(GRPO) 확인 완료"

RUN=${RUN:-sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling_GRPO}
OUT=${OUT:-$CKPT_DIR/gdpo/$RUN}
mkdir -p "$OUT"

# ── wandb (online) ──
if [[ "${WANDB_MODE:-online}" != "disabled" ]]; then
  if [[ -z "$WANDB_API_KEY" || "$WANDB_API_KEY" == "<"* ]]; then
    echo "❌ WANDB_API_KEY 미설정 (paths.env)"; exit 1
  fi
  export WANDB_MODE=${WANDB_MODE:-online}
  export WANDB_ENTITY
  export WANDB_DIR="$OUT"
  [ -s "$OUT/wandb_run_id" ] || python -c "import wandb;print(wandb.util.generate_id())" > "$OUT/wandb_run_id"
  export WANDB_RUN_ID=$(cat "$OUT/wandb_run_id")
  export WANDB_RESUME=allow
  echo "✅ wandb: mode=$WANDB_MODE entity=$WANDB_ENTITY project=${WANDB_PROJECT:-huggingface} id=$WANDB_RUN_ID"
fi

# 재개: RESUME=True (최신 ckpt 자동) 또는 RESUME=<ckpt 경로>
RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS=(--resume_from_checkpoint "$RESUME")

torchrun --standalone --nproc_per_node=1 --master_port=29543 \
  "$TRAINER" \
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
