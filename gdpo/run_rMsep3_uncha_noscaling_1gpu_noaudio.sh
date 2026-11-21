#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling_noaudio
#   = run_rMsep3_uncha_noscaling_1gpu.sh (sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling) 와 동일.
#     유일한 차이: **오디오 미사용** (참고: run_rMsep3_noscaling_noaudio.sh — unpucha 판 noaudio)
#     dataset → uncha_v3_local_noaudio.json  (tools/strip_audio.py: audio 키 제거 + use_audio:false,
#                                              10358개 순서 동일 → seed 2024 데이터 순서 baseline 과 동일)
#     val     → unav100_tail_val100_noaudio.json + charades_tail_val100_noaudio.json
#     그 외 baseline 그대로: SFT salmonn2p_7b_unav_v8 / reward_functions_rM_sep3 / multi_seg_weight=1.0 /
#              lora r32 MLP포함 / clip ε0.2-0.28 μ=2 / lr 1e-5 / warmup 0.05 / max_steps 5179 /
#              tti_mode on / seed 2024 / 1 GPU, grad_accum 4 (eff batch 4).
#   config: gdpo/config_sep3_uncha_1gpu_noaudio.yaml
#
#   ⚠️ TTI_DEBUG=1 — no-audio 는 video-only 마커 분기를 탄다. 로그에서 확인:
#        [TTI-DBG] ⑤ step N : #time_markers_in_prompt  ← 0 이면 실패, 즉시 중단.
#
#   wandb: online, entity=$WANDB_ENTITY, project=huggingface(기존 런들과 동일).
#          run id 를 $OUT/wandb_run_id 에 고정 → RESUME 시 같은 wandb run 에 이어서 기록.
#   실행:  tmux new -s uncha_noaudio  →  bash gdpo/run_rMsep3_uncha_noscaling_1gpu_noaudio.sh
#   재개:  RESUME=True bash gdpo/run_rMsep3_uncha_noscaling_1gpu_noaudio.sh
#   스모크: OUT=/tmp/x RUN=smoke WANDB_MODE=disabled bash <this> --max_steps 2
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /workspace/setup.sh
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate train
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TTI 정합성 진단 — no-audio 경로에서 타임마커 삽입 검증 (rank-0, 첫 10 step; 학습 무영향)
export TTI_DEBUG=1
export TTI_DEBUG_STEPS=10

SFT_CKPT=$CKPT_DIR/base/salmonn2p_7b_unav_v8
DATASET=$TRAIN_DIR/uncha_v3_local_noaudio.json
CONFIG=gdpo/config_sep3_uncha_1gpu_noaudio.yaml

# ── 사전 확인 — 없으면 여기서 죽는다 (모델 로드 한참 뒤에 실패하지 않도록) ──
for p in "$SFT_CKPT/config.json" "$SFT_CKPT/model.safetensors.index.json" "$DATASET" "$CONFIG" \
         "$BASE_ROOT/data/val/unav100_tail_val100_noaudio.json" \
         "$BASE_ROOT/data/val/charades_tail_val100_noaudio.json"; do
  [ -e "$p" ] || { echo "❌ 없음: $p"; exit 1; }
done
# 학습 데이터에 오디오가 하나라도 남아 있으면 중단
python - "$DATASET" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
bad = sum(1 for x in d if "audio" in x or x.get("use_audio"))
assert bad == 0, f"❌ 오디오 남은 샘플 {bad}개"
print(f"✅ no-audio 확인: {len(d)}개")
EOF
echo "✅ 모델/데이터 확인 완료"

RUN=${RUN:-sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling_noaudio}
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
  "${RESUME_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$OUT/train.log"
