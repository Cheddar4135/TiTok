#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling_{nocount,noprecision}   (reward ablation)
#   = run_rMsep3_uncha_noscaling_1gpu.sh (sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling) 와 동일.
#   유일한 차이: reward 채널 하나 제거 (참고: unpucha 판 run_rMsep3_unpucha_v8_ttifix_{nocount,noprecision}.sh)
#     ABL=nocount     → reward_functions_rM_sep3_nocount     [format, global, local, precision]
#     ABL=noprecision → reward_functions_rM_sep3_noprecision [format, count, global, local]
#     config = gdpo/config_sep3_uncha_1gpu_${ABL}.yaml (reward_weights 4개)
#   그 외 baseline 그대로: SFT salmonn2p_7b_unav_v8 / multi_seg_weight=1.0 / lora r32 MLP포함 /
#     clip ε0.2-0.28 μ=2 / lr 1e-5 / warmup 0.05 / max_steps 5179 / tti_mode on / seed 2024 /
#     1 GPU, grad_accum 4 (eff batch 4).
#   서버 차이: dataset = uncha_v3_local.json (uncha_v3.json 과 경로만 다르고 10358개 순서 동일
#     → seed 2024 + 같은 block-repeat sampler 로 데이터 순서 baseline 과 동일)
#
#   wandb: online, entity=$WANDB_ENTITY, project=huggingface(기존 런들과 동일).
#          run id 를 $OUT/wandb_run_id 에 고정 → RESUME 시 같은 wandb run 에 이어서 기록.
#   실행:  bash gdpo/launch_uncha_ablation_tmux.sh   (GPU0=nocount, GPU1=noprecision, tmux 2개)
#   단독:  ABL=nocount CUDA_VISIBLE_DEVICES=0 bash gdpo/run_rMsep3_uncha_noscaling_1gpu_ablation.sh
#   재개:  RESUME=True ABL=... CUDA_VISIBLE_DEVICES=... bash <this>
#   스모크: OUT=/tmp/x RUN=smoke WANDB_MODE=disabled ABL=nocount bash <this> --max_steps 2
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /workspace/setup.sh
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-train_blackwell}"
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

case "$ABL" in
  nocount)     PORT=29545 ;;
  noprecision) PORT=29546 ;;
  *) echo "❌ ABL=nocount|noprecision 를 지정하세요 (현재: '$ABL')"; exit 1 ;;
esac

[ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "❌ CUDA_VISIBLE_DEVICES 를 지정하세요"; exit 1; }
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# baseline 은 torchcodec 없는 환경 → decord 로 프레임 샘플링. train_blackwell 엔 torchcodec 이 있어
# 그대로 두면 ~10% 샘플의 frame_idx 가 달라짐 → decord 강제로 입력 동일화.
export DISABLE_TORCHCODEC=1

SFT_CKPT=$CKPT_DIR/base/salmonn2p_7b_unav_v8
DATASET=$TRAIN_DIR/uncha_v3_local.json
CONFIG=gdpo/config_sep3_uncha_1gpu_${ABL}.yaml
REWARD=reward_functions_rM_sep3_${ABL}

# ── 사전 확인 — 없으면 여기서 죽는다 (모델 로드 한참 뒤에 실패하지 않도록) ──
for p in "$SFT_CKPT/config.json" "$SFT_CKPT/model.safetensors.index.json" "$DATASET" "$CONFIG" \
         "gdpo/${REWARD}.py" \
         "$BASE_ROOT/data/val/unav100_tail_val100.json" \
         "$BASE_ROOT/data/val/charades_tail_val100_noaudio.json"; do
  [ -e "$p" ] || { echo "❌ 없음: $p"; exit 1; }
done
# reward 채널 수 == config reward_weights 길이
python - "$REWARD" "$CONFIG" <<'EOF'
import sys, yaml, importlib
sys.path.insert(0, "gdpo")
ch = [c[0] for c in importlib.import_module(sys.argv[1]).REWARD_CHANNELS]
w = yaml.safe_load(open(sys.argv[2]))["gdpo"]["reward_weights"]
assert len(ch) == len(w), f"❌ 채널 {ch} vs weights {w}"
print(f"✅ reward 채널 {ch}  weights {w}")
EOF
echo "✅ 모델/데이터 확인 완료"

RUN=${RUN:-sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling_${ABL}}
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

echo "▶ ABL=$ABL GPU=$CUDA_VISIBLE_DEVICES port=$PORT env=$CONDA_DEFAULT_ENV OUT=$OUT"
torchrun --standalone --nproc_per_node=1 --master_port=$PORT \
  gdpo/gdpo_trainer_batch.py \
  --config       "$CONFIG" \
  --model_path   "$SFT_CKPT" \
  --model_base   "$SFT_CKPT" \
  --dataset_path "$DATASET" \
  --reward_module "$REWARD" \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  "${RESUME_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$OUT/train.log"
