#!/bin/bash
# ============================================================
# TiTok inference + evaluation launcher (eval.sh)
#
# One script covers:
#   - CHUNK=on|off       : chunked inference (with automatic resume) / single-JSON inference
#   - model types        : base + LoRA / fully merged (MERGED_MODEL=) / base only (CKPT_STEP=base)
#   - MODE=infer|eval    : inference + scoring / score existing predictions only (no GPU)
#   - NATURAL=on|off     : lenient parsing (HH:MM:SS, M:SS, seconds, second{}...) / time tokens only (default)
#   - CoT outputs        : only <answer>...</answer> is parsed (handled by eval_miou.py)
#
# Outputs (result folder):
#   test_results_rank0.json   : predictions (MODE=infer)
#   pairwise_miou_summary.json  union_miou_summary.json  sample_miou_summary.json
#   inference.log (infer), eval_miou_progress.jsonl (chunk progress snapshots)
#
# ── Examples (run from the repo root) ─────────────────────
#  # 1) GDPO LoRA checkpoint on UnAV-100 (base model + adapter)
#  bash eval/eval.sh STAGE=gdpo CKPT_MODEL_ID=titok_gdpo CKPT_STEP=2000 #       BASE_MODEL_ID=base/salmonn2p_7b_unav_v8 TEST_JSON=data/test/unav100.json GPUS=0
#
#  # 2) Chunked inference with automatic resume (splits data/test/<TESTSET>.json)
#  bash eval/eval.sh CHUNK=on STAGE=gdpo CKPT_MODEL_ID=titok_gdpo CKPT_STEP=2000 #       BASE_MODEL_ID=base/salmonn2p_7b_unav_v8 TESTSET=unav100 GPUS=0
#
#  # 3) A fully merged model (no LoRA), e.g. the SFT model
#  bash eval/eval.sh MERGED_MODEL=checkpoints/base/salmonn2p_7b_unav_v8 #       TEST_JSON=data/test/unav100.json GPUS=0
#
#  # 4) Re-score existing predictions only (no GPU)
#  bash eval/eval.sh MODE=eval RESULTS=outputs/.../test_results_rank0.json #       TEST_JSON=data/test/unav100.json
# ============================================================
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Defaults (override with KEY=VALUE on the command line) ----
MODE=infer                 # infer | eval
CHUNK=off                  # on | off
CHUNK_SIZE=500
STAGE=sft                  # sft | gdpo
CKPT_MODEL_ID=
CKPT_STEP=                 # empty = latest checkpoint-*, base/no = base model, N or checkpoint-N
BASE_MODEL_ID=base/video_salmonn2_plus_7B_time_tokens
MERGED_MODEL=              # load this path directly (lora_ckpt=No); STAGE/CKPT_* are ignored
TEST_JSON=                 # GT (and inference input) for CHUNK=off / MODE=eval
TESTSET=                   # CHUNK=on: name of data/test/<TESTSET>.json
TESTSET_TAG=               # default: TEST_JSON file name or TESTSET
RESULTS=                   # MODE=eval: result file/folder to score (default: OUT_DIR)
OUT_DIR=                   # default: derived from the model and test set
LABEL=                     # summary label (default: CKPT_MODEL_ID / merged / base name)
NATURAL=off                # on | off
GT_REF=off                 # on = parse GT from ref (tokens). Default: auto (embedded > ref > test_json)
PRED_PERCENT=off           # on = predictions are percentages of the duration
GPUS=0
CONFIG=config.yaml
TTI_TIME_FORMAT_CLI=

for arg in "$@"; do
    case "$arg" in
        MODE=*) MODE="${arg#*=}" ;;
        CHUNK=*) CHUNK="${arg#*=}" ;;
        CHUNK_SIZE=*) CHUNK_SIZE="${arg#*=}" ;;
        STAGE=*) STAGE="${arg#*=}" ;;
        CKPT_MODEL_ID=*) CKPT_MODEL_ID="${arg#*=}" ;;
        CKPT_STEP=*) CKPT_STEP="${arg#*=}" ;;
        BASE_MODEL_ID=*) BASE_MODEL_ID="${arg#*=}" ;;
        MERGED_MODEL=*) MERGED_MODEL="${arg#*=}" ;;
        TEST_JSON=*) TEST_JSON="${arg#*=}" ;;
        TESTSET=*) TESTSET="${arg#*=}" ;;
        TESTSET_TAG=*) TESTSET_TAG="${arg#*=}" ;;
        RESULTS=*) RESULTS="${arg#*=}" ;;
        OUT_DIR=*) OUT_DIR="${arg#*=}" ;;
        LABEL=*) LABEL="${arg#*=}" ;;
        NATURAL=*) NATURAL="${arg#*=}" ;;
        GT_REF=*) GT_REF="${arg#*=}" ;;
        PRED_PERCENT=*) PRED_PERCENT="${arg#*=}" ;;
        GPUS=*) GPUS="${arg#*=}" ;;
        CONFIG=*) CONFIG="${arg#*=}" ;;
        TTI_TIME_FORMAT=*) TTI_TIME_FORMAT_CLI="${arg#*=}" ;;
        *) echo "[error] unsupported argument: $arg"; exit 1 ;;
    esac
done

EVAL_PY="${SCRIPT_DIR}/eval_miou.py"
HELPER="${SCRIPT_DIR}/_chunk_helpers.py"
MAKETABLE="${SCRIPT_DIR}/maketable.py"

# After scoring, refresh table.txt at the category root (gdpo/base/sft/merged).
#   The category folder directly below outputs/ is found from the result path and maketable.py runs there.
#   e.g. .../outputs/gdpo/<run>/checkpoint-200 -> .../outputs/gdpo/table.txt
#       .../outputs/base/MUSEG/Unav100        → .../outputs/base/table.txt
#   (maketable uses only the standard library; failures do not affect the results.)
update_table() {
    local _dir="$1"
    [ -f "$_dir" ] && _dir="$(dirname "$_dir")"
    local _abs; _abs="$(cd "$_dir" 2>/dev/null && pwd)" || _abs="$_dir"
    local _acc="" _root="" _p _oldIFS="$IFS"
    IFS='/'
    for _p in $_abs; do
        [ -z "$_p" ] && continue
        _acc="$_acc/$_p"
        case "$_p" in gdpo|base|sft|merged) _root="$_acc"; break ;; esac
    done
    IFS="$_oldIFS"
    if [ -z "$_root" ]; then
        echo "[table] no category root (gdpo/base/sft/merged) found, skipping table update: $_abs"; return 0
    fi
    echo "[table] maketable → ${_root}/table.txt"
    python3 "$MAKETABLE" "$_root" || echo "[table] maketable failed (ignored): $_root"
}

EXTRA_FLAGS=""
[ "$NATURAL" = "on" ] && EXTRA_FLAGS="$EXTRA_FLAGS --natural"
[ "$GT_REF" = "on" ] && EXTRA_FLAGS="$EXTRA_FLAGS --gt_ref"
[ "$PRED_PERCENT" = "on" ] && EXTRA_FLAGS="$EXTRA_FLAGS --pred_percent"

# ---- Load config.yaml (flat KEY: VALUE) ----
CONFIG_PATH="${SCRIPT_DIR}/${CONFIG}"
[ -f "$CONFIG_PATH" ] || { echo "[error] config not found: $CONFIG_PATH"; exit 1; }
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"; line="${line%%#*}"
    [[ -z "${line// }" ]] && continue
    key="${line%%:*}"; val="${line#*:}"
    key="$(echo "$key" | awk '{$1=$1;print}')"
    val="$(echo "$val" | awk '{$1=$1;print}')"
    val="${val#\"}"; val="${val%\"}"; val="${val#\'}"; val="${val%\'}"
    [[ -z "$key" ]] && continue
    eval "$key=\"\$val\""
done < "$CONFIG_PATH"
[ -n "$TTI_TIME_FORMAT_CLI" ] && TTI_TIME_FORMAT="$TTI_TIME_FORMAT_CLI"
MAX_TIME="${MAX_TIME:-9999.9}"

# ============================================================
# MODE=eval : skip inference and score existing predictions (no GPU)
# ============================================================
if [ "$MODE" = "eval" ]; then
    EVAL_TARGET="${RESULTS:-$OUT_DIR}"
    [ -n "$EVAL_TARGET" ] || { echo "[error] MODE=eval requires RESULTS= or OUT_DIR="; exit 1; }
    [ -e "$EVAL_TARGET" ] || { echo "[error] nothing to score: $EVAL_TARGET"; exit 1; }
    [ -z "$TESTSET_TAG" ] && [ -n "$TEST_JSON" ] && TESTSET_TAG=$(basename "$TEST_JSON" .json)
    [ -z "$LABEL" ] && LABEL="${CKPT_MODEL_ID:-eval}"
    echo "[MODE=eval] target=$EVAL_TARGET  natural=$NATURAL"
    python3 "$EVAL_PY" "$EVAL_TARGET" \
        ${TEST_JSON:+--test_json "$TEST_JSON"} $EXTRA_FLAGS \
        --max_time "$MAX_TIME" --label "$LABEL" --testset "$TESTSET_TAG"
    update_table "$EVAL_TARGET"
    echo "[done] eval-only"
    exit 0
fi

# ============================================================
# MODE=infer : inference + scoring
# ============================================================
# Paths (REPO_ROOT, DATA_DIR, CKPT_DIR, OUTPUT_DIR, MEDIA_ROOT) come from paths.env.
# Activate the Python environment (see README) before running this script.
source "$SCRIPT_DIR/../scripts/env.sh"
export EVAL_DIR="${OUT_ROOT:-$OUTPUT_DIR}"
TEST_DIR="$DATA_DIR/test"
# Relative CLI paths are given from the repo root; make them absolute before cd.
[ -n "$TEST_JSON" ] && TEST_JSON="$(realpath -m "$TEST_JSON")"
[ -n "$MERGED_MODEL" ] && MERGED_MODEL="$(realpath -m "$MERGED_MODEL")"
[ -n "$OUT_DIR" ] && OUT_DIR="$(realpath -m "$OUT_DIR")"

NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')
export CUDA_VISIBLE_DEVICES=$GPUS
export ARNOLD_WORKER_GPU=$NUM_GPUS
FIRST_GPU=$(echo "$GPUS" | awk -F',' '{print $1}')
MASTER_PORT="${MASTER_PORT:-$((12900 + FIRST_GPU))}"

# ---- EVAL_TAG = fps<N>_<format> ----
FPS_INT=$(awk -v b="${BASE_INTERVAL:-0.2}" 'BEGIN { printf "%d", (1.0/b)+0.5 }')
case "${TTI_TIME_FORMAT:-off}" in
    off) FORMAT_TAG=off ;;
    special_token) FORMAT_TAG=tti ;;
    *) FORMAT_TAG="${TTI_TIME_FORMAT:-off}" ;;
esac
EVAL_TAG="fps${FPS_INT}_${FORMAT_TAG}"

BASE_CODE="${REPO_ROOT}/titok"
MODEL_BASE="${CKPT_DIR}/${BASE_MODEL_ID}"

# ---- Model resolution: MERGED_MODEL > CKPT_STEP=base > base + LoRA ----
# IS_BASE_EVAL=true -> no LoRA merge needed (load model_name directly, lora_ckpt=No)
IS_BASE_EVAL=false
if [ -n "$MERGED_MODEL" ]; then
    [ -f "$MERGED_MODEL/config.json" ] || { echo "[error] merged model not found: $MERGED_MODEL"; exit 1; }
    LORA_CKPT="No"; IS_BASE_EVAL=true
    MODEL_LOAD="$MERGED_MODEL"
    OUT_BRANCH="merged/$(basename "$MERGED_MODEL")"
    [ -z "$LABEL" ] && LABEL="$(basename "$MERGED_MODEL")"
else
    STAGE=$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')
    [ "$STAGE" = "sft" ] || [ "$STAGE" = "gdpo" ] || { echo "[error] STAGE must be sft|gdpo"; exit 1; }
    [ -n "$CKPT_MODEL_ID" ] || { echo "[error] CKPT_MODEL_ID is required (or MERGED_MODEL=)"; exit 1; }
    MODEL_DIR="${CKPT_DIR}/${STAGE}/${CKPT_MODEL_ID}"
    CKPT_STEP_LOWER=$(echo "$CKPT_STEP" | tr '[:upper:]' '[:lower:]')
    _root_ckpt_folder() {
        if [ -f "$MODEL_DIR/latest" ]; then
            local _s; _s=$(sed -n 's/.*global_step//p' "$MODEL_DIR/latest")
            echo "checkpoint-${_s:-root}"
        else echo "root"; fi
    }
    if [ -z "$CKPT_STEP" ]; then
        LORA_CKPT=$(ls -d "$MODEL_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1) || true
        if [ -n "$LORA_CKPT" ]; then CKPT_FOLDER=$(basename "$LORA_CKPT")
        elif [ -f "$MODEL_DIR/adapter_config.json" ]; then LORA_CKPT="$MODEL_DIR"; CKPT_FOLDER=$(_root_ckpt_folder)
        else echo "[error] no checkpoint-* or adapter_config.json in $MODEL_DIR"; exit 1; fi
    elif [ "$CKPT_STEP_LOWER" = "base" ] || [ "$CKPT_STEP_LOWER" = "no" ]; then
        LORA_CKPT="No"; CKPT_FOLDER="base"; IS_BASE_EVAL=true
    elif [ "$CKPT_STEP_LOWER" = "root" ] || [ "$CKPT_STEP_LOWER" = "self" ]; then
        [ -f "$MODEL_DIR/adapter_config.json" ] || { echo "[error] no adapter in $MODEL_DIR"; exit 1; }
        LORA_CKPT="$MODEL_DIR"; CKPT_FOLDER=$(_root_ckpt_folder)
    else
        if [[ "$CKPT_STEP" == checkpoint-* ]]; then CKPT_FOLDER="$CKPT_STEP"; else CKPT_FOLDER="checkpoint-${CKPT_STEP}"; fi
        LORA_CKPT="${MODEL_DIR}/${CKPT_FOLDER}"
        [ -d "$LORA_CKPT" ] || { echo "[error] checkpoint not found: $LORA_CKPT"; exit 1; }
    fi
    MODEL_LOAD="$MODEL_BASE"
    if [ "$IS_BASE_EVAL" = "true" ]; then OUT_BRANCH="base/${BASE_MODEL_ID#base/}"
    else OUT_BRANCH="${STAGE}/${CKPT_MODEL_ID}/${CKPT_FOLDER}"; fi
    [ -z "$LABEL" ] && LABEL="${CKPT_MODEL_ID:-${BASE_MODEL_ID#base/}}"
fi

# ---- Test set source / tag / OUT_DIR ----
if [ "$CHUNK" = "on" ]; then
    [ -n "$TESTSET" ] || { echo "[error] CHUNK=on requires TESTSET= (reads data/test/<TESTSET>.json)"; exit 1; }
    CHUNKS_DIR="${EVAL_DIR}/.chunks/${TESTSET}_${CHUNK_SIZE}"
    if [ ! -d "$CHUNKS_DIR" ]; then
        SRC_JSON="${TEST_DIR}/${TESTSET}.json"
        [ -f "$SRC_JSON" ] || { echo "[error] test set not found: $SRC_JSON"; exit 1; }
        python3 "$HELPER" split --test_json "$SRC_JSON" --chunks_dir "$CHUNKS_DIR" --chunk_size "$CHUNK_SIZE"
    fi
    [ -z "$TESTSET_TAG" ] && TESTSET_TAG="$TESTSET"
    # _full.json (GT for scoring; created if missing)
    TEST_JSON="${CHUNKS_DIR}/_full.json"
    python3 "$HELPER" build_full --chunks_dir "$CHUNKS_DIR" >/dev/null
    mapfile -t CHUNK_FILES < <(ls "$CHUNKS_DIR"/chunk_*.json 2>/dev/null | sort)
    N_CHUNKS=${#CHUNK_FILES[@]}
    [ "$N_CHUNKS" -gt 0 ] || { echo "[error] no chunk_*.json in $CHUNKS_DIR"; exit 1; }
else
    [ -f "$TEST_JSON" ] || { echo "[error] TEST_JSON not found: $TEST_JSON"; exit 1; }
    [ -z "$TESTSET_TAG" ] && TESTSET_TAG=$(basename "$TEST_JSON" .json)
fi

[ -z "$OUT_DIR" ] && OUT_DIR="${EVAL_DIR}/${OUT_BRANCH}/${EVAL_TAG}/${TESTSET_TAG}"
mkdir -p "$OUT_DIR"
MASTER_RESULT="$OUT_DIR/test_results_rank0.json"
INFER_LOG="$OUT_DIR/inference.log"
PROGRESS_LOG="$OUT_DIR/eval_miou_progress.jsonl"

cat <<EOF
==================================================
  TiTok eval (MODE=infer)
  CHUNK          : $CHUNK   $( [ "$CHUNK" = on ] && echo "($N_CHUNKS chunks)" )
  MODEL_LOAD     : $MODEL_LOAD
  LORA_CKPT      : $LORA_CKPT   (base_eval=$IS_BASE_EVAL)
  LABEL          : $LABEL
  TEST           : ${TEST_JSON}  (tag=$TESTSET_TAG)
  EVAL_TAG       : $EVAL_TAG  (BASE_INTERVAL=$BASE_INTERVAL, TTI=${TTI_TIME_FORMAT:-off}, NATURAL=$NATURAL)
  GPUS           : $GPUS (count=$NUM_GPUS)
  OUT_DIR        : $OUT_DIR
==================================================
EOF

# ---- One inference call (input json, model, lora as arguments) ----
# Usage: run_inference <dataset_json> <model_name_or_path> <lora_ckpt> <output_dir>
run_inference() {
    local _ds="$1" _model="$2" _lora="$3" _outdir="$4"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        qwenvl/train/train_qwen.py \
        --model_base "$MODEL_BASE" \
        --run_test True --pred_rank 0 \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --model_name_or_path "$_model" \
        --dataset_use "$_ds" \
        --bf16 \
        --output_dir "$_outdir" \
        --num_train_epochs "$NUM_TRAIN_EPOCHS" \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --max_pixels "$MAX_PIXELS" --min_pixels "$MIN_PIXELS" \
        --video_max_frame_pixels "$VIDEO_MAX_FRAME_PIXELS" \
        --video_min_frame_pixels "$VIDEO_MIN_FRAME_PIXELS" \
        --eval_strategy "$EVAL_STRATEGY" --save_strategy "$SAVE_STRATEGY" \
        --learning_rate "$LEARNING_RATE" \
        --model_max_length "$MODEL_MAX_LENGTH" \
        --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
        --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
        --run_name "." --report_to "$REPORT_TO" \
        --video_min_frames "$VIDEO_MIN_FRAMES" --video_max_frames "$VIDEO_MAX_FRAMES" \
        --base_interval "$BASE_INTERVAL" \
        --lora_ckpt "$_lora" \
        --no_audio "$NO_AUDIO" \
        --tti_time_format "${TTI_TIME_FORMAT:-off}" \
        2>&1 | tee -a "$INFER_LOG"
}

# One scoring call -> refresh the three summaries
run_eval() {
    local _quiet="$1"
    python3 "$EVAL_PY" "$MASTER_RESULT" --test_json "$TEST_JSON" $EXTRA_FLAGS \
        --max_time "$MAX_TIME" --out_dir "$OUT_DIR" \
        --label "$LABEL" --testset "$TESTSET_TAG" \
        --progress_log "$PROGRESS_LOG" $_quiet
}

cd "$BASE_CODE"

if [ "$CHUNK" != "on" ]; then
    # ---------- single JSON ----------
    echo "[1/2] inference (tti=${TTI_TIME_FORMAT:-off})" | tee "$INFER_LOG"
    run_inference "$TEST_JSON" "$MODEL_LOAD" "$LORA_CKPT" "$OUT_DIR"
    [ -f "$MASTER_RESULT" ] || { echo "[error] no predictions: $MASTER_RESULT"; exit 1; }
    rm -rf "$OUT_DIR/generation_0" 2>/dev/null || true
    echo "[2/2] eval → 3 summaries"
    run_eval ""
    update_table "$OUT_DIR"
    echo "[done] $OUT_DIR"; exit 0
fi

# ---------- chunked (automatic resume + cached LoRA merge) ----------
CHUNK_IDX_FILE="$OUT_DIR/.chunk_idx"
CHUNK_WORKDIR="$OUT_DIR/.chunk_workdir"
MERGED_CACHE="$OUT_DIR/.merged_model"

RESUME_OUT=$(python3 "$HELPER" resume_offset --master "$MASTER_RESULT" \
    --chunks_dir "$CHUNKS_DIR" --chunk_idx_file "$CHUNK_IDX_FILE")
read START_CHUNK N_MASTER EXPECTED_N <<<"$RESUME_OUT"
if [ "$N_MASTER" -gt "$EXPECTED_N" ]; then
    python3 "$HELPER" truncate_master --master "$MASTER_RESULT" --keep "$EXPECTED_N" >/dev/null
    echo "[CLEAN] truncated master $N_MASTER -> $EXPECTED_N"
fi

SUCCESS=0
cleanup_all() {
    rm -rf "$CHUNK_WORKDIR" 2>/dev/null || true
    if [ "$SUCCESS" = "1" ]; then
        rm -rf "$MERGED_CACHE" 2>/dev/null || true
        rm -f "$CHUNK_IDX_FILE" 2>/dev/null || true
    fi
}
trap cleanup_all EXIT

if [ "$START_CHUNK" -ge "$N_CHUNKS" ]; then
    echo "[SKIP] all $N_CHUNKS chunks done; final scoring only"
    run_eval ""; SUCCESS=1; update_table "$OUT_DIR"; echo "[done] $OUT_DIR"; exit 0
fi

for ((i=START_CHUNK; i<N_CHUNKS; i++)); do
    chunk_json="${CHUNK_FILES[$i]}"
    chunk_id=$(basename "$chunk_json" .json | sed 's/^chunk_//')
    if [ -d "$MERGED_CACHE" ] && [ "$IS_BASE_EVAL" != "true" ]; then
        _model="$MERGED_CACHE"; _lora="No"
    else
        _model="$MODEL_LOAD"; _lora="$LORA_CKPT"
    fi
    rm -rf "$CHUNK_WORKDIR"; mkdir -p "$CHUNK_WORKDIR"
    echo "=== [CHUNK $((i+1))/$N_CHUNKS] id=$chunk_id $(date -Iseconds) ===" | tee -a "$INFER_LOG"
    run_inference "$chunk_json" "$_model" "$_lora" "$CHUNK_WORKDIR"

    chunk_result="$CHUNK_WORKDIR/test_results_rank0.json"
    [ -f "$chunk_result" ] || { echo "[error] no result for chunk $chunk_id"; exit 1; }
    # keep the first LoRA merge output as a cache for later chunks
    if [ "$IS_BASE_EVAL" != "true" ] && [ ! -d "$MERGED_CACHE" ] && [ -d "$CHUNK_WORKDIR/generation_0" ]; then
        mv "$CHUNK_WORKDIR/generation_0" "$MERGED_CACHE"; echo "[MERGE] cached -> $MERGED_CACHE"
    fi
    AGG_OUT=$(python3 "$HELPER" append --master "$MASTER_RESULT" --chunk_results "$chunk_result")
    read N_AFTER N_ADDED <<<"$AGG_OUT"
    echo "$((i+1))" > "$CHUNK_IDX_FILE.tmp" && mv "$CHUNK_IDX_FILE.tmp" "$CHUNK_IDX_FILE"
    run_eval "--quiet"
    echo "[CHUNK $((i+1))/$N_CHUNKS] master_n=$N_AFTER added=$N_ADDED $(date -Iseconds)"
done

echo ""; echo "=== FINAL EVAL $(date -Iseconds) ==="
run_eval ""
SUCCESS=1
update_table "$OUT_DIR"
echo "[done] $OUT_DIR"
