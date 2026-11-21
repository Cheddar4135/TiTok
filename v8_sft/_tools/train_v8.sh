#!/bin/bash
# ============================================================
# TiTok stage 1: SFT (paper setting, 4 GPUs, effective batch 8).
#
#   bash v8_sft/_tools/train_v8.sh
#   EVAL_DATASET=<json> bash v8_sft/_tools/train_v8.sh   # optional held-out loss every 250 steps
#
# The SFT stage only teaches the time-token output format and audio/video
# alignment; temporal grounding itself is learned in the GDPO stage.
#   - data:  UnAV-100 training split (data/sft/unav100_sft.json), 1 epoch
#   - start: video-SALMONN 2+ 7B with time tokens added
#            (tools/sft/add_time_tokens_salmonn2plus.py)
#   - LoRA r=16 / alpha=16 / dropout=0.1 on q,k,v,o to preserve instruction following
#   - only the time-token rows of embed_tokens / lm_head are trained
#   - TTI is off during SFT
# A checkpoint is saved every 250 steps; the paper uses checkpoint-250,
# merged into the base with v8_sft/_tools/merge_v8_unav_to_base.py.
# ============================================================
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export ARNOLD_WORKER_GPU="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')"
MASTER_PORT="${MASTER_PORT:-29500}"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${MODEL:-$CKPT_DIR/base/video_salmonn2_plus_7B_time_tokens}"
MODEL_BASE="$MODEL"
DATASET="${DATASET:-$DATA_DIR/sft/unav100_sft.json}"
RUN_NAME="${RUN_NAME:-titok_sft}"
OUTPUT_DIR="${OUT:-$CKPT_DIR/sft/$RUN_NAME}"

for p in "$MODEL/config.json" "$DATASET"; do
    [ -e "$p" ] || { echo "[error] not found: $p" >&2; exit 1; }
done

if [ -n "$EVAL_DATASET" ]; then
    EVAL_ARGS=(--eval_dataset_use "$EVAL_DATASET" --eval_strategy steps --eval_steps 250)
else
    EVAL_ARGS=(--eval_strategy no)
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT/v8_sft/video_SALMONN2_plus"

torchrun --standalone --nproc_per_node="$ARNOLD_WORKER_GPU" --master_port="$MASTER_PORT" \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$MODEL" --model_base "$MODEL_BASE" \
    --dataset_use "$DATASET" "${EVAL_ARGS[@]}" \
    --tune_mm_vision False --tune_mm_mlp True --tune_mm_llm False \
    --tune_mm_audio False --tune_mm_qformer True \
    --use_lora True --lora_r 16 --lora_alpha 16 --lora_dropout 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --bf16 --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_pixels 176400 --min_pixels 784 \
    --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
    --save_strategy "steps" --save_steps 250 \
    --save_total_limit 10 \
    --learning_rate 1e-4 --weight_decay 0 \
    --lora_lr 1e-4            --lora_wd 0.0 \
    --embed_lr 1e-4           --embed_wd 0.0 \
    --lm_head_lr 1e-4         --lm_head_wd 0.0 \
    --visual_merger_lr 1e-5   --visual_merger_wd 0.01 \
    --audio_qformer_lr 2e-5   --audio_qformer_wd 0.01 \
    --audio_proj_lr 2e-5      --audio_proj_wd 0.01 \
    --audio_q_tokens_lr 5e-5  --audio_q_tokens_wd 0.0 \
    --tti_time_format off \
    --warmup_ratio 0.03 --max_grad_norm 1 \
    --lr_scheduler_type "cosine" --logging_steps 1 \
    --model_max_length 32768 --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "$RUN_NAME" --report_to wandb \
    --video_min_frames 64 --video_max_frames 256 --base_interval 0.2 \
    --train_type sft --no_audio False \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"

echo "[sft] done. Merge the paper checkpoint with:"
echo "  python v8_sft/_tools/merge_v8_unav_to_base.py --lora_ckpt $OUTPUT_DIR/checkpoint-250"
