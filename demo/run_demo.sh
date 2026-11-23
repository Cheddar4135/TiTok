#!/bin/bash
# ============================================================
# Gradio demo for TiTok temporal grounding.
#   bash demo/run_demo.sh
#   DEMO_SHARE=1 bash demo/run_demo.sh      # also create a temporary public link
# Defaults load the paper model: GDPO checkpoint-2000 on top of the merged SFT model.
# ============================================================
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/env.sh"
cd "$REPO_ROOT"

export DEMO_MODEL_BASE="${DEMO_MODEL_BASE:-$CKPT_DIR/base/salmonn2p_7b_unav_v8}"
export DEMO_MODEL_PATH="${DEMO_MODEL_PATH:-$CKPT_DIR/gdpo/titok_gdpo/checkpoint-2000}"
export DEMO_TTI_MODE="${DEMO_TTI_MODE:-on}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

python demo/app.py
