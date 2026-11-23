#!/bin/bash
# Shared environment for all TiTok launchers. Source it, do not execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"
#
# Defines REPO_ROOT, DATA_DIR, CKPT_DIR, OUTPUT_DIR and loads paths.env.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

if [ -f "$REPO_ROOT/paths.env" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/paths.env"
else
    echo "[env] $REPO_ROOT/paths.env not found; using defaults (cp paths.example.env paths.env)" >&2
fi

export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
export CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs}"

if [ -z "$MEDIA_ROOT" ] || [ "$MEDIA_ROOT" = "/path/to/media" ]; then
    echo "[env] MEDIA_ROOT is not set. Set it in paths.env (see paths.example.env)." >&2
    return 1 2>/dev/null || exit 1
fi
export MEDIA_ROOT

# Log to W&B offline unless an API key is configured.
if [ -z "$WANDB_API_KEY" ] || [[ "$WANDB_API_KEY" == "<"* ]]; then
    unset WANDB_API_KEY
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

# Single-node defaults expected by the video-SALMONN 2+ training code.
export ARNOLD_WORKER_NUM="${ARNOLD_WORKER_NUM:-1}"
export ARNOLD_ID="${ARNOLD_ID:-0}"
export METIS_WORKER_0_HOST="${METIS_WORKER_0_HOST:-localhost}"
