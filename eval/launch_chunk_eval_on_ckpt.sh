#!/bin/bash
# ============================================================
# launch_chunk_eval_on_ckpt.sh
#   학습 중인 run 의 checkpoint-<STEP> 이 저장 완료될 때까지 기다렸다가,
#   eval.sh CHUNK=on 으로 unav100 / charades 청크 평가를 tmux 창 2개에서 동시에 시작.
#   (GPU 1장 공유: 학습 + eval 2개. ckpt-1000 때도 같은 구성으로 돌았음)
#
#   사용 (tmux 안에서 대기):
#     tmux new-session -d -s eval_ckpt2000 "bash /workspace/TiTok/eval/launch_chunk_eval_on_ckpt.sh STEP=2000; exec bash"
#   결과: /workspace/outputs/gdpo/<RUN>/checkpoint-<STEP>/fps5_tti/{unav100_titok,charades_sta_titok_unchastyle}/
#   중간에 끊겨도 같은 명령 재실행 → eval.sh 가 chunk 단위 resume.
# ============================================================
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEP=2000
RUN=sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling
BASE_MODEL_ID=base/salmonn2p_7b_unav_v8
GPU=0
CONDA_ENV=train_blackwell      # Blackwell(sm_120) 지원 torch — `train` env 는 이 GPU 에서 kernel 없음
SESSION=eval_ckpt${STEP}
for arg in "$@"; do case "$arg" in
    STEP=*) STEP="${arg#*=}"; SESSION=eval_ckpt${STEP};;
    RUN=*) RUN="${arg#*=}";;
    BASE_MODEL_ID=*) BASE_MODEL_ID="${arg#*=}";;
    GPU=*) GPU="${arg#*=}";;
    CONDA_ENV=*) CONDA_ENV="${arg#*=}";;
    SESSION=*) SESSION="${arg#*=}";;
    *) echo "[에러] 모르는 인자: $arg"; exit 1;;
esac; done

CKPT=/workspace/checkpoints/gdpo/$RUN/checkpoint-$STEP

# ---- 1) ckpt 저장 완료 대기: trainer_state.json 은 저장 마지막에 쓰임 ----
echo "[wait] $CKPT 대기 중... $(date -Iseconds)"
until [ -f "$CKPT/trainer_state.json" ] && [ -f "$CKPT/adapter_model.safetensors" ]; do sleep 30; done
sleep 60   # optimizer/rng 등 마저 flush 되도록 여유
echo "[ready] $CKPT $(date -Iseconds)"

# ---- 2) unav / charades 동시 실행 (MASTER_PORT 분리) ----
# eval.sh 가 `conda info --base` 로 conda.sh 를 찾으므로 PATH 에 conda bin 추가
ENVS="export PATH=/workspace/miniconda3/bin:\$PATH CONDA_ENV=$CONDA_ENV"
COMMON="CHUNK=on STAGE=gdpo CKPT_MODEL_ID=$RUN CKPT_STEP=$STEP BASE_MODEL_ID=$BASE_MODEL_ID GPUS=$GPU"
LOGDIR=/workspace/outputs
launch() {   # launch <window> <TESTSET> <port>
    local _log="$LOGDIR/eval_uncha_rl_ckpt${STEP}_$2.launch.log"
    tmux new-window -t "$SESSION" -n "$1" \
        "$ENVS MASTER_PORT=$3; bash $SCRIPT_DIR/eval.sh $COMMON TESTSET=$2 2>&1 | tee -a $_log; exec bash"
    echo "[launch] $1 → TESTSET=$2 port=$3 log=$_log"
}
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n main
launch unav     unav100_titok                  12910
sleep 120    # LoRA merge/모델 로드 시점 겹치지 않게 살짝 시차
launch charades charades_sta_titok_unchastyle  12911
echo "[done] tmux attach -t $SESSION  (창: unav / charades)"
