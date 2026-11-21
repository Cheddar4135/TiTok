#!/bin/bash
# ============================================================
# run_eval_at_2000_uncha.sh
#   1) checkpoint-2000 저장 완료까지 대기
#   2) 저장 직후(on_save val 시작 전후) 학습 종료 → val 생략
#   3) checkpoint-2000 을 unav100_titok, charades_sta_titok_unchastyle 로 평가 (eval.sh CHUNK=on)
#
#   로그: $WS/outputs/gdpo/$RUN/eval_ck2000.log
#   중단: kill $(cat $WS/.run_eval_2000.pid)
# ============================================================
set -uo pipefail

WS=/home/team404/workspace
RUN=sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling
CK=2000
CKDIR=$WS/checkpoints/gdpo/$RUN/checkpoint-$CK
TRAIN_PARENT=128062    # torchrun
TRAIN_CHILD=128082     # 실제 학습 프로세스
LOG=$WS/outputs/gdpo/$RUN/eval_ck$CK.log

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== START $(date -Is)  pid=$$"
echo $$ > $WS/.run_eval_2000.pid

# ---------- 1) checkpoint-2000 저장 완료 대기 ----------
# HF Trainer 는 model → optimizer/scheduler → rng → trainer_state.json 순으로 저장 후 on_save(val) 호출
while [ ! -f "$CKDIR/trainer_state.json" ]; do
    if ! kill -0 "$TRAIN_CHILD" 2>/dev/null; then
        echo "[에러] checkpoint-$CK 저장 전에 학습 프로세스 종료됨 — 중단"; exit 1
    fi
    sleep 20
done
sleep 15   # 마지막 파일 flush 여유
echo "[CKPT] $CKDIR 저장 확인 $(date -Is)"
ls -la "$CKDIR"
for f in adapter_model.safetensors adapter_config.json optimizer.pt scheduler.pt trainer_state.json; do
    [ -s "$CKDIR/$f" ] || { echo "[에러] $f 누락 — 학습 종료하지 않고 중단"; exit 1; }
done

# ---------- 2) 학습 종료 (val 생략) ----------
echo "[STOP] 학습 종료 요청 (SIGTERM) $(date -Is)"
kill -TERM "$TRAIN_PARENT" "$TRAIN_CHILD" 2>/dev/null
for i in $(seq 1 60); do
    kill -0 "$TRAIN_CHILD" 2>/dev/null || break
    sleep 5
done
if kill -0 "$TRAIN_CHILD" 2>/dev/null; then
    echo "[STOP] SIGTERM 무응답 → SIGKILL"
    kill -9 "$TRAIN_PARENT" "$TRAIN_CHILD" 2>/dev/null
fi
echo "[STOP] 학습 종료 확인 $(date -Is) — GPU 메모리 반환 대기"
for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && break
    sleep 5
done
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# ---------- 3) 평가 ----------
source $WS/setup.sh
export PATH="$CONDA_ROOT/bin:$PATH"   # eval.sh 의 `conda info --base` 용
export CONDA_ENV=salmonn2plus
for TESTSET in unav100_titok charades_sta_titok_unchastyle; do
    echo "[RUN] $TESTSET 시작 $(date -Is)"
    bash "$WS/TiTok/eval/eval.sh" CHUNK=on STAGE=gdpo \
        CKPT_MODEL_ID="$RUN" CKPT_STEP="$CK" \
        BASE_MODEL_ID=base/salmonn2p_7b_unav_v8 \
        TESTSET="$TESTSET" GPUS=0 \
        || echo "[WARN] $TESTSET eval.sh 비정상 종료"
    echo "[RUN] $TESTSET 끝 $(date -Is)"
done

echo "=== DONE $(date -Is) ==="
rm -f $WS/.run_eval_2000.pid
