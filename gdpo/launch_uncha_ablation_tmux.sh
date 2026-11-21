#!/bin/bash
# reward ablation 2개를 tmux 로 동시 실행: GPU0 = w/o count, GPU1 = w/o precision
#   bash gdpo/launch_uncha_ablation_tmux.sh            # 신규
#   RESUME=True bash gdpo/launch_uncha_ablation_tmux.sh  # 재개 (최신 ckpt + 같은 wandb run)
#   확인: tmux attach -t abl_nocount / abl_noprecision
set -e
GDPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for spec in "nocount 0" "noprecision 1"; do
  set -- $spec
  S=abl_$1
  tmux has-session -t "$S" 2>/dev/null && { echo "⚠️ tmux '$S' 이미 있음 — 건너뜀"; continue; }
  tmux new-session -d -s "$S" \
    "ABL=$1 CUDA_VISIBLE_DEVICES=$2 RESUME=$RESUME bash $GDPO/run_rMsep3_uncha_noscaling_1gpu_ablation.sh; echo '[종료] exit='\$?; exec bash"
  echo "✅ tmux '$S' 시작 (GPU $2)"
done
