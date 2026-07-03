#!/usr/bin/env bash
# P3 Stage1 orchestrator: monitor steps, eval at 4k/8k/12k/16k, restart training.
set -euo pipefail
REPO="/home/gamba/mahjong/Mortal"
RUN_DIR="${RUN_DIR:-/home/gamba/mahjong/runs/ppo/stage1_20260703_064427}"
LOG="$RUN_DIR/logs/orchestrator.log"
MAX_STEPS=16000
EVAL_STEPS=(4000 8000 12000 16000)
TMUX_TRAIN="${TMUX_TRAIN:-ppo_p3_20260703_064427}"

log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

current_step() {
  grep -oP 'ppo step \K[0-9]+' "$RUN_DIR/logs/trainer.log" 2>/dev/null | tail -1 || echo 0
}

stop_online() {
  log "stopping online processes"
  tmux kill-session -t "$TMUX_TRAIN" 2>/dev/null || true
  pkill -f "run_train_ppo.py" 2>/dev/null || true
  pkill -f "run_client.py" 2>/dev/null || true
  pkill -f "run_server.py" 2>/dev/null || true
  pkill -f "client._watchdog" 2>/dev/null || true
  fuser -k 5000/tcp 2>/dev/null || true
  sleep 5
}

wait_for_step() {
  local target=$1
  log "waiting for step >= $target"
  while true; do
    local s
    s=$(current_step)
    if [[ -f "$RUN_DIR/checkpoints/step_$(printf '%06d' "$target").pth" ]] || (( s >= target )); then
      log "reached step $s (target $target)"
      return 0
    fi
    sleep 120
    log "progress step=$s / $target"
  done
}

run_eval() {
  local step=$1
  if grep -q 'eval_sanity: 完了' "$RUN_DIR/logs/eval_sanity_step${step}.log" 2>/dev/null; then
    log "eval step $step already complete, skip"
    return 0
  fi
  log "eval_sanity step $step"
  stop_online
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
    sleep 5
  done
  if ! RUN_DIR="$RUN_DIR" MORTAL_FOREGROUND=1 bash "$REPO/freeparlor/scripts/run_ppo_p3_eval_checkpoint.sh" "$step" >> "$LOG" 2>&1; then
    log "eval step $step failed"
    return 1
  fi
}

start_training() {
  log "starting training tmux=$TMUX_TRAIN"
  RUN_SUFFIX="${RUN_SUFFIX:-20260703_064427}" \
  RUN_DIR="$RUN_DIR" \
  TMUX_SESSION="$TMUX_TRAIN" \
  bash "$REPO/freeparlor/scripts/run_ppo_p3_stage1.sh"
}

main() {
  log "orchestrator start RUN_DIR=$RUN_DIR RESUME_FROM=${RESUME_FROM:-0}"
  local start_idx=0
  if [[ -n "${RESUME_FROM:-}" && "$RESUME_FROM" != "0" ]]; then
    for i in "${!EVAL_STEPS[@]}"; do
      if (( EVAL_STEPS[i] >= RESUME_FROM )); then
        start_idx=$i
        break
      fi
    done
    log "resume from eval step ${EVAL_STEPS[start_idx]}"
  fi
  for (( i=start_idx; i<${#EVAL_STEPS[@]}; i++ )); do
    target=${EVAL_STEPS[i]}
    wait_for_step "$target"
    run_eval "$target" || exit 1
    if (( target < MAX_STEPS )); then
      start_training
    fi
  done
  log "orchestrator complete"
  conda run -n mortal python "$REPO/freeparlor/scripts/summarize_p3_stage1.py" "$RUN_DIR" >> "$LOG" 2>&1 || true
}

main "$@"
