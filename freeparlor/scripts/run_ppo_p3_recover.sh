#!/usr/bin/env bash
# P3 Stage1 recovery: retry failed eval, resume training toward 16k.
set -euo pipefail
REPO="/home/gamba/mahjong/Mortal"
export RUN_DIR="${RUN_DIR:-/home/gamba/mahjong/runs/ppo/stage1_20260703_064427}"
LOG="$RUN_DIR/logs/recover.log"
TMUX_TRAIN="${TMUX_TRAIN:-ppo_p3_20260703_064427}"
TMUX_ORCH="${TMUX_ORCH:-ppo_p3_orch}"

log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

eval_done() {
  local step=$1
  grep -q 'eval_sanity: 完了' "$RUN_DIR/logs/eval_sanity_step${step}.log" 2>/dev/null
}

stop_all() {
  tmux kill-session -t "$TMUX_ORCH" 2>/dev/null || true
  tmux kill-session -t "$TMUX_TRAIN" 2>/dev/null || true
  pkill -f "run_train_ppo.py" 2>/dev/null || true
  pkill -f "run_client.py" 2>/dev/null || true
  pkill -f "run_server.py" 2>/dev/null || true
  pkill -f "client._watchdog" 2>/dev/null || true
  pkill -f "eval_ppo_smoke_sanity.py" 2>/dev/null || true
  fuser -k 5000/tcp 2>/dev/null || true
  sleep 5
}

log "=== P3 recovery start ==="
stop_all

if ! eval_done 4000; then
  log "retry eval step4000"
  RUN_DIR="$RUN_DIR" MORTAL_FOREGROUND=1 bash "$REPO/freeparlor/scripts/run_ppo_p3_eval_checkpoint.sh" 4000 | tee -a "$LOG"
  if ! eval_done 4000; then
    log "FATAL: eval step4000 still incomplete"
    exit 1
  fi
else
  log "eval step4000 already complete, skip"
fi

log "resume training from mortal.pth (step ~4000)"
RUN_SUFFIX=20260703_064427 RUN_DIR="$RUN_DIR" TMUX_SESSION="$TMUX_TRAIN" \
  bash "$REPO/freeparlor/scripts/run_ppo_p3_stage1.sh"

log "restart orchestrator from step 8000"
tmux kill-session -t "$TMUX_ORCH" 2>/dev/null || true
tmux new-session -d -s "$TMUX_ORCH" \
  "RESUME_FROM=8000 RUN_DIR=$RUN_DIR MORTAL_FOREGROUND=1 bash $REPO/freeparlor/scripts/run_ppo_p3_orchestrator.sh; exec bash"

log "recovery launched: train=$TMUX_TRAIN orch=$TMUX_ORCH"
