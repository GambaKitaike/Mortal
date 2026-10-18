#!/usr/bin/env bash
# PPO P3 — eval_sanity at a numbered checkpoint (stops online, runs eval, optional resume hint).
set -euo pipefail
REPO="/home/gamba/mahjong/Mortal"
RUN_DIR="${RUN_DIR:-/home/gamba/mahjong/runs/ppo/stage1_20260703_064427}"
STEP="${1:?usage: run_ppo_p3_eval_checkpoint.sh STEP}"
CKPT="$RUN_DIR/checkpoints/step_$(printf '%06d' "$STEP").pth"
CFG="$RUN_DIR/config.toml"
EVAL_LABEL="step${STEP}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

echo "=== Stopping online for eval (single-stream) ==="
TMUX_TRAIN="${TMUX_TRAIN:-ppo_p3_20260703_064427}"
tmux kill-session -t "$TMUX_TRAIN" 2>/dev/null || true
pkill -f "run_train_ppo.py" 2>/dev/null || true
pkill -f "run_client.py" 2>/dev/null || true
pkill -f "run_server.py" 2>/dev/null || true
fuser -k 5000/tcp 2>/dev/null || true
sleep 5

MORTAL_CFG="$CFG" \
EVAL_LABEL="$EVAL_LABEL" \
EVAL_CHECKPOINT="$CKPT" \
RUN_DIR="$RUN_DIR" \
MORTAL_FOREGROUND=1 \
bash "$REPO/freeparlor/scripts/run_eval_ppo_smoke_sanity.sh"

echo "eval done: $RUN_DIR/logs/eval_sanity_${EVAL_LABEL}.log"
