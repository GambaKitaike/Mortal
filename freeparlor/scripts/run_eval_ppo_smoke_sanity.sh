#!/usr/bin/env bash
# PPO P2 smoke — standalone 100-hanchan self-play sanity eval (no server/client).
set -euo pipefail

RUN_DIR="/home/gamba/mahjong/runs/ppo/smoke_p2"
CFG="$RUN_DIR/config.toml"
LOG_DIR="$RUN_DIR/logs"
EVAL_LOG="$LOG_DIR/eval_sanity.log"
REPO="/home/gamba/mahjong/Mortal"
SCRIPT="$REPO/freeparlor/scripts/eval_ppo_smoke_sanity.py"

export MORTAL_CFG="$CFG"
export PYTHONPATH="$REPO/mortal"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"
: > "$EVAL_LOG"

{
echo "=== Stopping stale online processes (eval pre-check) ==="
pkill -f "run_train_ppo.py" 2>/dev/null || true
pkill -f "run_client.py" 2>/dev/null || true
pkill -f "run_server.py" 2>/dev/null || true
pkill -f "smoke_p2/config.toml" 2>/dev/null || true
fuser -k 5000/tcp 2>/dev/null || true
sleep 3

if ss -tlnp 2>/dev/null | grep -q 5000; then
  echo "ERROR: port 5000 still in use"
  ss -tlnp | grep 5000
  exit 1
fi
echo "port 5000 clear"
echo "=== Starting eval_sanity ==="
echo "$(date -Iseconds) eval_sanity start"
} | tee "$EVAL_LOG"

conda run --no-capture-output -n mortal python "$SCRIPT"
code=$?
echo "$(date -Iseconds) eval_sanity exit code=$code" | tee -a "$EVAL_LOG"
exit $code
