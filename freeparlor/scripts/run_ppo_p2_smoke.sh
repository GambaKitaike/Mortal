#!/usr/bin/env bash
# PPO P2 smoke — server×1 / trainer×1 / client×3
# Launches in tmux unless MORTAL_FOREGROUND=1.
set -euo pipefail

RUN_DIR="${RUN_DIR:-/home/gamba/mahjong/runs/ppo/smoke_p2}"
CFG="$RUN_DIR/config.toml"
LOG_DIR="$RUN_DIR/logs"
NUM_CLIENTS="${NUM_CLIENTS:-3}"
REPO="/home/gamba/mahjong/Mortal"
PPO_CONFIG="${PPO_CONFIG:-$REPO/freeparlor/configs/ppo_p2_smoke.toml}"
CONFIG_TAG="${CONFIG_TAG:-smoke_p2}"
TMUX_SESSION="${TMUX_SESSION:-ppo_p2_smoke}"

if [[ -z "${MORTAL_FOREGROUND:-}" ]]; then
  if ! command -v tmux >/dev/null; then
    echo "ERROR: tmux required (or set MORTAL_FOREGROUND=1)"
    exit 1
  fi
  tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
  tmux new-session -d -s "$TMUX_SESSION" \
    "MORTAL_FOREGROUND=1 NUM_CLIENTS=$NUM_CLIENTS TMUX_SESSION=$TMUX_SESSION RUN_DIR=$RUN_DIR PPO_CONFIG=$PPO_CONFIG CONFIG_TAG=$CONFIG_TAG bash $0; echo exit=\$?; exec bash"
  echo "Started in tmux session: $TMUX_SESSION"
  echo "  attach: tmux attach -t $TMUX_SESSION"
  echo "  log:    tail -f $LOG_DIR/trainer.log"
  exit 0
fi

export MORTAL_CFG="$CFG"
export PYTHONPATH="$REPO/mortal"

mkdir -p "$LOG_DIR" "$RUN_DIR/train_play"/{client0,client1,client2}
rm -f "$LOG_DIR"/*.log

CLIENT_PIDS=()
cleanup() {
  echo "Cleanup..."
  [[ -n "${TRAINER_WATCHDOG_PID:-}" ]] && kill "$TRAINER_WATCHDOG_PID" 2>/dev/null || true
  for pid in "${CLIENT_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  pkill -f "python /home/gamba/mahjong/runs/run_train_ppo.py" 2>/dev/null || true
  sleep 2
  pkill -f "${CONFIG_TAG}/config.toml" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Stopping stale processes ==="
pkill -f "run_train_ppo.py" 2>/dev/null || true
pkill -f "run_client.py" 2>/dev/null || true
pkill -f "run_server.py" 2>/dev/null || true
pkill -f "${CONFIG_TAG}/config.toml" 2>/dev/null || true
pkill -f "eval_ppo_smoke_sanity.py" 2>/dev/null || true
fuser -k 5000/tcp 2>/dev/null || true
sleep 3
if ss -tlnp | grep -q 5000; then
  echo "ERROR: port 5000 still in use"
  ss -tlnp | grep 5000
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "ERROR: GPU compute processes still running (single-stream rule)"
  nvidia-smi
  exit 1
fi
echo "port 5000 clear, GPU idle"

echo "=== Setup run dir ==="
mkdir -p "$RUN_DIR"/{tb,test_play,buffer,drain}
cp "$PPO_CONFIG" "$CFG"
rm -rf "$RUN_DIR/buffer" "$RUN_DIR/drain"
mkdir -p "$RUN_DIR/buffer" "$RUN_DIR/drain"

conda run -n mortal python -c "
import torch
from datetime import datetime
from pathlib import Path
RUN=Path('$RUN_DIR')
src=Path('/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth')
ws=RUN/'mortal_init.pth'
if not ws.exists():
    s=torch.load(src, weights_only=True, map_location='cpu')
    s['steps']=0
    s['timestamp']=datetime.now().timestamp()
    s['config']['control']['online']=False
    for k in ('optimizer','scheduler'): s.pop(k,None)
    torch.save(s, ws)
# PPO trainer loads init_checkpoint from config; do not overwrite with DQN-format mortal.pth
(Path('$RUN_DIR')/'mortal.pth').unlink(missing_ok=True)
print('init checkpoint OK (trainer uses ppo.init_checkpoint)')
"

echo "=== Starting server ==="
PYTHONUNBUFFERED=1 conda run --no-capture-output -n mortal python /home/gamba/mahjong/runs/run_server.py \
  > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 60); do
  sleep 1
  grep -q "listening on" "$LOG_DIR/server.log" 2>/dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || { cat "$LOG_DIR/server.log"; exit 1; }
done
grep -q "listening on" "$LOG_DIR/server.log" || { cat "$LOG_DIR/server.log"; exit 1; }
echo "server OK pid=$SERVER_PID"

echo "=== Starting PPO trainer (watchdog) ==="
start_trainer() {
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n mortal python /home/gamba/mahjong/runs/run_train_ppo.py \
    >> "$LOG_DIR/trainer.log" 2>&1
}
trainer_watchdog() {
  while true; do
    start_trainer
    code=$?
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    echo "$(date -Iseconds) trainer exited code=$code, restarting in 5s" >> "$LOG_DIR/trainer_watchdog.log"
    sleep 5
  done
}
trainer_watchdog &
TRAINER_WATCHDOG_PID=$!
sleep 20

echo "=== Starting $NUM_CLIENTS clients ==="
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  TRAIN_PLAY_PROFILE="client${i}" \
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n mortal python /home/gamba/mahjong/runs/run_client.py \
    > "$LOG_DIR/client${i}.log" 2>&1 &
  CLIENT_PIDS+=($!)
  sleep 3
done

echo "=== Waiting for max_steps (400) ==="
DEADLINE=$(( $(date +%s) + 14400 ))
while (( $(date +%s) < DEADLINE )); do
  steps=$(grep -oP 'ppo step \K[0-9]+' "$LOG_DIR/trainer.log" 2>/dev/null | tail -1 || true)
  steps=${steps:-0}
  chip_err=$(grep -h 'online chip resolution failed' "$LOG_DIR"/client*.log 2>/dev/null | wc -l | tr -d ' ' || true)
  chip_err=${chip_err:-0}
  nan_err=$(grep -ciE 'non-finite|FloatingPointError' "$LOG_DIR/trainer.log" 2>/dev/null | tr -d ' ' || true)
  nan_err=${nan_err:-0}
  if (( chip_err > 0 )); then
    echo "FATAL: chip resolution errors=$chip_err"
    exit 2
  fi
  if (( nan_err > 0 )); then
    echo "FATAL: NaN detected"
    exit 3
  fi
  if (( steps >= 400 )); then
    echo "reached step $steps"
    break
  fi
  sleep 30
  echo "  steps=$steps/400"
done

echo "=== Final log tail ==="
grep 'ppo step' "$LOG_DIR/trainer.log" | tail -5 || true
echo "mismatch: $(grep -h 'trajectory step count mismatch' "$LOG_DIR"/client*.log 2>/dev/null | wc -l)"
echo "chip errors: $(grep -h 'online chip resolution failed' "$LOG_DIR"/client*.log 2>/dev/null | wc -l)"
echo "=== Done ==="
