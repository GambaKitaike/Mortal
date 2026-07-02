#!/usr/bin/env bash
# Reproduce first-P2 cursor-based mismatch (client.py @ 1a64262). Restores client.py on exit.
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
RUN_DIR="/home/gamba/mahjong/runs/ppo/smoke_p2_forensic"
CFG="$RUN_DIR/config.toml"
LOG_DIR="$RUN_DIR/logs"
NUM_CLIENTS=3
TMUX_SESSION="${TMUX_SESSION:-ppo_p2_forensic}"
CLIENT_BACKUP="$REPO/mortal/client.py.bak_forensic"

if [[ -z "${MORTAL_FOREGROUND:-}" ]]; then
  tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
  tmux new-session -d -s "$TMUX_SESSION" \
    "MORTAL_FOREGROUND=1 TMUX_SESSION=$TMUX_SESSION bash $0; echo exit=\$?; exec bash"
  echo "Started forensic repro: tmux attach -t $TMUX_SESSION"
  exit 0
fi

restore_client() {
  if [[ -f "$CLIENT_BACKUP" ]]; then
    mv "$CLIENT_BACKUP" "$REPO/mortal/client.py"
    echo "restored mortal/client.py"
  fi
}
trap restore_client EXIT

cp "$REPO/mortal/client.py" "$CLIENT_BACKUP"
git show 1a64262:mortal/client.py > "$REPO/mortal/client.py"
echo "using client.py from commit 1a64262 (cursor-based pending)"

export MORTAL_CFG="$CFG"
export PYTHONPATH="$REPO/mortal"
mkdir -p "$LOG_DIR" "$RUN_DIR/train_play"/{client0,client1,client2}
rm -f "$LOG_DIR"/*.log

CLIENT_PIDS=()
cleanup() {
  echo "Cleanup..."
  for pid in "${CLIENT_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  [[ -n "${TRAINER_WATCHDOG_PID:-}" ]] && kill "$TRAINER_WATCHDOG_PID" 2>/dev/null || true
  pkill -f "smoke_p2_forensic/config.toml" 2>/dev/null || true
  fuser -k 5000/tcp 2>/dev/null || true
}
trap cleanup EXIT
trap restore_client EXIT

pkill -f "run_train_ppo.py" 2>/dev/null || true
pkill -f "run_client.py" 2>/dev/null || true
pkill -f "run_server.py" 2>/dev/null || true
fuser -k 5000/tcp 2>/dev/null || true
sleep 3

mkdir -p "$RUN_DIR"/{tb,test_play,buffer,drain}
cp "$REPO/freeparlor/configs/ppo_p2_smoke.toml" "$CFG"
sed -i 's|/ppo/smoke_p2/|/ppo/smoke_p2_forensic/|g' "$CFG"
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
(Path('$RUN_DIR')/'mortal.pth').unlink(missing_ok=True)
"

PYTHONUNBUFFERED=1 conda run --no-capture-output -n mortal python /home/gamba/mahjong/runs/run_server.py \
  > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 60); do
  sleep 1
  grep -q "listening on" "$LOG_DIR/server.log" 2>/dev/null && break
done

start_trainer() {
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n mortal python /home/gamba/mahjong/runs/run_train_ppo.py \
    >> "$LOG_DIR/trainer.log" 2>&1
}
trainer_watchdog() {
  while true; do
    start_trainer
    code=$?
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 5
  done
}
trainer_watchdog &
TRAINER_WATCHDOG_PID=$!
sleep 20

for i in $(seq 0 2); do
  TRAIN_PLAY_PROFILE="client${i}" \
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n mortal python /home/gamba/mahjong/runs/run_client.py \
    > "$LOG_DIR/client${i}.log" 2>&1 &
  CLIENT_PIDS+=($!)
  sleep 3
done

DEADLINE=$(( $(date +%s) + 14400 ))
while (( $(date +%s) < DEADLINE )); do
  steps=$(grep -oP 'ppo step \K[0-9]+' "$LOG_DIR/trainer.log" 2>/dev/null | tail -1 || true)
  steps=${steps:-0}
  if (( steps >= 400 )); then break; fi
  sleep 30
done

echo "mismatch total: $(grep -hc 'trajectory step count mismatch' "$LOG_DIR"/client*.log | awk '{s+=$1} END{print s+0}')"
for f in "$LOG_DIR"/client*.log; do
  echo "$(basename "$f"): $(grep -c 'trajectory step count mismatch' "$f" || true)"
done
conda run -n mortal python "$REPO/freeparlor/scripts/parse_p2_mismatch_forensic.py" "$LOG_DIR"
echo "=== forensic done ==="
