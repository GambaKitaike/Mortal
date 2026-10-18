#!/usr/bin/env bash
# Reproduce trajectory skip/mismatch with extended logging (client×1, ~50 games).
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
RUN_DIR="${RUN_DIR:-/home/gamba/mahjong/runs/ppo/ppo_p3_mismatch_repro_$(date +%Y%m%d_%H%M%S)}"
CFG="$RUN_DIR/config.toml"
LOG_DIR="$RUN_DIR/logs"
NUM_CLIENTS=1
TMUX_SESSION="${TMUX_SESSION:-ppo_p3_mismatch_repro}"
CKPT="${CKPT:-/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth}"
GRP="${GRP:-/home/gamba/mahjong/runs/grp.pth}"
# 13 seed sets × 4 splits ≈ 52 games
SEED_COUNT="${SEED_COUNT:-13}"

if [[ -z "${MORTAL_FOREGROUND:-}" ]]; then
  tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
  tmux new-session -d -s "$TMUX_SESSION" \
    "MORTAL_FOREGROUND=1 TMUX_SESSION=$TMUX_SESSION RUN_DIR=$RUN_DIR SEED_COUNT=$SEED_COUNT bash $0; echo exit=\$?; exec bash"
  echo "Started mismatch repro: tmux attach -t $TMUX_SESSION"
  echo "  run_dir: $RUN_DIR"
  exit 0
fi

export MORTAL_CFG="$CFG"
export PYTHONPATH="$REPO/mortal"
mkdir -p "$LOG_DIR" "$RUN_DIR/train_play/client0"

CLIENT_PIDS=()
cleanup() {
  echo "Cleanup..."
  for pid in "${CLIENT_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  pkill -f "${RUN_DIR}/config.toml" 2>/dev/null || true
  fuser -k 5000/tcp 2>/dev/null || true
}
trap cleanup EXIT

pkill -f "run_train_ppo.py" 2>/dev/null || true
pkill -f "run_client.py" 2>/dev/null || true
pkill -f "run_server.py" 2>/dev/null || true
fuser -k 5000/tcp 2>/dev/null || true
sleep 2

mkdir -p "$RUN_DIR"/{tb,buffer,drain,checkpoints}
cat > "$CFG" <<EOF
[control]
version = 4
online = true
device = 'cuda:0'
enable_cudnn_benchmark = false
enable_amp = true
enable_compile = false
batch_size = 128
save_every = 100000
test_every = 100000
submit_every = 50

[ppo]
enabled = true
tau_init = 1.0
lr = 2e-5
max_steps = 50
init_checkpoint = '$CKPT'

[train_play.client0]
games = $((SEED_COUNT * 4))
log_dir = '$RUN_DIR/train_play/client0'
repeats = 1

[test_play]
games = 4
log_dir = '$RUN_DIR/test_play'
self_play = true

[env]
pts = [35, 5, -15, -25]
alpha = 1.0
gamma_pt = 1.0
beta = 1.0
chip_value = 5.0

[resnet]
conv_channels = 192
num_blocks = 40

[baseline.train]
device = 'cuda:0'
enable_compile = false
state_file = '/home/gamba/mahjong/runs/grp_baseline.pth'

[baseline.test]
device = 'cuda:0'
enable_compile = false
state_file = '/home/gamba/mahjong/runs/grp_baseline.pth'

[online]
history_window = 10
enable_compile = false

[online.remote]
host = '127.0.0.1'
port = 5000

[online.server]
buffer_dir = '$RUN_DIR/buffer'
drain_dir = '$RUN_DIR/drain'
capacity = 400

[grp]
state_file = '$GRP'

[grp.network]
hidden_size = 64
num_layers = 2

[opponent_pool]
enabled = false
EOF

# init checkpoint for trainer
conda run -n mortal python -c "
import torch
from datetime import datetime
from pathlib import Path
RUN=Path('$RUN_DIR')
src=Path('$CKPT')
ws=RUN/'mortal_init.pth'
s=torch.load(src, weights_only=True, map_location='cpu')
s['steps']=0
s['timestamp']=datetime.now().timestamp()
s['config']['control']['online']=True
for k in ('optimizer','scheduler'): s.pop(k,None)
torch.save(s, ws)
(RUN/'mortal.pth').unlink(missing_ok=True)
"

PYTHONUNBUFFERED=1 conda run --no-capture-output -n mortal python /home/gamba/mahjong/runs/run_server.py \
  > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 30); do
  sleep 1
  grep -q "listening on" "$LOG_DIR/server.log" 2>/dev/null && break
done

PYTHONUNBUFFERED=1 conda run --no-capture-output -n mortal python /home/gamba/mahjong/runs/run_train_ppo.py \
  >> "$LOG_DIR/trainer.log" 2>&1 &
TRAINER_PID=$!

TRAIN_PLAY_PROFILE=client0 \
PYTHONUNBUFFERED=1 conda run --no-capture-output -n mortal python /home/gamba/mahjong/runs/run_client.py \
  > "$LOG_DIR/client0.log" 2>&1 &
CLIENT_PIDS+=($!)

DEADLINE=$(( $(date +%s) + 3600 ))
while (( $(date +%s) < DEADLINE )); do
  submits=$(grep -c 'logs have been submitted' "$LOG_DIR/client0.log" 2>/dev/null | head -1 || echo 0)
  submits=${submits:-0}
  if (( submits >= 3 )); then break; fi
  sleep 15
done

echo "=== Mismatch repro summary ==="
echo "run_dir: $RUN_DIR"
echo "submits: $(grep -c 'logs have been submitted' "$LOG_DIR/client0.log" || true)"
echo "mismatch: $(grep -c 'step count mismatch' "$LOG_DIR/client0.log" || true)"
echo "key_missing: $(grep -c 'game key missing' "$LOG_DIR/client0.log" || true)"
echo "orphan: $(grep -c 'trajectory orphan' "$LOG_DIR/client0.log" || true)"
echo "--- delta distribution ---"
grep 'step count mismatch' "$LOG_DIR/client0.log" | grep -oP 'delta=\K-?[0-9]+' | sort | uniq -c || true
echo "--- sample lines ---"
grep 'step count mismatch\|game key missing' "$LOG_DIR/client0.log" | head -10 || true

kill "$TRAINER_PID" 2>/dev/null || true
echo "=== repro done ==="
