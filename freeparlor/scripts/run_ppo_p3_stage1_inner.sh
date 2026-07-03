#!/usr/bin/env bash
# Inner launcher for P3 Stage1 (called from run_ppo_p3_stage1.sh)
set -euo pipefail

RUN_DIR="${RUN_DIR:?}"
CFG="$RUN_DIR/config.toml"
LOG_DIR="$RUN_DIR/logs"
NUM_CLIENTS="${NUM_CLIENTS:-3}"
REPO="/home/gamba/mahjong/Mortal"
PPO_CONFIG="${PPO_CONFIG:?}"
CONFIG_TAG="${CONFIG_TAG:?}"
TMUX_SESSION="${TMUX_SESSION:?}"
MAX_STEPS="${MAX_STEPS:-16000}"

if [[ -z "${MORTAL_FOREGROUND:-}" ]]; then
  if ! command -v tmux >/dev/null; then
    echo "ERROR: tmux required (or set MORTAL_FOREGROUND=1)"
    exit 1
  fi
  tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
  tmux new-session -d -s "$TMUX_SESSION" \
    "MORTAL_FOREGROUND=1 NUM_CLIENTS=$NUM_CLIENTS TMUX_SESSION=$TMUX_SESSION RUN_DIR=$RUN_DIR PPO_CONFIG=$PPO_CONFIG CONFIG_TAG=$CONFIG_TAG MAX_STEPS=$MAX_STEPS bash $0; echo exit=\$?; exec bash"
  echo "Started in tmux session: $TMUX_SESSION"
  echo "  attach: tmux attach -t $TMUX_SESSION"
  echo "  log:    tail -f $LOG_DIR/trainer.log"
  echo "  mem:    tail -f $LOG_DIR/mem_monitor.log"
  exit 0
fi

export MORTAL_CFG="$CFG"
export PYTHONPATH="$REPO/mortal"

mkdir -p "$LOG_DIR" "$RUN_DIR/train_play"/{client0,client1,client2} "$RUN_DIR/checkpoints"
rm -f "$LOG_DIR"/*.log

CLIENT_PIDS=()
cleanup() {
  echo "Cleanup..."
  [[ -n "${MEM_MONITOR_PID:-}" ]] && kill "$MEM_MONITOR_PID" 2>/dev/null || true
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

echo "=== Pre-flight: libriichi import check ==="
PYTHONPATH="$REPO/mortal" conda run -n mortal python -c "import libriichi.arena" \
  || { echo "ERROR: libriichi.so broken — rebuild with PYO3_PYTHON=\$CONDA_PREFIX/bin/python"; exit 1; }

echo "=== Pre-flight verify (checks 10/11) ==="
conda run -n mortal python "$REPO/freeparlor/scripts/verify_ppo_p1.py" \
  --checkpoint /home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth \
  --grp-state /home/gamba/mahjong/runs/grp.pth \
  2>&1 | tee "$LOG_DIR/verify_p1.log" | tail -20

echo "=== Setup run dir ==="
mkdir -p "$RUN_DIR"/{tb,test_play,buffer,drain,checkpoints}
cp "$PPO_CONFIG" "$CFG"
rm -rf "$RUN_DIR/buffer" "$RUN_DIR/drain"
mkdir -p "$RUN_DIR/buffer" "$RUN_DIR/drain"

if [[ -f "$RUN_DIR/mortal.pth" ]]; then
  echo "resume: keeping existing mortal.pth"
else
conda run -n mortal python -c "
import torch
from datetime import datetime
from pathlib import Path
from model import ActorCritic, Brain, load_ppo_from_mortal_checkpoint

RUN=Path('$RUN_DIR')
src=Path('/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth')
ws=RUN/'mortal_init.pth'
ckpt_dir=RUN/'checkpoints'
ckpt_dir.mkdir(parents=True, exist_ok=True)
if not ws.exists():
    s=torch.load(src, weights_only=True, map_location='cpu')
    s['steps']=0
    s['timestamp']=datetime.now().timestamp()
    s['config']['control']['online']=False
    for k in ('optimizer','scheduler'): s.pop(k,None)
    torch.save(s, ws)
init0=ckpt_dir/'step_000000.pth'
if not init0.exists():
    s=torch.load(src, weights_only=True, map_location='cpu')
    version=s['config']['control'].get('version', 4)
    mortal=Brain(version=version, **s['config']['resnet'])
    mortal.load_state_dict(s['mortal'])
    ac=ActorCritic(version=version, tau=1.0)
    load_ppo_from_mortal_checkpoint(ac, str(src), map_location='cpu')
    torch.save({
        'mortal': mortal.state_dict(),
        'actor_critic': ac.state_dict(),
        'steps': 0,
        'timestamp': datetime.now().timestamp(),
        'config': s['config'],
    }, init0)
(Path('$RUN_DIR')/'mortal.pth').unlink(missing_ok=True)
print('init checkpoint OK (step_000000.pth for opponent pool)')
"
fi

mem_monitor() {
  local end=$(( $(date +%s) + 3600 ))
  while (( $(date +%s) < end )); do
    {
      echo "=== $(date -Iseconds) ==="
      free -h | head -3
      nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
    } >> "$LOG_DIR/mem_monitor.log"
    sleep 300
  done
}
mem_monitor &
MEM_MONITOR_PID=$!

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

echo "=== Starting $NUM_CLIENTS clients (watchdog) ==="
start_client() {
  local i=$1
  TRAIN_PLAY_PROFILE="client${i}" \
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n mortal python /home/gamba/mahjong/runs/run_client.py \
    >> "$LOG_DIR/client${i}.log" 2>&1
}
client_watchdog() {
  local i=$1
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    start_client "$i"
    code=$?
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    echo "$(date -Iseconds) client${i} exited code=$code, restarting in 5s" >> "$LOG_DIR/client${i}_watchdog.log"
    sleep 5
  done
}
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  client_watchdog "$i" &
  CLIENT_PIDS+=($!)
  sleep 3
done

echo "=== P3 Stage1 running (max_steps=$MAX_STEPS, ~19h) ==="
echo "Config: $CFG"
DEADLINE=$(( $(date +%s) + 86400 ))
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
  if (( steps >= MAX_STEPS )); then
    echo "reached step $steps"
    break
  fi
  sleep 60
  echo "  steps=$steps/$MAX_STEPS"
done

echo "=== Final log tail ==="
grep 'ppo step' "$LOG_DIR/trainer.log" | tail -5 || true
echo "mismatch: $(grep -h 'trajectory step count mismatch' "$LOG_DIR"/client*.log 2>/dev/null | wc -l)"
echo "chip errors: $(grep -h 'online chip resolution failed' "$LOG_DIR"/client*.log 2>/dev/null | wc -l)"
echo "=== Done (eval at checkpoints: run_eval separately) ==="
