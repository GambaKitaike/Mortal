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
  # Reap the trainer tree BEFORE the server (2026-07-28). train_ppo.py's own
  # main() is a supervisor loop (`while True: Popen(child); wait()`) inherited
  # from the upstream online-DQN trainer, so ~3s after a normal completion it
  # spawns ONE more child. That child re-loads the checkpoint, hits
  # `steps >= max_steps` and exits without training a single step (verified in
  # train_ppo.py:632-639 and in the Arm K log), but it has two side effects:
  #   - it runs as `<python> .../mortal/train_ppo.py`, a cmdline the
  #     run_train_ppo.py pattern below does NOT match, so it survives cleanup
  #     as an orphan holding the GPU;
  #   - if the server is killed first it dies inside submit_param() with a
  #     ConnectionRefusedError traceback, which makes a clean 16000-step finish
  #     look like a crash to anyone reading trainer.log afterwards
  #     (observed: anchor_k_20260727_000805, diagnosed 2026-07-28).
  # Killing the whole trainer tree first removes both symptoms.
  pkill -f "python /home/gamba/mahjong/runs/run_train_ppo.py" 2>/dev/null || true
  pkill -f "python .*mortal/train_ppo\.py" 2>/dev/null || true
  for pid in "${CLIENT_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  # SERVER_PID / CLIENT_PIDS point at the `conda run` / watchdog-subshell
  # wrappers, which do NOT forward SIGTERM to the child python. Reap the real
  # python processes by pattern too, mirroring the trainer line — otherwise
  # server/client are orphaned and left alive (2026-07-10: 8h residual leak).
  # Single-stream rule (one training run at a time) makes broad pattern kill
  # safe; DRCA uses drca_run_probe.py and is unaffected by these patterns.
  pkill -f "python /home/gamba/mahjong/runs/run_server.py" 2>/dev/null || true
  pkill -f "python /home/gamba/mahjong/runs/run_client.py" 2>/dev/null || true
  sleep 2
  pkill -f "${CONFIG_TAG}/config.toml" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Stopping stale processes ==="
pkill -f "run_train_ppo.py" 2>/dev/null || true
# The trainer's own supervisor loop runs the real work as a grandchild whose
# cmdline is `<python> .../mortal/train_ppo.py` — not matched by the pattern
# above. Without this line a post-completion orphan from a previous run would
# only be caught by the GPU-idle gate below (loud, but a manual stop).
pkill -f "python .*mortal/train_ppo\.py" 2>/dev/null || true
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

"$REPO/freeparlor/scripts/preflight_libriichi.sh" "$REPO"

echo "=== Pre-flight verify (checks 1-16) ==="
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
  (
    set +e
    local restarts=0
    local window_start
    window_start=$(date +%s)
    while true; do
      start_trainer
      code=$?
      # exit 0 = trainer reached max_steps (normal completion). Do NOT restart:
      # the server is still alive so the old `kill -0 SERVER_PID` gate would
      # otherwise treat completion as a crash and spawn a benign double-start
      # (observed in Stage3 run, 2026-07-13).
      if (( code == 0 )); then
        echo "$(date -Iseconds) trainer completed normally (exit 0), watchdog stopping (no restart)" \
          >> "$LOG_DIR/trainer_watchdog.log"
        break
      fi
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
      now=$(date +%s)
      if (( now - window_start >= 3600 )); then
        window_start=$now
        restarts=0
      fi
      restarts=$((restarts + 1))
      if (( restarts > 3 )); then
        echo "$(date -Iseconds) ERROR trainer watchdog: >3 restarts/hour, stopping run" \
          | tee -a "$LOG_DIR/trainer_watchdog.log" "$LOG_DIR/monitor.log"
        exit 6
      fi
      echo "$(date -Iseconds) trainer exited code=$code, restart $restarts/3 this hour" \
        >> "$LOG_DIR/trainer_watchdog.log"
      sleep 5
    done
  )
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
  (
    set +e
    local restarts=0
    local window_start
    window_start=$(date +%s)
    while kill -0 "$SERVER_PID" 2>/dev/null; do
      start_client "$i"
      code=$?
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
      now=$(date +%s)
      if (( now - window_start >= 3600 )); then
        window_start=$now
        restarts=0
      fi
      restarts=$((restarts + 1))
      if (( restarts > 3 )); then
        echo "$(date -Iseconds) ERROR client${i} watchdog: >3 restarts/hour, stopping run" \
          | tee -a "$LOG_DIR/client${i}_watchdog.log" "$LOG_DIR/monitor.log"
        exit 6
      fi
      echo "$(date -Iseconds) client${i} exited code=$code, restart $restarts/3 this hour" \
        >> "$LOG_DIR/client${i}_watchdog.log"
      sleep 5
    done
  )
}
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  client_watchdog "$i" &
  CLIENT_PIDS+=($!)
  sleep 3
done

count_alive_clients() {
  ALIVE_CLIENTS=$(pgrep -fc "python /home/gamba/mahjong/runs/run_client.py" 2>/dev/null || true)
  ALIVE_CLIENTS=${ALIVE_CLIENTS:-0}
}

echo "=== P3 Stage1 running (max_steps=$MAX_STEPS, ~19h) ==="
echo "Config: $CFG"
count_monitor_metrics() {
  # LEGACY TRIPWIRE — structurally always 0 (2026-07-28). No code in this repo
  # emits 'trajectory step count mismatch'; the string survives only here, in
  # verify_ppo_p1.py's counter and in docs (the emitter was lost during the P2
  # rework). It is kept so that a restored emitter would be caught immediately,
  # but it must NOT be read as evidence that trajectory joining is healthy —
  # that is what MON_KEYMISS / MON_ORPHAN below are for.
  MON_MISMATCH=$(grep -h 'trajectory step count mismatch' "$LOG_DIR"/client*.log 2>/dev/null | wc -l | tr -d ' ' || true)
  MON_MISMATCH=${MON_MISMATCH:-0}
  # LIVE data-loss signals (client.py:108 / :184). These are what actually fire
  # when the trajectory join breaks: 'game key missing' DISCARDS A WHOLE KYOKU,
  # 'orphan steps' means recorded steps had no matching game log. verify_ppo_p1
  # check(13) already asserts both == 0 before launch; until 2026-07-28 nothing
  # watched them DURING a run. Empirical baseline over three completed 16k runs
  # (stage3 / anchor_c / anchor_k): 0 occurrences each, while the non-fatal
  # 'loader size delta' fired 8,759-12,137 times — so a non-zero count here is
  # a real integrity break, not noise, and gets the same immediate-stop
  # treatment as the other data-integrity signals (CLAUDE.md workflow rule).
  MON_KEYMISS=$(grep -h 'trajectory game key missing' "$LOG_DIR"/client*.log 2>/dev/null | wc -l | tr -d ' ' || true)
  MON_KEYMISS=${MON_KEYMISS:-0}
  MON_ORPHAN=$(grep -h 'trajectory orphan steps' "$LOG_DIR"/client*.log 2>/dev/null | wc -l | tr -d ' ' || true)
  MON_ORPHAN=${MON_ORPHAN:-0}
  MON_FALLBACK=$(grep -hE 'illegal_action_fallback_count=[1-9]' "$LOG_DIR"/client*.log 2>/dev/null | wc -l | tr -d ' ' || true)
  MON_FALLBACK=${MON_FALLBACK:-0}
  MON_CHIP=$(grep -h 'online chip resolution failed' "$LOG_DIR"/client*.log 2>/dev/null | wc -l | tr -d ' ' || true)
  MON_CHIP=${MON_CHIP:-0}
  MON_LOADER_DELTA=$(grep -h 'trajectory loader size delta' "$LOG_DIR"/client*.log 2>/dev/null | wc -l | tr -d ' ' || true)
  MON_LOADER_DELTA=${MON_LOADER_DELTA:-0}
}

# Monitor wall-clock budget. 2026-07-26 incident: anchor Arm C was the first run
# slower than the old hard-coded 24h (595-736 step/h vs ~700-1000 for Stage1-3;
# the anchor pool loads a different checkpoint on 25% of draws). The loop simply
# fell out of the `while` on deadline expiry and ran the SAME shutdown path as a
# normal finish (Final log tail -> Done -> Cleanup -> exit 0), SIGTERM-ing the
# run at step 14100/16000 while reporting success. Deadline expiry is now a loud
# failure (exit 8) and never reaches the completion path.
MONITOR_HOURS="${MONITOR_HOURS:-48}"
DEADLINE=$(( $(date +%s) + MONITOR_HOURS * 3600 ))
COMPLETED=0
ALIVE_CLIENTS=$NUM_CLIENTS
while (( $(date +%s) < DEADLINE )); do
  steps=$(grep -oP 'ppo step \K[0-9]+' "$LOG_DIR/trainer.log" 2>/dev/null | tail -1 || true)
  steps=${steps:-0}
  count_monitor_metrics
  count_alive_clients
  nan_err=$(grep -ciE 'non-finite|FloatingPointError' "$LOG_DIR/trainer.log" 2>/dev/null | tr -d ' ' || true)
  nan_err=${nan_err:-0}
  if ! kill -0 "$TRAINER_WATCHDOG_PID" 2>/dev/null; then
    # Watchdog now exits on trainer normal completion (exit 0). If we have
    # reached max_steps, that is a clean finish, not a crash → break to the
    # normal shutdown path instead of erroring out with exit 6.
    if (( steps >= MAX_STEPS )); then
      echo "trainer watchdog exited after reaching step $steps (normal completion)" | tee -a "$LOG_DIR/monitor.log"
      COMPLETED=1
      break
    fi
    echo "ERROR: trainer watchdog exited (see $LOG_DIR/trainer_watchdog.log)" | tee -a "$LOG_DIR/monitor.log"
    exit 6
  fi
  for pid in "${CLIENT_PIDS[@]:-}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: client watchdog pid=$pid exited (see client*_watchdog.log)" | tee -a "$LOG_DIR/monitor.log"
      exit 6
    fi
  done
  if (( ALIVE_CLIENTS < NUM_CLIENTS )); then
    echo "ERROR: alive clients ${ALIVE_CLIENTS}/${NUM_CLIENTS}" | tee -a "$LOG_DIR/monitor.log"
    exit 7
  fi
  if (( MON_MISMATCH > 0 )); then
    echo "FATAL: trajectory step count mismatch=$MON_MISMATCH" | tee -a "$LOG_DIR/monitor.log"
    exit 4
  fi
  if (( MON_KEYMISS > 0 )); then
    echo "FATAL: trajectory game key missing=$MON_KEYMISS (whole kyoku discarded)" \
      | tee -a "$LOG_DIR/monitor.log"
    exit 9
  fi
  if (( MON_ORPHAN > 0 )); then
    echo "FATAL: trajectory orphan steps=$MON_ORPHAN (recorded steps with no game log)" \
      | tee -a "$LOG_DIR/monitor.log"
    exit 10
  fi
  if (( MON_FALLBACK > 0 )); then
    echo "FATAL: illegal_action_fallback_count non-zero sessions=$MON_FALLBACK"
    exit 5
  fi
  if (( MON_CHIP > 0 )); then
    echo "FATAL: chip resolution errors=$MON_CHIP"
    exit 2
  fi
  if (( MON_LOADER_DELTA > 0 )); then
    echo "NOTICE: loader size delta (INFO) count=$MON_LOADER_DELTA (non-fatal early signal)"
  fi
  if (( nan_err > 0 )); then
    echo "FATAL: NaN detected"
    exit 3
  fi
  if (( steps >= MAX_STEPS )); then
    echo "reached step $steps"
    COMPLETED=1
    break
  fi
  sleep 60
  echo "  steps=$steps/$MAX_STEPS alive_clients=$ALIVE_CLIENTS/$NUM_CLIENTS monitor:" \
    "keymiss=$MON_KEYMISS orphan=$MON_ORPHAN fallback=$MON_FALLBACK chip=$MON_CHIP" \
    "loader_delta=$MON_LOADER_DELTA mismatch=$MON_MISMATCH(legacy:no-emitter)"
done

if (( COMPLETED == 1 )); then
  # Give the trainer time to finish its END-OF-RUN work before cleanup SIGTERMs it.
  # The monitor breaks the moment `ppo step MAX` appears in the log, but at that
  # point the trainer still has to flush stats and write its final checkpoint(s).
  # Cleanup used to leave the real worker (`<python> .../mortal/train_ppo.py`)
  # unmatched by its pkill patterns, so the write always completed by luck; once
  # that gap was closed (2026-07-28) the SIGTERM started landing mid-write and
  # truncated a checkpoint (early_probe_20260728_202533 lost step_002000.pth at
  # 108.8MB of 130.7MB). Wait for the save to be confirmed in the log instead.
  final_ckpt="step_$(printf '%06d' "$MAX_STEPS").pth"
  echo "waiting for the trainer to write $final_ckpt before cleanup ..."
  for _ in $(seq 1 "${FINAL_SAVE_WAIT_TRIES:-60}"); do
    grep -q "saved numbered checkpoint: .*$final_ckpt" "$LOG_DIR/trainer.log" 2>/dev/null && break
    sleep 5
  done
  if grep -q "saved numbered checkpoint: .*$final_ckpt" "$LOG_DIR/trainer.log" 2>/dev/null; then
    echo "final checkpoint confirmed: $final_ckpt"
    sleep 5   # let the file handle flush/close before any SIGTERM
  else
    echo "WARNING: $final_ckpt was not confirmed within the grace period; the last" \
      "checkpoint may be truncated — verify with torch.load before using it" \
      | tee -a "$LOG_DIR/monitor.log"
  fi
fi

if (( COMPLETED == 0 )); then
  steps=$(grep -oP 'ppo step \K[0-9]+' "$LOG_DIR/trainer.log" 2>/dev/null | tail -1 || true)
  echo "FATAL: monitor wall-clock budget (${MONITOR_HOURS}h) expired at step ${steps:-0}/$MAX_STEPS" \
    | tee -a "$LOG_DIR/monitor.log"
  echo "  run is TRUNCATED, not complete. Resume from the last checkpoint." \
    | tee -a "$LOG_DIR/monitor.log"
  echo "  (raise MONITOR_HOURS for slower runs)" | tee -a "$LOG_DIR/monitor.log"
  exit 8
fi

echo "=== Final log tail ==="
grep 'ppo step' "$LOG_DIR/trainer.log" | tail -5 || true
count_monitor_metrics
echo "mismatch: $MON_MISMATCH"
echo "fallback: $MON_FALLBACK"
echo "loader_delta: $MON_LOADER_DELTA"
echo "chip errors: $MON_CHIP"
echo "=== Done (eval at checkpoints: run_eval separately) ==="
