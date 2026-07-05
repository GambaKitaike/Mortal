#!/usr/bin/env bash
# PPO P3 Stage1 — pause/resume: new run dir from step_010000, preflight, ready to launch.
# Does NOT start training unless LAUNCH=1.
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
SOURCE_RUN="${SOURCE_RUN:-/home/gamba/mahjong/runs/ppo/stage1_20260705_053301}"
RESUME_STEP="${RESUME_STEP:-10000}"
MAX_STEPS="${MAX_STEPS:-16000}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)_resume}"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/stage1_${RUN_SUFFIX}"
export PPO_CONFIG="$REPO/freeparlor/configs/ppo_p3_stage1.toml"
export CONFIG_TAG="stage1_${RUN_SUFFIX}"
export TMUX_SESSION="${TMUX_SESSION:-ppo_p3_${RUN_SUFFIX}}"
LAUNCH="${LAUNCH:-0}"

SRC_CKPT="$SOURCE_RUN/checkpoints/step_$(printf '%06d' "$RESUME_STEP").pth"
log() { echo "$(date -Iseconds) $*"; }

if [[ ! -f "$SRC_CKPT" ]]; then
  echo "ERROR: source checkpoint not found: $SRC_CKPT"
  exit 1
fi

log "=== P3 resume setup ==="
log "source run: $SOURCE_RUN"
log "resume step: $RESUME_STEP"
log "new run dir: $RUN_DIR"

python3 - <<PY
from pathlib import Path
repo = Path("$REPO")
run_dir = Path("$RUN_DIR")
source = Path("$SOURCE_RUN")
template = (repo / "freeparlor/configs/ppo_p2c_advantage_decomp.toml").read_text()
text = template.replace("/home/gamba/mahjong/runs/ppo/smoke_p2c_20260703_054732", str(run_dir))
text = text.replace(
    "# PPO P2c — advantage decomp instrumentation (generated smoke_p2c_20260703_054732)",
    f"# PPO P3 Stage1 resume (generated {run_dir.name})",
)
text = text.replace("save_every = 50", "save_every = 2000")
text = text.replace("submit_every = 50", "submit_every = 50")
text = text.replace("max_steps = 400", f"max_steps = {int('$MAX_STEPS')}")
text = text.replace("test_every = 100000", "test_every = 100000")
if "[opponent_pool]" not in text:
    text += """

[opponent_pool]
enabled = true
past_k = 5
latest_prob = 0.5
"""
out = repo / "freeparlor/configs/ppo_p3_stage1.toml"
out.write_text(text)
print(f"config -> {out}")
PY

mkdir -p "$RUN_DIR"/{logs,checkpoints,tb,buffer,drain,test_play}
mkdir -p "$RUN_DIR/train_play"/{client0,client1,client2}

log "copy mortal.pth from step_$(printf '%06d' "$RESUME_STEP")"
cp -f "$SRC_CKPT" "$RUN_DIR/mortal.pth"

log "copy numbered checkpoints for opponent pool"
shopt -s nullglob
for f in "$SOURCE_RUN/checkpoints"/step_*.pth; do
  step_num=$(basename "$f" .pth | sed 's/step_0*//')
  if (( step_num <= RESUME_STEP )); then
    cp -f "$f" "$RUN_DIR/checkpoints/$(basename "$f")"
  fi
done
shopt -u nullglob

if [[ ! -f "$RUN_DIR/checkpoints/step_000000.pth" ]]; then
  INIT0_SRC="$SOURCE_RUN/../stage1_20260705_072900/checkpoints/step_000000.pth"
  if [[ -f "$INIT0_SRC" ]]; then
    log "copy step_000000.pth fallback from run7 launch dir"
    cp -f "$INIT0_SRC" "$RUN_DIR/checkpoints/step_000000.pth"
  fi
fi

cp "$PPO_CONFIG" "$RUN_DIR/config.toml"

conda run -n mortal python -c "
import torch
p='$RUN_DIR/mortal.pth'
s=torch.load(p, weights_only=True, map_location='cpu')
assert s['steps'] == $RESUME_STEP, f'steps={s[\"steps\"]} expected $RESUME_STEP'
assert 'optimizer' in s and len(s['optimizer']['state']) > 0, 'missing optimizer state'
print(f'resume checkpoint OK: steps={s[\"steps\"]:,} optimizer_params={len(s[\"optimizer\"][\"state\"])}')
"

log "=== Pre-flight: rebuild libriichi.so ==="
CARGO_TARGET_DIR="$REPO/target" PYO3_PYTHON="$(conda run -n mortal python -c 'import sys; print(sys.executable)')" \
  cargo build --release -p libriichi --lib
cp -f "$REPO/target/release/libriichi.so" "$REPO/mortal/libriichi.so"

log "=== Pre-flight: libriichi import check ==="
PYTHONPATH="$REPO/mortal" conda run -n mortal python -c "import libriichi.arena"

log "=== Pre-flight verify (checks 1-16) ==="
conda run -n mortal python "$REPO/freeparlor/scripts/verify_ppo_p1.py" \
  --checkpoint /home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth \
  --grp-state /home/gamba/mahjong/runs/grp.pth \
  2>&1 | tee "$RUN_DIR/logs/verify_p1.log" | tail -25

grep -q 'ALL 16 CHECKS PASSED' "$RUN_DIR/logs/verify_p1.log"

log "=== Resume preflight PASSED ==="
log "RUN_DIR=$RUN_DIR"
log "mortal.pth steps=$RESUME_STEP (next trainer step will be $((RESUME_STEP + 1)))"
log "opponent pool ckpt_dir=$RUN_DIR/checkpoints ($(ls "$RUN_DIR/checkpoints"/step_*.pth | wc -l) files)"

if [[ "$LAUNCH" == "1" ]]; then
  log "LAUNCH=1: starting training via run_ppo_p3_stage1_inner.sh"
  export RUN_DIR PPO_CONFIG CONFIG_TAG TMUX_SESSION MAX_STEPS
  exec bash "$REPO/freeparlor/scripts/run_ppo_p3_stage1_inner.sh"
else
  log "Ready. Confirm then launch:"
  log "  LAUNCH=1 RUN_DIR=$RUN_DIR RUN_SUFFIX=${RUN_SUFFIX} bash $REPO/freeparlor/scripts/run_ppo_p3_resume.sh"
  log "  or: RUN_DIR=$RUN_DIR bash $REPO/freeparlor/scripts/run_ppo_p3_stage1_inner.sh"
fi
