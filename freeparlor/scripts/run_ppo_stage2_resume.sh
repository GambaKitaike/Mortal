#!/usr/bin/env bash
# PPO Stage2 resume — after the 2026-07-09 disk-exhaustion crash (server OSError
# ENOSPC -> trainer UnexpectedEOF -> tmux server itself died). Last good
# checkpoint was step_006000.pth; resumes into a new run dir from there,
# mirroring freeparlor/scripts/run_ppo_p3_resume.sh. Single variable (p_enrich)
# is unchanged; only the run path moves, same as any other resume.
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
SOURCE_RUN="${SOURCE_RUN:-/home/gamba/mahjong/runs/ppo/stage2_20260709_092541}"
RESUME_STEP="${RESUME_STEP:-6000}"
MAX_STEPS="${MAX_STEPS:-16000}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)_resume}"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/stage2_${RUN_SUFFIX}"
export PPO_CONFIG="$REPO/freeparlor/configs/ppo_stage2.toml"
export CONFIG_TAG="stage2_${RUN_SUFFIX}"
export TMUX_SESSION="${TMUX_SESSION:-ppo_stage2_${RUN_SUFFIX}}"

SRC_CKPT="$SOURCE_RUN/checkpoints/step_$(printf '%06d' "$RESUME_STEP").pth"
log() { echo "$(date -Iseconds) $*"; }

if [[ ! -f "$SRC_CKPT" ]]; then
  echo "ERROR: source checkpoint not found: $SRC_CKPT"
  exit 1
fi

OLD_BASENAME="$(basename "$SOURCE_RUN")"
NEW_BASENAME="stage2_${RUN_SUFFIX}"

if ! grep -q "$OLD_BASENAME" "$PPO_CONFIG"; then
  echo "ERROR: expected old run basename $OLD_BASENAME not found in $PPO_CONFIG (already resolved to a different run?)"
  exit 1
fi

log "=== Stage2 resume setup ==="
log "source run: $SOURCE_RUN"
log "resume step: $RESUME_STEP"
log "new run dir: $RUN_DIR"

python3 - <<PY
from pathlib import Path
cfg = Path("$PPO_CONFIG")
text = cfg.read_text()
text = text.replace("$OLD_BASENAME", "$NEW_BASENAME")
cfg.write_text(text)
print(f"config resolved in place -> {cfg} ($OLD_BASENAME -> $NEW_BASENAME)")
PY

mkdir -p "$RUN_DIR"/{logs,checkpoints,tb,buffer,drain,test_play}
mkdir -p "$RUN_DIR/train_play"/{client0,client1,client2}

log "copy mortal.pth from step_$(printf '%06d' "$RESUME_STEP")"
cp -f "$SRC_CKPT" "$RUN_DIR/mortal.pth"

log "copy numbered checkpoints (<= resume step) for opponent pool"
shopt -s nullglob
for f in "$SOURCE_RUN/checkpoints"/step_*.pth; do
  step_num=$(basename "$f" .pth | sed 's/step_0*//')
  step_num=${step_num:-0}
  if (( step_num <= RESUME_STEP )); then
    cp -f "$f" "$RUN_DIR/checkpoints/$(basename "$f")"
  fi
done
shopt -u nullglob

cp "$PPO_CONFIG" "$RUN_DIR/config.toml"

source /home/gamba/miniconda3/etc/profile.d/conda.sh
conda activate mortal
python -c "
import torch
p='$RUN_DIR/mortal.pth'
s=torch.load(p, weights_only=True, map_location='cpu')
assert s['steps'] == $RESUME_STEP, f'steps={s[\"steps\"]} expected $RESUME_STEP'
assert 'optimizer' in s and len(s['optimizer']['state']) > 0, 'missing optimizer state'
print(f'resume checkpoint OK: steps={s[\"steps\"]:,} optimizer_params={len(s[\"optimizer\"][\"state\"])}')
"

log "opponent pool ckpt_dir=$RUN_DIR/checkpoints ($(ls "$RUN_DIR/checkpoints"/step_*.pth | wc -l) files)"
log "=== Stage2 resume preflight (data) PASSED, handing off to inner launcher ==="
exec bash "$REPO/freeparlor/scripts/run_ppo_p3_stage1_inner.sh"
