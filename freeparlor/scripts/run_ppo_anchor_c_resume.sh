#!/usr/bin/env bash
# anchor Arm C resume — 2026-07-26 の launcher truncation 事故からの再開。
#
# 事故: monitor ループのハードコード 24h DEADLINE が step 14100/16000 の時点で
# 期限切れになり、期限切れが正常完走と同じ shutdown 経路（Final log tail -> Done ->
# Cleanup -> exit 0）へ落ちたため、run が「成功」を報告しながら SIGTERM で切られた。
# Arm C は 24h を超えた最初の run（595-736 step/h。anchor は draw の 25% で別 checkpoint
# をロードするぶん遅い）。inner launcher 側は同日修正済み（期限切れ = exit 8、
# MONITOR_HOURS で可変。デフォルト 48h）。
#
# 単一変数（anchor_prob=0.25 / anchor_checkpoint）は不変。動くのは run パスのみで、
# これは Stage2 の disk 事故 resume（run_ppo_stage2_resume.sh）と同じ扱い。
# 判定条件（anchored_ppo_design.md §6: step16000 checkpoint・判定窓 8000-16000）は不変。
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
SOURCE_RUN="${SOURCE_RUN:-/home/gamba/mahjong/runs/ppo/anchor_c_20260725_164756}"
RESUME_STEP="${RESUME_STEP:-14000}"
MAX_STEPS="${MAX_STEPS:-16000}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)_resume}"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/anchor_c_${RUN_SUFFIX}"
export PPO_CONFIG="$REPO/freeparlor/configs/ppo_anchor_c.toml"
export CONFIG_TAG="anchor_c_${RUN_SUFFIX}"
export TMUX_SESSION="${TMUX_SESSION:-ppo_anchor_c_${RUN_SUFFIX}}"
export MAX_STEPS
# 残り ~2000 step（実測 ~600 step/h で ~3.4h）だが、余裕を持って 48h を明示する
export MONITOR_HOURS="${MONITOR_HOURS:-48}"

SRC_CKPT="$SOURCE_RUN/checkpoints/step_$(printf '%06d' "$RESUME_STEP").pth"
log() { echo "$(date -Iseconds) $*"; }

[[ -f "$SRC_CKPT" ]] || { echo "FATAL: source checkpoint not found: $SRC_CKPT" >&2; exit 1; }

OLD_BASENAME="$(basename "$SOURCE_RUN")"
NEW_BASENAME="anchor_c_${RUN_SUFFIX}"

if ! grep -q "$OLD_BASENAME" "$PPO_CONFIG"; then
  echo "FATAL: expected old run basename $OLD_BASENAME not found in $PPO_CONFIG" >&2
  exit 1
fi

RUNS_ROOT="/home/gamba/mahjong/runs"
DISK_MIN_GB="${DISK_MIN_GB:-100}"
AVAIL_GB=$(df -BG --output=avail "$RUNS_ROOT" | tail -n1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt "$DISK_MIN_GB" ]; then
  echo "FATAL: available disk ${AVAIL_GB}G < DISK_MIN_GB=${DISK_MIN_GB}G" >&2; exit 1
fi
log "disk check passed: ${AVAIL_GB}G available"

log "=== anchor Arm C resume setup ==="
log "source run : $SOURCE_RUN"
log "resume step: $RESUME_STEP -> $MAX_STEPS"
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
import torch, tomllib
p='$RUN_DIR/mortal.pth'
s=torch.load(p, weights_only=True, map_location='cpu')
assert s['steps'] == $RESUME_STEP, f'steps={s[\"steps\"]} expected $RESUME_STEP'
assert 'optimizer' in s and len(s['optimizer']['state']) > 0, 'missing optimizer state'
assert 'actor_critic' in s, 'missing actor_critic'
cfg = tomllib.load(open('$RUN_DIR/config.toml','rb'))
op = cfg['opponent_pool']
assert op['anchor_prob'] == 0.25, f'anchor_prob={op[\"anchor_prob\"]} != 0.25 (単一変数が動いている)'
assert cfg['ppo'].get('kl_beta', 0.0) == 0.0, 'kl_beta must stay OFF in Arm C'
print(f'resume ckpt OK: steps={s[\"steps\"]:,} optimizer_params={len(s[\"optimizer\"][\"state\"])}')
print(f'single variable OK: anchor_prob={op[\"anchor_prob\"]} anchor={op[\"anchor_checkpoint\"]}')
"

log "opponent pool ckpt_dir=$RUN_DIR/checkpoints ($(ls "$RUN_DIR/checkpoints"/step_*.pth | wc -l) files)"
log "=== resume preflight (data) PASSED, handing off to inner launcher ==="
exec bash "$REPO/freeparlor/scripts/run_ppo_p3_stage1_inner.sh"
