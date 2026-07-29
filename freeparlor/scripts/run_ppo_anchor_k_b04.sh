#!/usr/bin/env bash
# PPO Anchor Arm K 再走 kl_beta=0.4 — anchored_ppo_design.md §6a の機械的適用
set -euo pipefail
REPO="/home/gamba/mahjong/Mortal"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/anchor_k_b04_${RUN_SUFFIX}"
export PPO_CONFIG="$REPO/freeparlor/configs/ppo_anchor_k_b04.toml"
export CONFIG_TAG="anchor_k_b04_${RUN_SUFFIX}"
export TMUX_SESSION="${TMUX_SESSION:-ppo_anchor_k_b04_${RUN_SUFFIX}}"
MAX_STEPS="${MAX_STEPS:-16000}"

RUNS_ROOT="/home/gamba/mahjong/runs"
DISK_MIN_GB="${DISK_MIN_GB:-450}"
AVAIL_GB=$(df -BG --output=avail "$RUNS_ROOT" | tail -n1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt "$DISK_MIN_GB" ]; then
  echo "FATAL: available disk on $RUNS_ROOT is ${AVAIL_GB}G, below DISK_MIN_GB=${DISK_MIN_GB}G" >&2
  exit 1
fi
echo "disk check passed: ${AVAIL_GB}G available on $RUNS_ROOT (>= ${DISK_MIN_GB}G)"

PLACEHOLDER="anchor_k_b04_PENDING_LAUNCH"
RESOLVED="anchor_k_b04_${RUN_SUFFIX}"

if ! grep -q "$PLACEHOLDER" "$PPO_CONFIG"; then
  echo "FATAL: placeholder $PLACEHOLDER not found in $PPO_CONFIG (already resolved by a prior launch?)" >&2
  exit 1
fi

# §6a が定めた値を発進時に検査する（Arm K の config を取り違えると再走にならない）
for kv in "kl_beta = 0.4" "max_steps = 16000" "save_every = 2000"; do
  if ! grep -qF "$kv" "$PPO_CONFIG"; then
    echo "FATAL: expected '$kv' in $PPO_CONFIG (anchored_ppo_design.md §6a)" >&2
    exit 1
  fi
done
if grep -qF "kl_beta = 0.1" "$PPO_CONFIG"; then
  echo "FATAL: $PPO_CONFIG still has kl_beta = 0.1 (Arm K の config では再走にならない)" >&2
  exit 1
fi
echo "frozen config values verified: kl_beta=0.4 / max_steps=16000 / save_every=2000"

python3 - <<PY
from pathlib import Path
cfg = Path("$PPO_CONFIG")
text = cfg.read_text()
text = text.replace("$PLACEHOLDER", "$RESOLVED")
cfg.write_text(text)
print(f"config resolved in place -> {cfg} (placeholder -> $RESOLVED)")
PY

echo "run dir -> $RUN_DIR"

exec bash "$REPO/freeparlor/scripts/run_ppo_p3_stage1_inner.sh"
