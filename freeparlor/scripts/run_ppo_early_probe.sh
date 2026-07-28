#!/usr/bin/env bash
# PPO Early Damage Probe — step 0-2000 の内部形状（early_damage_probe_design.md）
# 診断 run（判定非関与）。Arm K と同一構成のまま max_steps=2000 で止め、
# 観測専用の checkpoints_diag/ を 100 step ごとに残す。
set -euo pipefail
REPO="/home/gamba/mahjong/Mortal"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/early_probe_${RUN_SUFFIX}"
export PPO_CONFIG="$REPO/freeparlor/configs/ppo_early_probe.toml"
export CONFIG_TAG="early_probe_${RUN_SUFFIX}"
export TMUX_SESSION="${TMUX_SESSION:-ppo_early_probe_${RUN_SUFFIX}}"
# 2000 step ≈ 3.1h（Arm K 実績 16000 step / 24.6h）。既定 48h の監視予算で十分余裕。
export MAX_STEPS="${MAX_STEPS:-2000}"

RUNS_ROOT="/home/gamba/mahjong/runs"
# 16k run より drain が小さい（2000 step ≈ 30-45GB）ので下限も下げてよいが、
# 「1系統ルールで他の run と同居しない」前提を崩さないため既定は据え置く。
DISK_MIN_GB="${DISK_MIN_GB:-450}"
AVAIL_GB=$(df -BG --output=avail "$RUNS_ROOT" | tail -n1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt "$DISK_MIN_GB" ]; then
  echo "FATAL: available disk on $RUNS_ROOT is ${AVAIL_GB}G, below DISK_MIN_GB=${DISK_MIN_GB}G" >&2
  exit 1
fi
echo "disk check passed: ${AVAIL_GB}G available on $RUNS_ROOT (>= ${DISK_MIN_GB}G)"

PLACEHOLDER="early_probe_PENDING_LAUNCH"
RESOLVED="early_probe_${RUN_SUFFIX}"

if ! grep -q "$PLACEHOLDER" "$PPO_CONFIG"; then
  echo "FATAL: placeholder $PLACEHOLDER not found in $PPO_CONFIG (already resolved by a prior launch?)" >&2
  exit 1
fi

# 設計書が凍結した3値を発進時に検査する（config の取り違え・手編集の検出）
for kv in "max_steps = 2000" "diag_save_every = 100" "kl_beta = 0.1" "save_every = 2000"; do
  if ! grep -qF "$kv" "$PPO_CONFIG"; then
    echo "FATAL: expected '$kv' in $PPO_CONFIG (early_damage_probe_design.md §9)" >&2
    exit 1
  fi
done
echo "frozen config values verified: max_steps/diag_save_every/kl_beta/save_every"

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
