#!/usr/bin/env bash
# PPO L1 O1 — submit_every 50 -> 10
# 事前登録: freeparlor/docs/design/l1_o1_submit_every_design.md（2026-07-30 凍結）
#
# run_ppo_anchor_k_b04.sh の構造を踏襲（新規ロジックを持たない。inner.sh へ exec するだけ）。
# 発進前に「凍結された値」を config から検査する = config を取り違えたら発進しない。
set -euo pipefail
REPO="/home/gamba/mahjong/Mortal"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/l1_o1_${RUN_SUFFIX}"
export PPO_CONFIG="$REPO/freeparlor/configs/ppo_l1_o1.toml"
export CONFIG_TAG="l1_o1_${RUN_SUFFIX}"
export TMUX_SESSION="${TMUX_SESSION:-ppo_l1_o1_${RUN_SUFFIX}}"
MAX_STEPS="${MAX_STEPS:-16000}"

RUNS_ROOT="/home/gamba/mahjong/runs"
DISK_MIN_GB="${DISK_MIN_GB:-450}"
AVAIL_GB=$(df -BG --output=avail "$RUNS_ROOT" | tail -n1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt "$DISK_MIN_GB" ]; then
  echo "FATAL: available disk on $RUNS_ROOT is ${AVAIL_GB}G, below DISK_MIN_GB=${DISK_MIN_GB}G" >&2
  exit 1
fi
echo "disk check passed: ${AVAIL_GB}G available on $RUNS_ROOT (>= ${DISK_MIN_GB}G)"

PLACEHOLDER="l1_o1_PENDING_LAUNCH"
RESOLVED="l1_o1_${RUN_SUFFIX}"

if ! grep -q "$PLACEHOLDER" "$PPO_CONFIG"; then
  echo "FATAL: placeholder $PLACEHOLDER not found in $PPO_CONFIG (already resolved by a prior launch?)" >&2
  exit 1
fi

# §11 が凍結した値を発進時に検査する（単一変数 = submit_every のみ）
for kv in "submit_every = 10" "max_steps = 16000" "save_every = 2000" "ppo_epochs = 4"; do
  if ! grep -qF "$kv" "$PPO_CONFIG"; then
    echo "FATAL: expected '$kv' in $PPO_CONFIG (l1_o1_submit_every_design.md §11)" >&2
    exit 1
  fi
done
if grep -qE '^submit_every = 50$' "$PPO_CONFIG"; then
  echo "FATAL: $PPO_CONFIG still has submit_every = 50 (O1 の介入になっていない)" >&2
  exit 1
fi
# base は plain PPO。他層の介入キーが混入したら 2 変数になるので落とす
for forbidden in "kl_beta" "anchor_prob" "anchor_checkpoint" "p_enrich" "call_bonus"; do
  if grep -qE "^[[:space:]]*${forbidden}[[:space:]]*=" "$PPO_CONFIG"; then
    echo "FATAL: $PPO_CONFIG contains '$forbidden' — base は plain PPO でなければならない" >&2
    exit 1
  fi
done
echo "frozen config values verified: submit_every=10 / max_steps=16000 / save_every=2000 / ppo_epochs=4"
echo "plain PPO base verified: kl_beta / anchor_prob / p_enrich / call_bonus いずれもキー不在"

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
