#!/usr/bin/env bash
# PPO Stage3 — 16k step main run (anneal-tied per-decision call bonus,
# stage3_design.md)
#
# Resolves the `stage3_PENDING_LAUNCH` placeholder in the frozen
# freeparlor/configs/ppo_stage3.toml to the real run dir, then execs the
# generic P3 inner launcher (preflight + tmux + trainer/clients), mirroring
# freeparlor/scripts/run_ppo_stage2.sh. This does NOT generate the config
# from a template — ppo_stage3.toml already exists (pre-registered) with all
# hyperparameters fixed except the placeholder run-dir string. Only that
# placeholder is substituted here.
set -euo pipefail
REPO="/home/gamba/mahjong/Mortal"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/stage3_${RUN_SUFFIX}"
export PPO_CONFIG="$REPO/freeparlor/configs/ppo_stage3.toml"
export CONFIG_TAG="stage3_${RUN_SUFFIX}"
export TMUX_SESSION="${TMUX_SESSION:-ppo_stage3_${RUN_SUFFIX}}"
MAX_STEPS="${MAX_STEPS:-16000}"

PLACEHOLDER="stage3_PENDING_LAUNCH"
RESOLVED="stage3_${RUN_SUFFIX}"

if ! grep -q "$PLACEHOLDER" "$PPO_CONFIG"; then
  echo "FATAL: placeholder $PLACEHOLDER not found in $PPO_CONFIG (already resolved by a prior launch?)" >&2
  exit 1
fi

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
