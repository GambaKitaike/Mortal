#!/usr/bin/env bash
# PPO Stage2 — 16k step main run (aka-enrichment curriculum, stage2_design.md)
#
# Resolves the `stage2_PENDING_LAUNCH` placeholder in the frozen
# freeparlor/configs/ppo_stage2.toml to the real run dir, then execs the
# generic P3 inner launcher (preflight + tmux + trainer/clients), mirroring
# freeparlor/scripts/run_ppo_p3_stage1.sh. Unlike Stage1, this does NOT
# generate the config from a template — ppo_stage2.toml already exists
# (pre-registered, prompt②) with all hyperparameters fixed except the
# placeholder run-dir string. Only that placeholder is substituted here.
set -euo pipefail
REPO="/home/gamba/mahjong/Mortal"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/stage2_${RUN_SUFFIX}"
export PPO_CONFIG="$REPO/freeparlor/configs/ppo_stage2.toml"
export CONFIG_TAG="stage2_${RUN_SUFFIX}"
export TMUX_SESSION="${TMUX_SESSION:-ppo_stage2_${RUN_SUFFIX}}"
MAX_STEPS="${MAX_STEPS:-16000}"

PLACEHOLDER="stage2_PENDING_LAUNCH"
RESOLVED="stage2_${RUN_SUFFIX}"

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
