#!/usr/bin/env bash
# Control eval: compare mortal_init.pth vs step400 mortal.pth (same seeds).
set -euo pipefail

RUN_DIR="/home/gamba/mahjong/runs/ppo/smoke_p2"
CFG="$RUN_DIR/config.toml"
REPO="/home/gamba/mahjong/Mortal"
SCRIPT="$REPO/freeparlor/scripts/eval_ppo_smoke_sanity.sh"

export MORTAL_CFG="$CFG"
export PYTHONPATH="$REPO/mortal"
export PYTHONUNBUFFERED=1

run_one() {
  local label="$1"
  local ckpt="$2"
  export EVAL_LABEL="$label"
  export EVAL_CHECKPOINT="$ckpt"
  bash "$REPO/freeparlor/scripts/run_eval_ppo_smoke_sanity.sh"
}

run_one init "$RUN_DIR/mortal_init.pth"
run_one step400 "$RUN_DIR/mortal.pth"
