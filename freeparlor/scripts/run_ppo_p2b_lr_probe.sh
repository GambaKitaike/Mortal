#!/usr/bin/env bash
# PPO P2b lr probe — lr=2e-5 only (wrapper around run_ppo_p2_smoke.sh)
set -euo pipefail
REPO="/home/gamba/mahjong/Mortal"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/smoke_p2b"
export PPO_CONFIG="$REPO/freeparlor/configs/ppo_p2b_lr_probe.toml"
export CONFIG_TAG="smoke_p2b"
export TMUX_SESSION="${TMUX_SESSION:-ppo_p2b_lr}"
exec bash "$REPO/freeparlor/scripts/run_ppo_p2_smoke.sh"
