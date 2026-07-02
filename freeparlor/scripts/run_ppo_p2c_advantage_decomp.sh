#!/usr/bin/env bash
# PPO P2c — advantage / reward decomposition instrumentation (lr=2e-5, 400 step)
set -euo pipefail
REPO="/home/gamba/mahjong/Mortal"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/smoke_p2c_${RUN_SUFFIX}"
export PPO_CONFIG="$REPO/freeparlor/configs/ppo_p2c_advantage_decomp.toml"
export CONFIG_TAG="smoke_p2c_${RUN_SUFFIX}"
export TMUX_SESSION="${TMUX_SESSION:-ppo_p2c_${RUN_SUFFIX}}"

python3 - <<PY
from pathlib import Path
repo = Path("$REPO")
run_dir = Path("$RUN_DIR")
template = (repo / "freeparlor/configs/ppo_p2b_lr_probe.toml").read_text()
text = template.replace("/home/gamba/mahjong/runs/ppo/smoke_p2b", str(run_dir))
text = text.replace(
    "# PPO P2b — lr probe (lr=2e-5 only; otherwise identical to ppo_p2_smoke.toml)",
    f"# PPO P2c — advantage decomp instrumentation (generated {run_dir.name})",
)
out = repo / "freeparlor/configs/ppo_p2c_advantage_decomp.toml"
out.write_text(text)
print(f"config -> {out}")
print(f"run dir -> {run_dir}")
PY

exec bash "$REPO/freeparlor/scripts/run_ppo_p2_smoke.sh"
