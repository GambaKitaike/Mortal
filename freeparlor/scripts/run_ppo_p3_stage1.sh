#!/usr/bin/env bash
# PPO P3 Stage1 — 16k step main run (opponent pool + P2c instrumentation)
set -euo pipefail
REPO="/home/gamba/mahjong/Mortal"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/stage1_${RUN_SUFFIX}"
export PPO_CONFIG="$REPO/freeparlor/configs/ppo_p3_stage1.toml"
export CONFIG_TAG="stage1_${RUN_SUFFIX}"
export TMUX_SESSION="${TMUX_SESSION:-ppo_p3_${RUN_SUFFIX}}"
MAX_STEPS="${MAX_STEPS:-16000}"

python3 - <<PY
from pathlib import Path
repo = Path("$REPO")
run_dir = Path("$RUN_DIR")
template = (repo / "freeparlor/configs/ppo_p2c_advantage_decomp.toml").read_text()
text = template.replace("/home/gamba/mahjong/runs/ppo/smoke_p2c_20260703_054732", str(run_dir))
text = text.replace(
    "# PPO P2c — advantage decomp instrumentation (generated smoke_p2c_20260703_054732)",
    f"# PPO P3 Stage1 main run (generated {run_dir.name})",
)
text = text.replace("save_every = 50", "save_every = 2000")
text = text.replace("submit_every = 50", "submit_every = 50")
text = text.replace("max_steps = 400", f"max_steps = {int('$MAX_STEPS')}")
text = text.replace("test_every = 100000", "test_every = 100000")
if "[opponent_pool]" not in text:
    text += """

[opponent_pool]
enabled = true
past_k = 5
latest_prob = 0.5
"""
out = repo / "freeparlor/configs/ppo_p3_stage1.toml"
out.write_text(text)
print(f"config -> {out}")
print(f"run dir -> {run_dir}")
PY

exec bash "$REPO/freeparlor/scripts/run_ppo_p3_stage1_inner.sh"
