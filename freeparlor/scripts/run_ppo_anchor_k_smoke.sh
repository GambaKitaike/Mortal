#!/usr/bin/env bash
# anchor Arm K — 400-step 配管スモーク（**発進ではない**）
#
# 目的は配管検査のみ（`CLAUDE.md` 実験の規律「400 step 級スモークで挙動の結論を出さない」）:
#   - kl_anchor イベントが毎バッチ diag に出る
#   - kl_beta が設定値 0.1
#   - kl_ref_mean が有限・NaN/inf なし・step0 は ~0（ref = step0 方策の陽性対照）
#   - trainer が NaN で落ちない（CPU/GPU 単体検査では届かない実データ経路の確認）
#
# **凍結された `ppo_anchor_k.toml` の placeholder は消費しない。**
# 本スクリプトは config を smoke run dir へコピーしてから置換するので、本発進用の
# placeholder は無傷のまま残る（消費すると本発進が FATAL で撃てなくなる）。
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_NAME="smoke_anchor_k_${RUN_SUFFIX}"
export RUN_DIR="/home/gamba/mahjong/runs/ppo/${SMOKE_NAME}"
export CONFIG_TAG="$SMOKE_NAME"
export TMUX_SESSION="${TMUX_SESSION:-ppo_${SMOKE_NAME}}"
export MAX_STEPS="${MAX_STEPS:-400}"
export MONITOR_HOURS="${MONITOR_HOURS:-3}"

SRC_CONFIG="$REPO/freeparlor/configs/ppo_anchor_k.toml"
PLACEHOLDER="anchor_k_PENDING_LAUNCH"

[[ -f "$SRC_CONFIG" ]] || { echo "FATAL: $SRC_CONFIG not found" >&2; exit 1; }
if ! grep -q "$PLACEHOLDER" "$SRC_CONFIG"; then
  echo "FATAL: placeholder $PLACEHOLDER not in $SRC_CONFIG — 本発進用 config が既に消費済み" >&2
  exit 1
fi

# GPU 1系統ガード。pgrep -f は「パターン文字列を引数に持つシェル」に自己マッチする
# ため使わない。プロセスの comm が python であることを条件にして実プロセスだけを見る。
gpu_busy_list() {
  ps -eo comm=,pid=,args= | awk '($1=="python"||$1=="python3") && $0 ~ /(run_train_ppo|run_client|run_server|drca_run_probe|eval_ppo_smoke_sanity|eval_grp_baseline_1v3|eval_meta_stage1_vs_stage2)\.py/'
}
if [[ -n "$(gpu_busy_list)" ]]; then
  echo "FATAL: 学習/eval プロセスが稼働中 (GPU 1系統ルール)" >&2
  gpu_busy_list | head >&2
  exit 1
fi
echo "残党なし"

mkdir -p "$RUN_DIR"
SMOKE_CONFIG="$RUN_DIR/ppo_anchor_k_smoke.toml"
python3 - <<PY
from pathlib import Path
src = Path("$SRC_CONFIG").read_text()
out = src.replace("$PLACEHOLDER", "$SMOKE_NAME").replace("max_steps = 16000", "max_steps = $MAX_STEPS")
Path("$SMOKE_CONFIG").write_text(out)
import tomllib
cfg = tomllib.loads(out)
assert cfg['ppo']['kl_beta'] == 0.1, f"kl_beta={cfg['ppo']['kl_beta']} != 0.1"
assert cfg['ppo']['max_steps'] == $MAX_STEPS, cfg['ppo']['max_steps']
assert cfg['opponent_pool'].get('anchor_prob', 0.0) == 0.0, "Arm K で anchor_prob が立っている"
print(f"smoke config generated: $SMOKE_CONFIG (kl_beta={cfg['ppo']['kl_beta']}, max_steps={cfg['ppo']['max_steps']}, anchor_prob=0)")
PY
# 凍結 config が無傷であることを再確認（消費していない）
grep -q "$PLACEHOLDER" "$SRC_CONFIG" || { echo "FATAL: 凍結 config の placeholder を壊した" >&2; exit 1; }
echo "凍結 config の placeholder は無傷"

export PPO_CONFIG="$SMOKE_CONFIG"
echo "=== anchor Arm K 配管スモーク (max_steps=$MAX_STEPS) ==="
echo "run dir -> $RUN_DIR"
exec bash "$REPO/freeparlor/scripts/run_ppo_p3_stage1_inner.sh"
