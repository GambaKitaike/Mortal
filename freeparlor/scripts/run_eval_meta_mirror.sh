#!/usr/bin/env bash
# メタ対決ハーネスのミラー較正脚 (backlog 5)。
#
# 目的: レンズ3 (メタ対決 probe, eval_meta_stage1_vs_stage2.py) は challenger の
# 収支を「理論ミラー値 (素点 -5 / 順位点 0 / チップ 0)」からの逸脱として解釈する
# (例: Stage3-16000 vs Stage1-16000 の チップ -0.675 ≈ -2.9SE)。本スクリプトは
# 同一 checkpoint を challenger と baseline の両方に置いた対称対戦 (X vs 3X) を
# 走らせ、ハーネス実測が理論ミラー値へ SE 圏内で一致することを確認する。
# これにより「理論値からの逸脱」解釈のゼロ点をハーネス実測で裏取りし、座席ローテの
# 非対称・有限標本バイアスが無いことを保証する。レンズ1/2 の init 行整合性チェックの
# レンズ3 版に相当。
#
# 較正脚は全席同一方策なので challenger 統計 = ミラー値そのもの。REFERENCE_CKPT を
# 変えれば任意の checkpoint でハーネスの中立性を検査できる (デフォルトはレンズ3 の
# baseline = Stage1 step16000)。
#
# GPU 単一系統規律: 他の GPU ワークロード (DRCA 測定等) が走っている間は
# nvidia-smi idle ガードで abort する。DRCA 完走後に実行すること。
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"

# 較正の基準 checkpoint (デフォルト = レンズ3 の baseline = Stage1 step16000)。
REFERENCE_CKPT="${REFERENCE_CKPT:-/home/gamba/mahjong/runs/ppo/stage1_20260706_020120_resume/checkpoints/step_016000.pth}"
# 出力先 run dir (config.toml を MORTAL_CFG に使う。既定は reference の run dir)。
REFERENCE_RUN_DIR="${REFERENCE_RUN_DIR:-/home/gamba/mahjong/runs/ppo/stage1_20260706_020120_resume}"
LABEL="${LABEL:-mirror_stage1step16000}"
K_SE="${K_SE:-2.0}"

CFG="$REFERENCE_RUN_DIR/config.toml"
RESULTS_DIR="$REFERENCE_RUN_DIR/logs/eval_meta"
mkdir -p "$RESULTS_DIR"

if [[ ! -f "$REFERENCE_CKPT" ]]; then
  echo "FATAL: REFERENCE_CKPT not found: $REFERENCE_CKPT" >&2; exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "FATAL: config.toml not found: $CFG" >&2; exit 1
fi

export PYTHONPATH="$REPO/mortal"
export PYTHONUNBUFFERED=1
export MORTAL_CFG="$CFG"

echo "=== Meta mirror calibration leg (backlog 5): $(date -Iseconds) ==="
echo "REFERENCE_CKPT=$REFERENCE_CKPT  (both challenger and baseline seats)"
echo "REFERENCE_RUN_DIR=$REFERENCE_RUN_DIR"
echo "LABEL=$LABEL  K_SE=$K_SE"

if ss -tlnp 2>/dev/null | grep -q 5000; then
  echo "ERROR: port 5000 in use — abort" >&2; exit 1
fi
echo "port 5000 clear"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "ERROR: GPU compute processes still running (single-stream rule — run after DRCA)" >&2
  nvidia-smi
  exit 1
fi
echo "GPU idle"

"$REPO/freeparlor/scripts/preflight_libriichi.sh" "$REPO"

log="$RESULTS_DIR/eval_${LABEL}.log"
echo ""
echo "--- $LABEL (X vs 3X, challenger==baseline==reference) ---"
echo "$(date -Iseconds) start $LABEL"
EVAL_LABEL="$LABEL" \
EVAL_CHALLENGER_CHECKPOINT="$REFERENCE_CKPT" \
EVAL_BASELINE_CHECKPOINT="$REFERENCE_CKPT" \
conda run --no-capture-output -n mortal python "$REPO/freeparlor/scripts/eval_meta_stage1_vs_stage2.py" 2>&1 | tee "$log"
echo "$(date -Iseconds) done $LABEL"

echo "--- mirror calibration: $LABEL ---"
# --strict: 較正 FAIL 時に exit 1 (pipefail で本スクリプトも abort)。
conda run --no-capture-output -n mortal python "$REPO/freeparlor/scripts/analyze_freeparlor_pnl_1v3.py" \
  "$RESULTS_DIR/game_logs_${LABEL}" -o "$RESULTS_DIR/pnl_${LABEL}.txt" \
  --mirror-calibration --k-se "$K_SE" --strict 2>&1 | tee "$RESULTS_DIR/pnl_${LABEL}.log"

echo ""
echo "=== Mirror calibration complete (overall PASS): $(date -Iseconds) ==="
echo "理論ミラー値 (素点 -5 / 順位点 0 / チップ 0) への SE 圏内一致を確認。"
echo "レンズ3 の逸脱解釈のゼロ点がハーネス実測で裏取りされた。"
