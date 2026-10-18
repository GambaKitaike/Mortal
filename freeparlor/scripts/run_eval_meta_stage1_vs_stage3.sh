#!/usr/bin/env bash
# メタ対決 probe (stage3_design.md §6 レンズ3):
# challenger = Stage3 step16000 (stage3_20260712_033403) ×1 席
# baseline   = Stage1 step16000 (stage1_20260706_020120_resume) ×3 席
# 座席ローテ・seed 範囲は eval_grp_baseline_1v3.py と同一仕様 (400半荘、100 seed*4)
# ハーネスは既存 eval_meta_stage1_vs_stage2.py をそのまま流用 (challenger/baseline
# のチェックポイントパスのみ差し替え、スクリプト本体は無変更)
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
RUN_DIR="${RUN_DIR:-/home/gamba/mahjong/runs/ppo/stage3_20260712_033403}"
CFG="$RUN_DIR/config.toml"
CHALLENGER_CKPT="$RUN_DIR/checkpoints/step_016000.pth"
BASELINE_CKPT="/home/gamba/mahjong/runs/ppo/stage1_20260706_020120_resume/checkpoints/step_016000.pth"
RESULTS_DIR="$RUN_DIR/logs/eval_meta"
mkdir -p "$RESULTS_DIR"

export PYTHONPATH="$REPO/mortal"
export PYTHONUNBUFFERED=1
export MORTAL_CFG="$CFG"

echo "=== Stage3 meta probe (Stage3 step16000 vs Stage1 step16000): $(date -Iseconds) ==="
echo "RUN_DIR=$RUN_DIR"
echo "CHALLENGER_CKPT=$CHALLENGER_CKPT"
echo "BASELINE_CKPT=$BASELINE_CKPT"

if ss -tlnp 2>/dev/null | grep -q 5000; then
  echo "ERROR: port 5000 in use — abort" >&2; exit 1
fi
echo "port 5000 clear"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "ERROR: GPU compute processes still running (single-stream rule)" >&2
  nvidia-smi
  exit 1
fi
echo "GPU idle"

"$REPO/freeparlor/scripts/preflight_libriichi.sh" "$REPO"

label="meta_stage3step16000_vs_stage1step16000"
log="$RESULTS_DIR/eval_${label}.log"
echo ""
echo "--- $label ---"
echo "$(date -Iseconds) start $label"
EVAL_LABEL="$label" \
EVAL_CHALLENGER_CHECKPOINT="$CHALLENGER_CKPT" \
EVAL_BASELINE_CHECKPOINT="$BASELINE_CKPT" \
conda run --no-capture-output -n mortal python "$REPO/freeparlor/scripts/eval_meta_stage1_vs_stage2.py" 2>&1 | tee "$log"
echo "$(date -Iseconds) done $label"

echo "--- pnl: $label ---"
conda run --no-capture-output -n mortal python "$REPO/freeparlor/scripts/analyze_freeparlor_pnl_1v3.py" \
  "$RESULTS_DIR/game_logs_${label}" -o "$RESULTS_DIR/pnl_${label}.txt" 2>&1 | tee "$RESULTS_DIR/pnl_${label}.log"

echo ""
echo "=== Meta probe complete: $(date -Iseconds) ==="
echo ""
echo "=== Summary (challenger=Stage3 step16000 視点 vs baseline=Stage1 step16000 ×3, 400 半荘) ==="
if [[ -f "$log" ]]; then
  avg_rank=$(grep "^avg_rank=" "$log" | tail -1 | cut -d= -f2)
  fuuro=$(grep "^fuuro_rate=" "$log" | tail -1 | cut -d= -f2)
  riichi=$(grep "^riichi_rate=" "$log" | tail -1 | cut -d= -f2)
  agari=$(grep "^agari_rate=" "$log" | tail -1 | cut -d= -f2)
  houjuu=$(grep "^houjuu_rate=" "$log" | tail -1 | cut -d= -f2)
  ryukyoku=$(grep "^ryukyoku_rate=" "$log" | tail -1 | cut -d= -f2)
  echo "avg_rank=$avg_rank fuuro=$fuuro riichi=$riichi agari=$agari houjuu=$houjuu ryukyoku=$ryukyoku"
else
  echo "(missing log)"
fi
