#!/usr/bin/env bash
# grp_baseline (DQN) 相手の 1v3 eval バッテリー
# checkpoints: init / step16000
# challenger: seed [10000, 10100) = 100 seed * 4 hanchan/seed = 400 半荘/checkpoint
# (OneVsThree の席替え仕様は eval_grp_baseline_1v3.py 内コメント参照)
# stage2_design.md §5 レンズ2 (grp_baseline 対戦、DQN 時代の競技レンズ)
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
RUN_DIR="${RUN_DIR:-/home/gamba/mahjong/runs/ppo/stage1_20260706_020120_resume}"
CFG="$RUN_DIR/config.toml"
INIT_CKPT="/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth"
# GRP_BASELINE_CKPT デフォルト: beta1_huber_192x40/mortal.pth (2026-07-08 変更)。
# grp_baseline.pth は steps=0 の配管 fixture であり強さ測定に使用禁止
# (2026-07-08 判明: 無差別副露・fuuro 89%。検定(9)は配管回帰なので
# 壊れたモデルでも PASS し続けており、素性がどこにも文書化されて
# いなかったため誤用された。詳細は supervisor_handbook.md §4c 参照)。
export GRP_BASELINE_CKPT="${GRP_BASELINE_CKPT:-$INIT_CKPT}"
RESULTS_DIR="$RUN_DIR/logs/eval_grp_baseline"
mkdir -p "$RESULTS_DIR"

export PYTHONPATH="$REPO/mortal"
export PYTHONUNBUFFERED=1
export MORTAL_CFG="$CFG"

echo "=== grp_baseline 1v3 eval battery: $(date -Iseconds) ==="
echo "RUN_DIR=$RUN_DIR"

# Pre-checks
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

declare -A LABELS
LABELS=(
  [init]="$INIT_CKPT"
  [step16000]="$RUN_DIR/checkpoints/step_016000.pth"
)

ORDER=(init step16000)

for label in "${ORDER[@]}"; do
  ckpt="${LABELS[$label]}"
  log="$RESULTS_DIR/eval_${label}.log"
  echo ""
  echo "--- $label: $ckpt ---"
  echo "$(date -Iseconds) start $label"
  EVAL_LABEL="$label" \
  EVAL_CHECKPOINT="$ckpt" \
  conda run --no-capture-output -n mortal python "$REPO/freeparlor/scripts/eval_grp_baseline_1v3.py" 2>&1 | tee "$log"
  echo "$(date -Iseconds) done $label"
done

echo ""
echo "=== Battery complete: $(date -Iseconds) ==="
echo ""
echo "=== Summary (challenger視点 vs grp_baseline DQN, 1v3, 400 半荘/checkpoint) ==="
echo "label | avg_rank | fuuro | riichi | agari | houjuu | ryukyoku"
echo "------|----------|-------|--------|-------|--------|----------"
for label in "${ORDER[@]}"; do
  log="$RESULTS_DIR/eval_${label}.log"
  if [[ -f "$log" ]]; then
    avg_rank=$(grep "^avg_rank=" "$log" | tail -1 | cut -d= -f2)
    fuuro=$(grep "^fuuro_rate=" "$log" | tail -1 | cut -d= -f2)
    riichi=$(grep "^riichi_rate=" "$log" | tail -1 | cut -d= -f2)
    agari=$(grep "^agari_rate=" "$log" | tail -1 | cut -d= -f2)
    houjuu=$(grep "^houjuu_rate=" "$log" | tail -1 | cut -d= -f2)
    ryukyoku=$(grep "^ryukyoku_rate=" "$log" | tail -1 | cut -d= -f2)
    echo "$label | $avg_rank | $fuuro | $riichi | $agari | $houjuu | $ryukyoku"
  else
    echo "$label | (missing)"
  fi
done
