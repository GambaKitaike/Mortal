#!/usr/bin/env bash
# PPO P3 Stage1 完走後 eval バッテリー
# checkpoints: init / 2000 / 4000 / 8000 / 12000 / 16000
# seeds [10000, 10100), 100 半荘直列実行
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
RUN_DIR="${RUN_DIR:-/home/gamba/mahjong/runs/ppo/stage1_20260706_020120_resume}"
CFG="$RUN_DIR/config.toml"
INIT_CKPT="/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth"
RESULTS_DIR="$RUN_DIR/logs/eval_battery"
mkdir -p "$RESULTS_DIR"

export PYTHONPATH="$REPO/mortal"
export PYTHONUNBUFFERED=1
export MORTAL_CFG="$CFG"

echo "=== Stage1 eval battery: $(date -Iseconds) ==="
echo "RUN_DIR=$RUN_DIR"

# Pre-checks
if ss -tlnp 2>/dev/null | grep -q 5000; then
  echo "ERROR: port 5000 in use — abort" >&2; exit 1
fi
echo "port 5000 clear"

declare -A LABELS
LABELS=(
  [init]="$INIT_CKPT"
  [step2000]="$RUN_DIR/checkpoints/step_002000.pth"
  [step4000]="$RUN_DIR/checkpoints/step_004000.pth"
  [step8000]="$RUN_DIR/checkpoints/step_008000.pth"
  [step12000]="$RUN_DIR/checkpoints/step_012000.pth"
  [step16000]="$RUN_DIR/checkpoints/step_016000.pth"
)

ORDER=(init step2000 step4000 step8000 step12000 step16000)

for label in "${ORDER[@]}"; do
  ckpt="${LABELS[$label]}"
  log="$RESULTS_DIR/eval_${label}.log"
  echo ""
  echo "--- $label: $ckpt ---"
  echo "$(date -Iseconds) start $label"
  EVAL_LABEL="$label" \
  EVAL_CHECKPOINT="$ckpt" \
  conda run --no-capture-output -n mortal python "$REPO/freeparlor/scripts/eval_ppo_smoke_sanity.py" 2>&1 | tee "$log"
  echo "$(date -Iseconds) done $label"
done

echo ""
echo "=== Battery complete: $(date -Iseconds) ==="
echo ""
echo "=== Summary ==="
echo "label | fuuro | riichi | agari | houjuu | ryukyoku | avg_rank"
echo "------|-------|--------|-------|--------|----------|----------"
for label in "${ORDER[@]}"; do
  log="$RESULTS_DIR/eval_${label}.log"
  if [[ -f "$log" ]]; then
    fuuro=$(grep "^fuuro_rate=" "$log" | tail -1 | cut -d= -f2)
    riichi=$(grep "^riichi_rate=" "$log" | tail -1 | cut -d= -f2)
    agari=$(grep "^agari_rate=" "$log" | tail -1 | cut -d= -f2)
    houjuu=$(grep "^houjuu_rate=" "$log" | tail -1 | cut -d= -f2)
    ryukyoku=$(grep "^ryukyoku_rate=" "$log" | tail -1 | cut -d= -f2)
    avg_rank=$(grep "^avg_rank=" "$log" | tail -1 | cut -d= -f2)
    echo "$label | $fuuro | $riichi | $agari | $houjuu | $ryukyoku | $avg_rank"
  else
    echo "$label | (missing)"
  fi
done
