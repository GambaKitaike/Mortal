#!/usr/bin/env bash
# 初期損傷プローブの eval（early_damage_probe_design.md §4・診断/判定非関与）
#
# `run_eval_trajectory_1v3.sh` の派生。違いは3点だけ:
#   1. checkpoint を **checkpoints_diag/** から取る（観測専用ストリーム。§2b）
#   2. n=400（seeds [10000,10100)）— 判定標準 n=800 の半分。**SE は約 1.41 倍**
#   3. init 脚を GPU で焼き直さず、既存 n=800 init 脚の **厳密な部分集合**として作る
#      （eval は seed 決定論なので seeds [10000,10100) の 400 件は同一の産物。
#      焼き直しても同じものが出るだけで GPU を無駄にする）
#
# 使い方:
#   RUN_DIR=/home/gamba/mahjong/runs/ppo/early_probe_<日時> \
#   INIT_SRC=/home/gamba/mahjong/runs/ppo/anchor_k_20260727_000805/logs/eval_grp_baseline/game_logs_init \
#   bash run_eval_early_probe.sh
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
# 設計 §4 は step 200 刻みの 10 点。§4a amendment（2026-07-28・インシデント起因）で
# 終端のみ 2000 -> 1900（step_002000.pth が完走時の cleanup で切り詰められたため）。
STEPS="${STEPS:-200 400 600 800 1000 1200 1400 1600 1800 1900}"
N_SEEDS="${N_SEEDS:-100}"          # 100 seeds × 4 席 = 400 半荘
INIT_CKPT="/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth"
INIT_SRC="${INIT_SRC:-/home/gamba/mahjong/runs/ppo/anchor_k_20260727_000805/logs/eval_grp_baseline/game_logs_init}"
export GRP_BASELINE_CKPT="${GRP_BASELINE_CKPT:-$INIT_CKPT}"
RESULTS_DIR="$RUN_DIR/logs/eval_grp_baseline"
SEED_LO=10000
SEED_HI=$((SEED_LO + N_SEEDS))

gpu_busy_list() {
  ps -eo comm=,pid=,args= | awk '($1=="python"||$1=="python3") && $0 ~ /(run_train_ppo|run_client|run_server|drca_run_probe|eval_ppo_smoke_sanity|eval_grp_baseline_1v3|eval_meta_stage1_vs_stage2)\.py/'
}
if [[ -n "$(gpu_busy_list)" ]]; then
  echo "FATAL: 学習/eval プロセスが稼働中 (GPU 1系統ルール)" >&2
  gpu_busy_list | head >&2; exit 1
fi
if ss -tlnp 2>/dev/null | grep -q 5000; then echo "FATAL: port 5000 in use" >&2; exit 1; fi

mkdir -p "$RESULTS_DIR"

# --- init 脚: 既存 n=800 からの決定論的部分集合（GPU 不要） ---
if [[ ! -d "$RESULTS_DIR/game_logs_init" ]]; then
  [[ -d "$INIT_SRC" ]] || { echo "FATAL: INIT_SRC が無い: $INIT_SRC" >&2; exit 1; }
  mkdir -p "$RESULTS_DIR/game_logs_init"
  n=0
  for f in "$INIT_SRC"/*.json.gz; do
    seed=${f##*/}; seed=${seed%%_*}
    if (( seed >= SEED_LO && seed < SEED_HI )); then
      cp "$f" "$RESULTS_DIR/game_logs_init/"; n=$((n + 1))
    fi
  done
  if (( n != N_SEEDS * 4 )); then
    echo "FATAL: init 脚 $n 件（期待 $((N_SEEDS * 4))）。INIT_SRC の seed 範囲を確認" >&2
    exit 1
  fi
  echo "init 脚: $INIT_SRC から seeds [$SEED_LO,$SEED_HI) の $n 件を複製（GPU 不使用）"
else
  echo "init 脚: 既存 $(ls "$RESULTS_DIR/game_logs_init" | wc -l) 件を再利用"
fi

# init 脚の pnl（集計スクリプトが差分の基準として読む）。CPU のみ。
if [[ ! -f "$RESULTS_DIR/pnl_init.txt" ]]; then
  conda run --no-capture-output -n mortal python \
    "$REPO/freeparlor/scripts/analyze_freeparlor_pnl_1v3.py" \
    "$RESULTS_DIR/game_logs_init" -o "$RESULTS_DIR/pnl_init.txt" >/dev/null
  echo "init 脚の pnl を書き出した: $RESULTS_DIR/pnl_init.txt"
fi

echo "=== 初期損傷プローブ eval: $(date -Iseconds) ==="
echo "RUN_DIR=$RUN_DIR / STEPS=$STEPS / n=$((N_SEEDS * 4)) 半荘 / seeds [$SEED_LO,$SEED_HI)"
"$REPO/freeparlor/scripts/preflight_libriichi.sh" "$REPO"

export PYTHONPATH="$REPO/mortal" PYTHONUNBUFFERED=1 MORTAL_CFG="$RUN_DIR/config.toml"
export EVAL_SEED_COUNT="$N_SEEDS"

for s in $STEPS; do
  label="step$s"
  ckpt="$RUN_DIR/checkpoints_diag/step_$(printf '%06d' "$s").pth"
  [[ -f "$ckpt" ]] || { echo "FATAL: $ckpt not found（diag ストリームが無い）" >&2; exit 1; }
  if [[ -d "$RESULTS_DIR/game_logs_$label" ]] && \
     [[ $(ls "$RESULTS_DIR/game_logs_$label" | wc -l) -eq $((N_SEEDS * 4)) ]]; then
    echo "--- $label: 既存 $((N_SEEDS * 4)) 件をスキップ ---"; continue
  fi
  echo ""; echo "--- $label ---"; date -Iseconds
  EVAL_LABEL="$label" EVAL_CHECKPOINT="$ckpt" \
    conda run --no-capture-output -n mortal python \
    "$REPO/freeparlor/scripts/eval_grp_baseline_1v3.py" 2>&1 | tee "$RESULTS_DIR/eval_${label}.log"
  conda run --no-capture-output -n mortal python \
    "$REPO/freeparlor/scripts/analyze_freeparlor_pnl_1v3.py" \
    "$RESULTS_DIR/game_logs_${label}" -o "$RESULTS_DIR/pnl_${label}.txt" >/dev/null
done

echo ""; echo "=== eval 完了: $(date -Iseconds) ==="
echo "集計: python freeparlor/scripts/aggregate_early_probe.py --run-dir $RUN_DIR"
