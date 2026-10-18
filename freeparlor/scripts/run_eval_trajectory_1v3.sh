#!/usr/bin/env bash
# 中間 checkpoint の 1v3 軌跡測定（探索的診断・判定非関与）
#
# 動機: anchor C/K の判定は step16000 の1点のみ（`anchored_ppo_design.md` §6）。
# 「劣化がいつ起きたか」「判定1（基礎維持）と判定2（チップ獲得）を両立する
# 中間 checkpoint が実在するか」は判定が答えていない。本スクリプトは既存の
# 1v3 ハーネスを中間 checkpoint に当てて軌跡を出す。
#
# **判定条件を変更しない。** 判定は step16000 で確定済みであり、本測定は
# 探索的診断（`CLAUDE.md`「層別分析は exploratory」と同じ扱い）。
#
# init 脚は決定論的に同一なので再実行しない（既存の game_logs_init を使い回す）。
#
# 使い方: RUN_DIR=<run dir> STEPS="4000 8000 12000" bash run_eval_trajectory_1v3.sh
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
STEPS="${STEPS:-4000 8000 12000}"
N_SEEDS="${N_SEEDS:-200}"          # 判定と同じ n=800（seeds [10000,10200)）
INIT_CKPT="/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth"
export GRP_BASELINE_CKPT="${GRP_BASELINE_CKPT:-$INIT_CKPT}"
RESULTS_DIR="$RUN_DIR/logs/eval_grp_baseline"

gpu_busy_list() {
  ps -eo comm=,pid=,args= | awk '($1=="python"||$1=="python3") && $0 ~ /(run_train_ppo|run_client|run_server|drca_run_probe|eval_ppo_smoke_sanity|eval_grp_baseline_1v3|eval_meta_stage1_vs_stage2)\.py/'
}
if [[ -n "$(gpu_busy_list)" ]]; then
  echo "FATAL: 学習/eval プロセスが稼働中 (GPU 1系統ルール)" >&2
  gpu_busy_list | head >&2; exit 1
fi
if ss -tlnp 2>/dev/null | grep -q 5000; then echo "FATAL: port 5000 in use" >&2; exit 1; fi
[[ -d "$RESULTS_DIR/game_logs_init" ]] || { echo "FATAL: init 脚が無い: $RESULTS_DIR/game_logs_init" >&2; exit 1; }

echo "=== 1v3 軌跡測定: $(date -Iseconds) ==="
echo "RUN_DIR=$RUN_DIR / STEPS=$STEPS / n=$((N_SEEDS * 4)) 半荘"
"$REPO/freeparlor/scripts/preflight_libriichi.sh" "$REPO"

export PYTHONPATH="$REPO/mortal" PYTHONUNBUFFERED=1 MORTAL_CFG="$RUN_DIR/config.toml"
export EVAL_SEED_COUNT="$N_SEEDS"

for s in $STEPS; do
  label="step$s"
  ckpt="$RUN_DIR/checkpoints/step_$(printf '%06d' "$s").pth"
  [[ -f "$ckpt" ]] || { echo "FATAL: $ckpt not found" >&2; exit 1; }
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

echo ""; echo "=== 軌跡測定完了: $(date -Iseconds) ==="
