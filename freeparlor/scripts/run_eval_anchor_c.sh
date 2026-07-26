#!/usr/bin/env bash
# anchor Arm C — eval バッテリー一括実行 (anchored_ppo_design.md §9「eval バッテリー」)
#
# 既存ハーネスを env で再利用するオーケストレータ。**新規の測定ロジックを持たない**
# (レンズ実装の複製禁止)。GPU 1系統ルールのため全レンズを直列実行する。
#
#   レンズ1: argmax eval 6 checkpoint  (run_eval_battery_stage3.sh を RUN_DIR 差し替えで再利用)
#   レンズ2: grp_baseline 1v3 **n=800 両脚** (EVAL_SEED_COUNT=200 = seeds [10000,10200))
#            + 収支 pnl + **基礎技能の有意性パス** (判定1 の測定器)
#   較正脚: ミラー較正 (バックログ5 の初適用。理論ミラー値からの逸脱のゼロ点確認)
#   レンズ3: メタ対決 anchor_c-16000 vs init ×3 (§6-4)
#
# 判定はしない。数値を出すだけ (判定は anchored_ppo_design.md §6 に従い監督側の別タスク)。
# レンズ4 (Gamba の定性レビュー) は本スクリプトの後・判定の前に通すこと
# (`qualitative_review_protocol.md`)。
#
# 使い方:  bash freeparlor/scripts/run_eval_anchor_c.sh [--dry-run]
set -euo pipefail

REPO="/home/gamba/mahjong/Mortal"
RUN_DIR="${RUN_DIR:-/home/gamba/mahjong/runs/ppo/anchor_c_20260726_171144_resume}"
INIT_CKPT="/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth"
FINAL_CKPT="$RUN_DIR/checkpoints/step_016000.pth"
# 判定条件 (§6): 1v3 両脚 n=800 = seeds [10000,10200)、4 半荘/seed
N_SEEDS="${N_SEEDS:-200}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

echo "=== anchor Arm C eval battery: $(date -Iseconds) ==="
echo "RUN_DIR=$RUN_DIR"
echo "FINAL_CKPT=$FINAL_CKPT"
echo "1v3 seeds=[10000, $((10000 + N_SEEDS))) => $((N_SEEDS * 4)) 半荘/脚"

# ---- preflight ----
for f in "$INIT_CKPT" "$FINAL_CKPT" "$RUN_DIR/config.toml"; do
  [[ -f "$f" ]] || { echo "FATAL: not found: $f" >&2; exit 1; }
done
echo "checkpoint/config 実在 OK"

if pgrep -f "run_train_ppo.py|run_client.py|run_server.py|drca_run_probe.py" >/dev/null 2>&1; then
  echo "FATAL: 学習/測定プロセスが稼働中 (GPU 1系統ルール)。完走を確認してから実行すること" >&2
  pgrep -af "run_train_ppo.py|run_client.py|run_server.py|drca_run_probe.py" | head >&2
  exit 1
fi
echo "残党なし"

if ss -tlnp 2>/dev/null | grep -q 5000; then
  echo "FATAL: port 5000 in use" >&2; exit 1
fi
echo "port 5000 clear"

if (( DRY_RUN )); then
  echo "--dry-run: 構成解決と preflight のみで終了 (eval は実行しない)"
  exit 0
fi

"$REPO/freeparlor/scripts/preflight_libriichi.sh" "$REPO"

GRP_DIR="$RUN_DIR/logs/eval_grp_baseline"

# ---- レンズ1: argmax eval (6 checkpoint) ----
echo ""
echo "########## レンズ1: argmax eval (6 checkpoint) ##########"
RUN_DIR="$RUN_DIR" bash "$REPO/freeparlor/scripts/run_eval_battery_stage3.sh"

# ---- レンズ2: grp_baseline 1v3 (n=800 両脚) ----
echo ""
echo "########## レンズ2: grp_baseline 1v3 (n=$((N_SEEDS * 4)) 両脚) ##########"
RUN_DIR="$RUN_DIR" EVAL_SEED_COUNT="$N_SEEDS" \
  bash "$REPO/freeparlor/scripts/run_eval_grp_baseline_1v3_stage3.sh"

# ---- 判定1 の測定器: 基礎技能の有意性パス ----
echo ""
echo "########## 判定1 の測定器: 基礎技能 (放銃/和了/着順) の有意性 ##########"
PYTHONPATH="$REPO/mortal" conda run --no-capture-output -n mortal python \
  "$REPO/freeparlor/scripts/analyze_fundamentals_1v3.py" \
  --init-logs "$GRP_DIR/game_logs_init" \
  --ckpt-logs "$GRP_DIR/game_logs_step16000" \
  --label anchor_c_16k 2>&1 | tee "$GRP_DIR/fundamentals_anchor_c.txt"

# ---- 較正脚: ミラー較正 (バックログ5 初適用) ----
echo ""
echo "########## 較正脚: ミラー較正 (anchor_c-16000 を両席に) ##########"
REFERENCE_CKPT="$FINAL_CKPT" REFERENCE_RUN_DIR="$RUN_DIR" \
  bash "$REPO/freeparlor/scripts/run_eval_meta_mirror.sh"

# ---- レンズ3: メタ対決 (anchor_c-16000 vs init ×3) ----
echo ""
echo "########## レンズ3: メタ対決 anchor_c-16000 vs init ×3 ##########"
META_DIR="$RUN_DIR/logs/eval_meta"
mkdir -p "$META_DIR"
label="meta_anchorC16000_vs_init"
export PYTHONPATH="$REPO/mortal" PYTHONUNBUFFERED=1 MORTAL_CFG="$RUN_DIR/config.toml"
EVAL_LABEL="$label" \
EVAL_CHALLENGER_CHECKPOINT="$FINAL_CKPT" \
EVAL_BASELINE_CHECKPOINT="$INIT_CKPT" \
conda run --no-capture-output -n mortal python \
  "$REPO/freeparlor/scripts/eval_meta_stage1_vs_stage2.py" 2>&1 | tee "$META_DIR/eval_${label}.log"
conda run --no-capture-output -n mortal python \
  "$REPO/freeparlor/scripts/analyze_freeparlor_pnl_1v3.py" \
  "$META_DIR/game_logs_${label}" -o "$META_DIR/pnl_${label}.txt" 2>&1 \
  | tee "$META_DIR/pnl_${label}.log"

echo ""
echo "=== eval battery complete: $(date -Iseconds) ==="
echo "次: レンズ4 (Gamba の牌譜定性レビュー) -> その後に §6 判定"
echo "  牌譜 HTML: python freeparlor/scripts/mjai_log_to_html.py <game_logs dir>"
