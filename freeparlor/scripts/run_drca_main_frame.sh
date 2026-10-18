#!/usr/bin/env bash
# run_drca_main_frame.sh — DRCA 本測定 残枠 launcher（§5a-1b 適用後の残4枠）
#
# Usage:
#   bash freeparlor/scripts/run_drca_main_frame.sh <frame> [--dry-run]
#   frame ∈ a_init | a_s3final | a_s3mid | b_s3final
#
# 実施順（drca_probe_design.md §5a-1b）:
#   a_init → a_s3final → a_s3mid → b_s3final
#   （セット(b)×init は §5a-1b の 48h 条項機械適用で脱落済み）
#
# 発進規律（第1枠 main_s1final_20260715_075837 と同一）:
#   - 事前に `conda activate mortal` 済みであること
#   - 発進前 preflight（libriichi 鮮度 + `verify_ppo_p1.py` 全検定 PASS）を
#     ローカル WSL で済ませてから実行すること（本スクリプトが検査するのは
#     checkpoint 実在・ディスク・残党プロセスのみ）
#   - GPU 1系統: 他の DRCA/学習プロセス稼働中は FATAL
#
# --dry-run: 解決済み構成の表示と preflight（checkpoint → ディスク → 残党）まで
#   実行して終了。run dir 作成・採取・probe 発進は行わない。
set -euo pipefail

FRAME="${1:-}"
DRY_RUN=0
if [[ "${2:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

REPO="/home/gamba/mahjong/Mortal"
RUNS_ROOT="/home/gamba/mahjong/runs"
CFG="$REPO/freeparlor/configs/ppo_drca_probe.toml"

CKPT_INIT="/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth"
CKPT_S1FINAL="/home/gamba/mahjong/runs/ppo/stage1_20260706_020120_resume/checkpoints/step_016000.pth"
CKPT_S3MID="/home/gamba/mahjong/runs/ppo/stage3_20260712_033403/checkpoints/step_008000.pth"
CKPT_S3FINAL="/home/gamba/mahjong/runs/ppo/stage3_20260712_033403/checkpoints/step_016000.pth"

# §5a-1a / §5a-4 凍結値
N_BP=485
K=8
SEED_BASE=20260713   # 枠ごとに凍結基点へ戻す = 全 checkpoint が同一配牌母集団から採取（§5a-4）
TORCH_SEED=20260713
EXTRACT_SEED=713
DISK_MIN_GB="${DISK_MIN_GB:-100}"

case "$FRAME" in
  a_init)
    MODE=a; COLLECT_CKPT="$CKPT_INIT";    PROBE_CKPT="$CKPT_INIT";    REF_CKPT=""; SEAT_ONLY=0 ;;
  a_s3final)
    MODE=a; COLLECT_CKPT="$CKPT_S3FINAL"; PROBE_CKPT="$CKPT_S3FINAL"; REF_CKPT=""; SEAT_ONLY=0 ;;
  a_s3mid)
    MODE=a; COLLECT_CKPT="$CKPT_S3MID";   PROBE_CKPT="$CKPT_S3MID";   REF_CKPT=""; SEAT_ONLY=0 ;;
  b_s3final)
    # セット(b): 採取は参照方策 Stage1-16000 の self-play（--challenger-seat-only）、
    # probe は trainee=Stage3-16000 / 他3席=Stage1-16000（§5a-3）
    MODE=b; COLLECT_CKPT="$CKPT_S1FINAL"; PROBE_CKPT="$CKPT_S3FINAL"; REF_CKPT="$CKPT_S1FINAL"; SEAT_ONLY=1 ;;
  "")
    echo "FATAL: frame argument required (a_init|a_s3final|a_s3mid|b_s3final)"
    exit 1 ;;
  *)
    echo "FATAL: unknown frame '$FRAME' (a_init|a_s3final|a_s3mid|b_s3final)"
    exit 1 ;;
esac

TS=$(date +%Y%m%d_%H%M%S)
MAIN_DIR="$RUNS_ROOT/drca/main_${FRAME}_${TS}"
SESSION="drca_main_${FRAME}_${TS}"

echo "=== resolved config (frame=$FRAME) ==="
echo "MODE=$MODE"
echo "COLLECT_CKPT=$COLLECT_CKPT"
echo "PROBE_CKPT=$PROBE_CKPT"
echo "REF_CKPT=${REF_CKPT:-<none>}"
echo "SEAT_ONLY=$SEAT_ONLY"
echo "N_BP=$N_BP K=$K SEED_BASE=$SEED_BASE TORCH_SEED=$TORCH_SEED EXTRACT_SEED=$EXTRACT_SEED"
echo "MAIN_DIR=$MAIN_DIR"
echo "TMUX_SESSION=$SESSION"
echo "SHARDS=162/162/161"

# --- preflight 1: checkpoint / config 実在 ---
for f in "$COLLECT_CKPT" "$PROBE_CKPT" "$CFG"; do
  if [[ ! -f "$f" ]]; then
    echo "FATAL: not found: $f"
    exit 1
  fi
done
if [[ -n "$REF_CKPT" && ! -f "$REF_CKPT" ]]; then
  echo "FATAL: not found: $REF_CKPT"
  exit 1
fi
echo "PREFLIGHT: checkpoints/config exist OK"

# --- preflight 2: ディスク（2026-07-09 枯渇インシデント再発防止）---
AVAIL_GB=$(df -BG --output=avail "$RUNS_ROOT" | tail -1 | tr -dc '0-9')
if (( AVAIL_GB < DISK_MIN_GB )); then
  echo "FATAL: disk avail ${AVAIL_GB}GB < DISK_MIN_GB=${DISK_MIN_GB}GB on $RUNS_ROOT"
  exit 1
fi
echo "PREFLIGHT: disk avail ${AVAIL_GB}GB >= ${DISK_MIN_GB}GB OK"

# --- preflight 3: 残党チェック（GPU 1系統）---
if pgrep -f 'drca_run_probe|drca_collect_branchpoints|train_ppo' >/dev/null; then
  echo "FATAL: DRCA/学習プロセスが稼働中（GPU 1系統ルール）:"
  pgrep -af 'drca_run_probe|drca_collect_branchpoints|train_ppo' || true
  exit 1
fi
echo "PREFLIGHT: no residual DRCA/training processes OK"

if (( DRY_RUN )); then
  echo "DRY-RUN OK: all preflight checks passed, not launching"
  exit 0
fi

# --- 発進 ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mortal
CONDA_BASE=$(conda info --base)

export MORTAL_CFG="$CFG"
export PYTHONPATH="$REPO/mortal"
cd "$REPO"

mkdir -p "$MAIN_DIR"
exec > >(tee -a "$MAIN_DIR/main.log") 2>&1

echo "=== DRCA main_${FRAME} collect+launch $(date -Is) ==="

SEAT_ARG=""
if (( SEAT_ONLY )); then
  SEAT_ARG="--challenger-seat-only"
fi
REF_ARG=""
if [[ -n "$REF_CKPT" ]]; then
  REF_ARG="--reference-checkpoint '$REF_CKPT'"
fi

# --- 採取 ---
COLLECT_START=$(date +%s)
# shellcheck disable=SC2086
python freeparlor/scripts/drca_collect_branchpoints.py \
  --checkpoint "$COLLECT_CKPT" \
  --n "$N_BP" \
  --seed-base "$SEED_BASE" \
  --torch-seed "$TORCH_SEED" \
  --extract-seed "$EXTRACT_SEED" \
  $SEAT_ARG \
  --out "$MAIN_DIR/bp.jsonl" \
  --log-dir "$MAIN_DIR/bp.logs" \
  2>&1 | tee "$MAIN_DIR/collect.log"
COLLECT_END=$(date +%s)
N_COLLECTED=$(wc -l < "$MAIN_DIR/bp.jsonl")
echo "COLLECT_DONE: n=$N_COLLECTED wall=$((COLLECT_END - COLLECT_START))s"
echo "{\"collect_start\": $COLLECT_START, \"collect_end\": $COLLECT_END, \"n\": $N_COLLECTED}" > "$MAIN_DIR/timing.json"

# --- shard 分割 162/162/161 ---
python3 - "$MAIN_DIR/bp.jsonl" "$MAIN_DIR" <<'PY'
import json, sys
from pathlib import Path

src = Path(sys.argv[1])
main = Path(sys.argv[2])
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
assert len(rows) == 485, f"expected 485 branch points, got {len(rows)}"
sizes = [162, 162, 161]
start = 0
for i, sz in enumerate(sizes):
    shard = rows[start:start+sz]
    out = main / f"bp_shard{i}.jsonl"
    out.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in shard) + '\n')
    print(f"shard{i}: {len(shard)} -> {out}")
    start += sz
PY

# --- tmux ペインコマンド（発進と resume 生成の単一ソース）---
pane_cmd() {
  local i="$1" extra="$2"
  echo "source $CONDA_BASE/etc/profile.d/conda.sh && conda activate mortal && export MORTAL_CFG='$CFG' PYTHONPATH='$REPO/mortal' && cd '$REPO' && python freeparlor/scripts/drca_run_probe.py --branchpoints '$MAIN_DIR/bp_shard${i}.jsonl' --checkpoint '$PROBE_CKPT' --mode $MODE $REF_ARG --k $K --parallel 1 $extra --out '$MAIN_DIR/probe_shard${i}.jsonl' --tmp-root '$MAIN_DIR/probe_shard${i}.tmp' 2>&1 | tee -a '$MAIN_DIR/probe_shard${i}.log'; echo shard${i} exit=\$?; exec bash"
}

# --- 停止・再開スクリプトを run dir に自動生成（第1枠で実証済みのパターン）---
cat > "$MAIN_DIR/stop_for_relocation.sh" <<EOF
#!/usr/bin/env bash
# Stop procedure for frame ${FRAME} (auto-generated by run_drca_main_frame.sh)
set -euo pipefail

MAIN_DIR="$MAIN_DIR"
SESSION="$SESSION"
BACKUP_ROOT="/home/gamba/mahjong/backups/drca"
STOP_TS=\$(date +%Y%m%d_%H%M%S)

mkdir -p "\$BACKUP_ROOT"
echo "=== DRCA stop \$(date -Is) ==="

for i in 0 1 2; do
  OUT="\$MAIN_DIR/probe_shard\${i}.jsonl"
  if [[ -f "\$OUT" ]]; then
    N=\$(wc -l < "\$OUT")
    BP=\$(( (N + 15) / 16 ))  # 2 arms x K=8 = 16 rollouts per branch point
    echo "shard\${i}: rollouts=\$N completed_branch_points~=\$BP"
  else
    echo "shard\${i}: no output yet"
  fi
done

tmux kill-session -t "\$SESSION" 2>/dev/null || true
pkill -f "drca_run_probe.py.*main_${FRAME}_${TS}" 2>/dev/null || true

tar -czf "\$BACKUP_ROOT/main_${FRAME}_${TS}_\${STOP_TS}.tar.gz" -C "\$(dirname "\$MAIN_DIR")" "\$(basename "\$MAIN_DIR")"
echo "BACKUP: \$BACKUP_ROOT/main_${FRAME}_${TS}_\${STOP_TS}.tar.gz"
echo "=== Resume: bash \$MAIN_DIR/run_resume.sh ==="
EOF
chmod +x "$MAIN_DIR/stop_for_relocation.sh"

# resume スクリプト内では pane 文字列が再度 double-quote 展開されるため、
# `$?` を `\$?` へエスケープしてから埋め込む（二重展開防止）
PANE_R0=$(pane_cmd 0 --resume | sed 's/\$?/\\$?/g')
PANE_R1=$(pane_cmd 1 --resume | sed 's/\$?/\\$?/g')
PANE_R2=$(pane_cmd 2 --resume | sed 's/\$?/\\$?/g')

cat > "$MAIN_DIR/run_resume.sh" <<EOF
#!/usr/bin/env bash
# Resume probes for frame ${FRAME} after interruption (auto-generated, --resume per shard)
set -euo pipefail

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n probe "$PANE_R0"
tmux split-window -h -t "$SESSION:probe" "$PANE_R1"
tmux split-window -v -t "$SESSION:probe.1" "$PANE_R2"
tmux select-layout -t "$SESSION:probe" even-horizontal
echo "RESUMED tmux_session=$SESSION \$(date -Is)"
EOF
chmod +x "$MAIN_DIR/run_resume.sh"
echo "GENERATED: $MAIN_DIR/stop_for_relocation.sh / run_resume.sh"

# --- tmux: 3 probe プロセス並走 ---
PROBE_START=$(date +%s)
echo "$PROBE_START" > "$MAIN_DIR/probe_start_epoch.txt"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n probe "$(pane_cmd 0 "")"
tmux split-window -h -t "$SESSION:probe" "$(pane_cmd 1 "")"
tmux split-window -v -t "$SESSION:probe.1" "$(pane_cmd 2 "")"
tmux select-layout -t "$SESSION:probe" even-horizontal

echo "PROBE_LAUNCHED tmux_session=$SESSION probe_start_epoch=$PROBE_START"
echo "=== collect+launch script done $(date -Is) ==="
