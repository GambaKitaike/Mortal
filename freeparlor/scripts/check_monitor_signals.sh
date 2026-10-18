#!/usr/bin/env bash
# Isolated replica of the inner.sh monitor counting + FATAL branches.
# Verifies the backlog-11 additions fire on the real client.py warning strings
# and stay silent on the known-benign 'loader size delta' INFO.
set -uo pipefail

SRC="/home/gamba/mahjong/Mortal/freeparlor/scripts/run_ppo_p3_stage1_inner.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Extract the real count_monitor_metrics() body from the launcher so the test
# cannot drift from the implementation.
sed -n '/^count_monitor_metrics() {$/,/^}$/p' "$SRC" > "$WORK/counters.sh"
if [[ ! -s "$WORK/counters.sh" ]]; then
  echo "FAIL: could not extract count_monitor_metrics from $SRC"
  exit 1
fi
# shellcheck disable=SC1090
source "$WORK/counters.sh"

run_case() {
  local name="$1" ; shift
  local expect_code="$1" ; shift
  local LOG_DIR="$WORK/$name"
  mkdir -p "$LOG_DIR"
  : > "$LOG_DIR/client0.log"
  for line in "$@"; do echo "$line" >> "$LOG_DIR/client0.log"; done

  count_monitor_metrics

  local code=0
  if   (( MON_MISMATCH > 0 )); then code=4
  elif (( MON_KEYMISS  > 0 )); then code=9
  elif (( MON_ORPHAN   > 0 )); then code=10
  elif (( MON_FALLBACK > 0 )); then code=5
  elif (( MON_CHIP     > 0 )); then code=2
  fi

  if (( code == expect_code )); then
    echo "PASS $name -> exit $code (keymiss=$MON_KEYMISS orphan=$MON_ORPHAN" \
         "fallback=$MON_FALLBACK chip=$MON_CHIP loader_delta=$MON_LOADER_DELTA" \
         "mismatch=$MON_MISMATCH)"
  else
    echo "FAIL $name -> exit $code, expected $expect_code"
    FAILED=1
  fi
}

FAILED=0

# Exact strings emitted by mortal/client.py:108 and :184.
KEYMISS="2026-07-28 01:00:00,000 WARNING client.py:108 trajectory game key missing, skipping game client=client0 game_key=10000_8192_a expected_game_size=131 actual_steps=0 pending_had_key=False pending_partial_match=0 seed=10000 split=8192 trainee_seat=0 file=/x.json.gz"
ORPHAN="2026-07-28 01:00:00,000 WARNING client.py:184 trajectory orphan steps (17 steps) for game_id=10000_8192_a client=client0"
DELTA="2026-07-28 01:00:00,000 INFO client.py:130 trajectory loader size delta=-1 client=client0 game_key=10000_8192_a loader_game_size=131 recorded_steps=130 seed=10000 split=8192 trainee_seat=0 file=/x.json.gz"
FB="2026-07-28 01:00:00,000 WARNING client.py:253 illegal_action_fallback_count=3 (expected 0)"
CHIP="2026-07-28 01:00:00,000 ERROR online chip resolution failed for game_key=10000_8192_a"
OK="2026-07-28 01:00:00,000 INFO client.py:255 illegal_action_fallback_count=0"

run_case healthy            0  "$OK"
run_case loader_delta_only  0  "$DELTA" "$DELTA" "$OK"
run_case key_missing        9  "$OK" "$KEYMISS"
run_case orphan_steps      10  "$OK" "$ORPHAN"
run_case both_new           9  "$KEYMISS" "$ORPHAN"
run_case fallback           5  "$FB"
run_case chip               2  "$CHIP"

# Regression: the legacy tripwire must still be wired, and must read 0 on a
# corpus that contains every live signal (i.e. it is genuinely dead, not just
# unwatched).
run_case legacy_dead        9  "$KEYMISS" "$ORPHAN" "$DELTA"
if (( MON_MISMATCH != 0 )); then
  echo "FAIL legacy tripwire should be 0 on a corpus with no emitter"
  FAILED=1
else
  echo "PASS legacy tripwire reads 0 (no emitter in repo)"
fi

# Positive control: if an emitter were ever restored, exit 4 must still fire.
run_case legacy_restored    4  "WARNING trajectory step count mismatch: 5 vs 6"

if (( FAILED )); then echo "=== SOME CHECKS FAILED ==="; exit 1; fi
echo "=== ALL MONITOR CHECKS PASSED ==="
