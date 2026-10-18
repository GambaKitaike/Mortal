#!/usr/bin/env bash
# libriichi rebuild + import smoke, shared by run launchers.
#
# 2026-07-08 ビルド事故対応: CARGO_TARGET_DIR の継承汚染
# （tmux セッション残留由来）でビルド先と cp 元が乖離した。
# 呼び出し元の環境がどうであれ、このスクリプト内では常に
# "$REPO/target" に明示ピンする。
#
# Usage: preflight_libriichi.sh [REPO]
#   REPO defaults to $REPO env var, then /home/gamba/mahjong/Mortal
set -euo pipefail

REPO="${1:-${REPO:-/home/gamba/mahjong/Mortal}}"

if [[ -n "${CARGO_TARGET_DIR:-}" && "$CARGO_TARGET_DIR" != "$REPO/target" ]]; then
  echo "WARNING: inherited CARGO_TARGET_DIR=$CARGO_TARGET_DIR overridden -> $REPO/target"
fi
CARGO_TARGET_DIR="$REPO/target"
export CARGO_TARGET_DIR

PYO3_PYTHON="$(conda run -n mortal python -c 'import sys; print(sys.executable)')"
export PYO3_PYTHON

echo "=== Pre-flight: rebuild libriichi.so ==="
echo "REPO=$REPO"
echo "CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
echo "PYO3_PYTHON=$PYO3_PYTHON"

if ! ( cd "$REPO" && cargo build --release -p libriichi --lib ); then
  echo "FATAL: cargo build failed" >&2
  exit 1
fi

if ! cp -f "$REPO/target/release/libriichi.so" "$REPO/mortal/libriichi.so"; then
  echo "FATAL: cp libriichi.so failed" >&2
  exit 1
fi

echo "=== Pre-flight: libriichi import check ==="
if ! PYTHONPATH="$REPO/mortal" conda run -n mortal python -c "import libriichi.arena; from libriichi.stat import Stat"; then
  echo "FATAL: libriichi.so broken — import smoke failed" >&2
  exit 1
fi

echo "libriichi preflight OK"
