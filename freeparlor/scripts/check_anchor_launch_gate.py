#!/usr/bin/env python3
"""Anchor 系列 発進ゲート — 機械ゲートのみ (anchored_ppo_design.md §5)。

使い方:
  # Arm C: pool draw ログから anchor 採択率を検証
  python3 check_anchor_launch_gate.py --arm c <run_dir> [--anchor-prob 0.25]

  # Arm K: ppo_diag.jsonl から kl_anchor イベントを検証
  python3 check_anchor_launch_gate.py --arm k <run_dir_or_diag.jsonl> [--kl-beta 0.1]

Arm C ログ形式 (run_dir/logs/pool_draw_*.jsonl):
  1行1 JSON。必須キー: event='pool_draw', draw_kind ('anchor'|'latest'|'past'|'fallback')。
  採択率 = draw_kind=='anchor' の行数 / 全 pool_draw 行数。
  @step200 時点で全 client ログを glob して合算する。

Arm K ログ形式 (run_dir/logs/ppo_diag.jsonl):
  event='kl_anchor' レコード。必須キー: kl_beta, kl_ref_mean, kl_term_total。
  trainer_step in [0, 200] の全バッチで kl_beta=設定値・有限値・NaN/inf なし。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

MECH_HI = 200
ANCHOR_RATE_TOL = 0.05


def _load_jsonl(path: Path) -> list[dict]:
    recs = []
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def run_arm_c(run_dir: Path, anchor_prob: float):
    draw_files = sorted(run_dir.glob('logs/pool_draw_*.jsonl'))
    print(f'pool_draw ログファイル: {len(draw_files)}')
    for p in draw_files:
        print(f'  {p}')

    if not draw_files:
        print('FATAL: pool_draw_*.jsonl が見つからない -> FAIL')
        sys.exit(1)

    draws = []
    for p in draw_files:
        for rec in _load_jsonl(p):
            if rec.get('event') == 'pool_draw':
                draws.append(rec)

    print(f'\n=== Anchor Arm C 機械ゲート (@step{MECH_HI}) ===')
    print(f'pool_draw レコード総数: {len(draws)}')

    if not draws:
        print('FATAL: pool_draw レコード 0 件 -> FAIL')
        sys.exit(1)

    n_anchor = sum(1 for d in draws if d.get('draw_kind') == 'anchor')
    rate = n_anchor / len(draws)
    lo = anchor_prob - ANCHOR_RATE_TOL
    hi = anchor_prob + ANCHOR_RATE_TOL
    ok = lo <= rate <= hi
    print(f'anchor 採択率: {rate:.4f} ({n_anchor}/{len(draws)}) '
          f'(期待 {anchor_prob}±{ANCHOR_RATE_TOL} = [{lo:.2f}, {hi:.2f}]) -> '
          f"{'OK' if ok else 'NG'}")

    if not ok:
        print('\n機械ゲート判定: 未達 (run停止・実装調査)')
        sys.exit(1)
    print('\n機械ゲート判定: 通過')


def run_arm_k(diag_path: Path, kl_beta: float):
    recs = []
    with diag_path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('event') == 'kl_anchor' and d.get('trainer_step') is not None:
                recs.append(d)

    window = [d for d in recs if 0 <= d['trainer_step'] <= MECH_HI]
    print(f'\n=== Anchor Arm K 機械ゲート (@step{MECH_HI}) ===')
    print(f'kl_anchor レコード総数: {len(recs)}')
    print(f'窓 [0, {MECH_HI}] のバッチ数: {len(window)}')

    if not window:
        print('FATAL: 窓内 kl_anchor レコード 0 件 -> FAIL')
        sys.exit(1)

    beta_values = {d.get('kl_beta') for d in window}
    beta_ok = beta_values == {kl_beta}
    print(f'kl_beta の値集合: {sorted(beta_values)} (期待 {{{kl_beta}}}) -> '
          f"{'OK' if beta_ok else 'NG'}")

    finite_ok = True
    for d in window:
        v = d.get('kl_ref_mean')
        if v is None or not math.isfinite(v) or v <= 0:
            finite_ok = False
            print(f'  NG: trainer_step={d.get("trainer_step")} kl_ref_mean={v!r}')
            break
    if finite_ok:
        means = [d['kl_ref_mean'] for d in window]
        print(f'kl_ref_mean: min={min(means):.6f} max={max(means):.6f} '
              f'mean={sum(means)/len(means):.6f} -> OK (全て有限かつ >0)')

    mech_pass = beta_ok and finite_ok
    print(f'\n機械ゲート判定: {"通過" if mech_pass else "未達 (run停止・実装調査)"}')
    if not mech_pass:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='run dir (arm c) or ppo_diag.jsonl (arm k)')
    parser.add_argument('--arm', choices=['c', 'k'], required=True)
    parser.add_argument('--anchor-prob', type=float, default=0.25)
    parser.add_argument('--kl-beta', type=float, default=0.1)
    args = parser.parse_args()

    if args.arm == 'c':
        run_dir = args.path
        if run_dir.is_file():
            run_dir = run_dir.parent.parent
        run_arm_c(run_dir, args.anchor_prob)
    else:
        diag_path = args.path
        if diag_path.is_dir():
            diag_path = diag_path / 'logs' / 'ppo_diag.jsonl'
        if not diag_path.is_file():
            print(f'FATAL: diag not found: {diag_path}')
            sys.exit(1)
        run_arm_k(diag_path, args.kl_beta)


if __name__ == '__main__':
    main()
