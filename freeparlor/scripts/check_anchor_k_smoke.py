#!/usr/bin/env python3
"""anchor Arm K 配管スモークの合否判定（発進ゲートではなく preflight 検査）。

400-step スモークの diag を読み、KL アンカーの配管が実データ経路で健全かを検査する。
**挙動の結論は出さない**（`CLAUDE.md`「400 step 級スモークで挙動の結論を出さない」）。

合格条件:
  1. kl_anchor イベントが 1 件以上ある
  2. 全レコードで kl_beta == 期待値（既定 0.1）
  3. 全レコードで kl_ref_mean が有限（NaN/inf なし）かつ >= 0
  4. trainer_step 0 の kl_ref_mean が ~0（|v| <= 1e-6。ref = step0 方策の陽性対照。
     ここが有意に大きければ ref checkpoint の取り違えを疑う）
  5. kl_ref_mean が最後には > 0（方策が ref から動き出している = 項が生きている）
  6. trainer ログに非有限勾配 / NaN の痕跡がない

使い方: python3 check_anchor_k_smoke.py <run_dir> [--kl-beta 0.1]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--kl-beta', type=float, default=0.1)
    args = ap.parse_args()

    diag = args.run_dir / 'logs' / 'ppo_diag.jsonl'
    if not diag.is_file():
        print(f'FATAL: diag not found: {diag}')
        return 1

    recs = []
    for line in diag.open(encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('event') == 'kl_anchor':
            recs.append(d)

    print(f'=== anchor Arm K 配管スモーク検査: {args.run_dir.name} ===')
    print(f'kl_anchor レコード数: {len(recs)}')
    ok = True

    if not recs:
        print('[1] FATAL: kl_anchor イベントが 0 件 -> 配管が繋がっていない')
        return 1
    print('[1] OK: kl_anchor イベントあり')

    betas = {d.get('kl_beta') for d in recs}
    if betas != {args.kl_beta}:
        print(f'[2] NG: kl_beta の値集合 {sorted(betas)} != {{{args.kl_beta}}}')
        ok = False
    else:
        print(f'[2] OK: 全レコードで kl_beta={args.kl_beta}')

    bad = [d for d in recs
           if d.get('kl_ref_mean') is None
           or not math.isfinite(d['kl_ref_mean'])
           or d['kl_ref_mean'] < 0]
    if bad:
        print(f'[3] NG: 非有限/負の kl_ref_mean が {len(bad)} 件 '
              f'(先頭 step={bad[0].get("trainer_step")} v={bad[0].get("kl_ref_mean")!r})')
        ok = False
    else:
        vals = [d['kl_ref_mean'] for d in recs]
        print(f'[3] OK: kl_ref_mean 全て有限かつ>=0 (min={min(vals):.6g} max={max(vals):.6g})')

    step0 = [d for d in recs if d.get('trainer_step') == 0]
    if step0:
        v0 = step0[0]['kl_ref_mean']
        if abs(v0) <= 1e-6:
            print(f'[4] OK: step0 kl_ref_mean={v0:.3g} ~ 0 (ref = step0 方策の陽性対照)')
        else:
            print(f'[4] NG: step0 kl_ref_mean={v0:.6g} が 0 から乖離 -> ref checkpoint 取り違えの疑い')
            ok = False
    else:
        print('[4] SKIP: trainer_step=0 のレコードなし (resume 起動なら正常)')

    last = max(recs, key=lambda d: d.get('trainer_step', 0))
    if last['kl_ref_mean'] > 0:
        print(f'[5] OK: 最終 step={last.get("trainer_step")} kl_ref_mean={last["kl_ref_mean"]:.6g} > 0 '
              '(方策が ref から動いている)')
    else:
        print(f'[5] NG: 最終 kl_ref_mean={last["kl_ref_mean"]!r} <= 0 -> KL 項が死んでいる疑い')
        ok = False

    trainer_log = args.run_dir / 'logs' / 'trainer.log'
    if trainer_log.is_file():
        text = trainer_log.read_text(encoding='utf-8', errors='replace').lower()
        hits = [k for k in ('non-finite', 'floatingpointerror', 'nan detected') if k in text]
        if hits:
            print(f'[6] NG: trainer ログに {hits} の痕跡')
            ok = False
        else:
            print('[6] OK: trainer ログに非有限/NaN の痕跡なし')
    else:
        print('[6] SKIP: trainer.log なし')

    print()
    print(f'配管スモーク: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
