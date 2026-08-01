#!/usr/bin/env python3
"""L1 O3 の発進ゲート（l1_o3_ppo_epochs_design.md §4-2）。read-only。

`ppo_epochs = 1` が実際に効いていることを機械的に検査する:

  1. 窓内の全 `ppo_epoch` レコードが `epoch == 1` であること
     （`ppo_epochs = 4` のままなら epoch 2/3/4 のレコードが出るので必ず落ちる）
  2. 窓内の `ppo_epoch` レコード数が distinct な trainer_step 数と一致すること
     （1 step あたり 1 epoch）

判定には一切関与しない。INFO として clip@epoch1 / ratio を出す。

  python freeparlor/scripts/gate_l1_o3_epochs.py --run <RUN_DIR> [--gate-window 1 200]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run dir')
    ap.add_argument('--gate-window', nargs=2, type=int, default=[1, 200],
                    metavar=('LO', 'HI'), help='検査する trainer_step の窓（両端含む）')
    args = ap.parse_args()

    diag = Path(args.run) / 'logs' / 'ppo_diag.jsonl'
    if not diag.exists():
        print(f'FAIL: {diag} が無い', file=sys.stderr)
        return 1

    lo, hi = args.gate_window
    rows = []
    with diag.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('event') != 'ppo_epoch':
                continue
            step = d.get('trainer_step')
            if step is None or not (lo <= step <= hi):
                continue
            rows.append(d)

    print(f'=== {Path(args.run).name} ===')
    print(f'ゲート窓 trainer_step [{lo}, {hi}]: ppo_epoch レコード n={len(rows)}')
    if not rows:
        print('FAIL: 窓内にレコードが無い（step 到達前か diag 未書き出し）', file=sys.stderr)
        return 1

    bad = [d for d in rows if d.get('epoch') != 1]
    steps = {d['trainer_step'] for d in rows}
    ok = True

    if bad:
        seen = sorted({d.get('epoch') for d in bad})
        print(f'FAIL: epoch != 1 のレコードが {len(bad)} 件（epoch={seen}）'
              f' — ppo_epochs=1 が効いていない', file=sys.stderr)
        ok = False
    else:
        print(f'PASS: 全 {len(rows)} 件が epoch == 1')

    if len(rows) != len(steps):
        print(f'FAIL: レコード数 {len(rows)} != distinct step 数 {len(steps)}'
              f' — 1 step あたり 1 epoch になっていない', file=sys.stderr)
        ok = False
    else:
        print(f'PASS: 1 step あたり 1 レコード（distinct step={len(steps)}）')

    clip = [d['clip_fraction'] for d in rows if 'clip_fraction' in d]
    ratio_m = [d['ratio_mean'] for d in rows if 'ratio_mean' in d]
    ratio_s = [d['ratio_std'] for d in rows if 'ratio_std' in d]
    if clip:
        print(f'INFO（合否条件ではない）: clip@epoch1 mean={st.mean(clip):.4f} '
              f'median={st.median(clip):.4f}')
    if ratio_m and ratio_s:
        print(f'INFO: ratio_mean={st.mean(ratio_m):.4f} ratio_std={st.mean(ratio_s):.4f}')

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
