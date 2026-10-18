#!/usr/bin/env python3
"""Summarize PPO P2 diagnostic jsonl for ppo_p2_diag.md."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/gamba/mahjong/runs/ppo/smoke_p2/logs/ppo_diag.jsonl')
    if not path.is_file():
        print('missing', path)
        return

    epoch_rows = []
    lags = []
    by_step_epoch: dict[int, dict[int, dict]] = defaultdict(dict)

    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('event') == 'batch_lag':
            lags.append(int(row['lag']))
        elif row.get('event') == 'ppo_epoch':
            epoch_rows.append(row)
            by_step_epoch[int(row['trainer_step'])][int(row['epoch'])] = row

    print('=== param_version lag ===')
    if lags:
        print(f'n={len(lags)} min={min(lags)} max={max(lags)} mean={statistics.mean(lags):.2f} median={statistics.median(lags):.1f}')
        hist = defaultdict(int)
        for lag in lags:
            hist[lag] += 1
        for lag in sorted(hist):
            print(f'  lag={lag}: {hist[lag]}')
    else:
        print('no batch_lag rows')

    print('\n=== epoch clip (first/last trainer steps) ===')
    steps = sorted(by_step_epoch)
    for label, step in [('first', steps[0] if steps else None), ('last', steps[-1] if steps else None)]:
        if step is None:
            continue
        print(f'-- step {step} ({label}) --')
        for epoch in sorted(by_step_epoch[step]):
            r = by_step_epoch[step][epoch]
            print(
                f"  epoch {epoch}: clip={r['clip_fraction']:.4f} "
                f"ratio={r['ratio_mean']:.4f}±{r['ratio_std']:.4f} lag={r.get('param_lag')}"
            )

    if epoch_rows:
        e1 = [r['clip_fraction'] for r in epoch_rows if r['epoch'] == 1]
        e_last = [r['clip_fraction'] for r in epoch_rows if r['epoch'] == max(r['epoch'] for r in epoch_rows)]
        print('\n=== aggregate ===')
        print(f"epoch1 clip mean={statistics.mean(e1):.4f} (n={len(e1)})")
        print(f"epoch{max(r['epoch'] for r in epoch_rows)} clip mean={statistics.mean(e_last):.4f} (n={len(e_last)})")


if __name__ == '__main__':
    main()
