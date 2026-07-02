#!/usr/bin/env python3
"""Summarize action_mass events from ppo_diag.jsonl for P2b lr probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/gamba/mahjong/runs/ppo/smoke_p2b/logs/ppo_diag.jsonl')
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('event') == 'action_mass':
            rows.append(row)

    if not rows:
        print('no action_mass rows')
        return

    print(f'n_steps={len(rows)}')
    print('step\tpi_call\tpi_riichi')
    for i, r in enumerate(rows):
        step = r.get('trainer_step', i)
        pc = r.get('pi_call_given_possible')
        pr = r.get('pi_riichi_given_possible')
        pc_s = f'{pc:.6f}' if pc is not None else 'n/a'
        pr_s = f'{pr:.6f}' if pr is not None else 'n/a'
        print(f'{step}\t{pc_s}\t{pr_s}')

    first = rows[0]
    last = rows[-1]
    print('\n=== trend ===')
    for label, r in [('first', first), ('last', last)]:
        pc = r.get('pi_call_given_possible')
        pr = r.get('pi_riichi_given_possible')
        print(f"{label} step={r.get('trainer_step')}: pi_call={pc} pi_riichi={pr}")


if __name__ == '__main__':
    main()
