#!/usr/bin/env python3
"""Summarize P3 Stage1 ppo_diag.jsonl for results md."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_action_mass(rows: list[dict], step: int) -> dict | None:
    best = None
    for r in rows:
        if r.get('event') != 'action_mass':
            continue
        s = int(r.get('trainer_step', -1))
        if s <= step and (best is None or s >= best.get('trainer_step', -1)):
            if s == step or best is None or s > best.get('trainer_step', -1):
                if s == step:
                    return r
                best = r
    return best if best and best.get('trainer_step') == step else best


def window_mean(rows: list[dict], event: str, lo: int, hi: int, key: str) -> tuple[float | None, int]:
    vals = []
    for r in rows:
        if r.get('event') != event:
            continue
        s = int(r.get('trainer_step', -1))
        if lo <= s < hi:
            v = r.get(key)
            if v is not None:
                vals.append(float(v))
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def trend_up(values: list[float]) -> bool:
    if len(values) < 4:
        return False
    n = len(values)
    mid = n // 2
    early = sum(values[:mid]) / mid
    late = sum(values[mid:]) / (n - mid)
    return late > early * 1.05


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        '/home/gamba/mahjong/runs/ppo/stage1_20260703_064427',
    )
    diag = run_dir / 'logs' / 'ppo_diag.jsonl'
    rows = load_rows(diag)

    checkpoints = [0, 4000, 8000, 12000, 16000]
    print('=== action_mass time series ===')
    print('| step | pi_call | pi_call_aka | pi_call_no_aka | ratio | n_aka | n_no_aka |')
    print('|---:|---:|---:|---:|---:|---:|---:|')
    for step in checkpoints:
        r = pick_action_mass(rows, step)
        if not r:
            print(f'| {step} | — | — | — | — | — | — |')
            continue
        print(
            f"| {r.get('trainer_step', step)} "
            f"| {r.get('pi_call_given_possible', 0):.4f} "
            f"| {r.get('pi_call_given_possible_aka_held', 0) or 0:.4f} "
            f"| {r.get('pi_call_given_possible_no_aka', 0) or 0:.4f} "
            f"| {r.get('pi_call_aka_over_no_aka', 0) or 0:.4f} "
            f"| {r.get('n_call_possible_aka_held', 0)} "
            f"| {r.get('n_call_possible_no_aka', 0)} |"
        )

    init_aka, _ = window_mean(rows, 'action_mass', 0, 1, 'pi_call_given_possible_aka_held')
    if init_aka is None:
        r0 = pick_action_mass(rows, 0)
        init_aka = r0.get('pi_call_given_possible_aka_held') if r0 else None

    j_lo, j_hi = 8000, 16000
    win_aka, n_win = window_mean(rows, 'action_mass', j_lo, j_hi, 'pi_call_given_possible_aka_held')
    win_series = [
        float(r['pi_call_given_possible_aka_held'])
        for r in rows
        if r.get('event') == 'action_mass'
        and j_lo <= int(r.get('trainer_step', -1)) < j_hi
        and r.get('pi_call_given_possible_aka_held') is not None
    ]

    print('\n=== judgment window 8000-16000 ===')
    print(f'init pi_call_aka: {init_aka}')
    print(f'window mean pi_call_aka: {win_aka} (n={n_win})')
    if init_aka and win_aka is not None:
        ratio = win_aka / init_aka if init_aka > 0 else float('inf')
        print(f'window/init ratio: {ratio:.4f}')
        print(f'below 2x init: {ratio < 2.0}')
        print(f'uptrend: {trend_up(win_series)}')

    print('\n=== grp_calibration ===')
    for r in rows:
        if r.get('event') == 'grp_calibration':
            print(
                f"step {r.get('trainer_step')}: "
                f"mean_abs_rank_err={r.get('mean_abs_rank_err'):.4f} "
                f"n={r.get('n_hanchan')}"
            )

    print('\n=== advantage_decomp window 8000-16000 (raw call_taken vs declined mean) ===')
    ct, cd = [], []
    for r in rows:
        if r.get('event') != 'advantage_decomp':
            continue
        s = int(r.get('trainer_step', -1))
        if not (j_lo <= s < j_hi):
            continue
        rt = r.get('raw', {}).get('call_taken')
        rd = r.get('raw', {}).get('call_declined')
        if rt and rt.get('mean') is not None:
            ct.append(rt['mean'])
        if rd and rd.get('mean') is not None:
            cd.append(rd['mean'])
    if ct and cd:
        print(f'call_taken raw mean: {sum(ct)/len(ct):.4f}')
        print(f'call_declined raw mean: {sum(cd)/len(cd):.4f}')


if __name__ == '__main__':
    main()
