#!/usr/bin/env python3
"""Collect PPO P2 smoke metrics from logs and TensorBoard for ppo_p2_smoke.md."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_trainer_log(log_path: Path) -> dict:
    steps = []
    for line in log_path.read_text(errors='replace').splitlines():
        m = re.search(
            r'ppo step (\d+): total=([\-\d.]+) pi=([\-\d.]+) vf=([\-\d.]+) '
            r'H=([\-\d.]+) clip=([\-\d.]+) ev=([\-\d.]+)',
            line,
        )
        if m:
            steps.append({
                'step': int(m.group(1)),
                'total': float(m.group(2)),
                'policy_loss': float(m.group(3)),
                'value_loss': float(m.group(4)),
                'entropy': float(m.group(5)),
                'clip_fraction': float(m.group(6)),
                'explained_variance': float(m.group(7)),
            })
    return {'steps': steps}


def tb_scalars(tb_dir: Path, tags: list[str]) -> dict:
    acc = EventAccumulator(str(tb_dir))
    acc.Reload()
    out = {}
    for tag in tags:
        if tag not in acc.Tags().get('scalars', []):
            continue
        events = acc.Scalars(tag)
        out[tag] = [(e.step, e.value) for e in events]
    return out


def count_in_logs(pattern: str, log_paths: list[Path]) -> int:
    total = 0
    rx = re.compile(pattern)
    for p in log_paths:
        if not p.exists():
            continue
        for line in p.read_text(errors='replace').splitlines():
            if rx.search(line):
                total += 1
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default='/home/gamba/mahjong/runs/ppo/smoke_p2')
    args = ap.parse_args()
    run = Path(args.run_dir)
    logs = run / 'logs'

    trainer = parse_trainer_log(logs / 'trainer.log')
    client_logs = sorted(logs.glob('client*.log'))

    mismatch = count_in_logs('trajectory step count mismatch', client_logs)
    chip_err = count_in_logs('online chip resolution failed', client_logs)
    nan_err = count_in_logs(r'non-finite|FloatingPointError', [logs / 'trainer.log'])

    tags = [
        'loss/total', 'loss/policy_loss', 'loss/value_loss', 'loss/entropy',
        'ppo/clip_fraction', 'ppo/explained_variance',
        'test_play/avg_ranking', 'test_play/behavior/houjuu',
    ]
    tb = tb_scalars(run / 'tb', tags)

    print('=== PPO P2 smoke metrics ===')
    print(f'steps_logged: {len(trainer["steps"])}')
    if trainer['steps']:
        first = trainer['steps'][0]
        last = trainer['steps'][-1]
        print(f'first_step: {first}')
        print(f'last_step: {last}')
        ent_vals = [s['entropy'] for s in trainer['steps']]
        print(f'entropy_first: {ent_vals[0]:.4f} entropy_last: {ent_vals[-1]:.4f}')
        clip_vals = [s['clip_fraction'] for s in trainer['steps']]
        print(f'clip_mean: {sum(clip_vals)/len(clip_vals):.4f} clip_max: {max(clip_vals):.4f}')
    print(f'mismatch_count: {mismatch}')
    print(f'chip_errors: {chip_err}')
    print(f'nan_errors: {nan_err}')
    for tag, series in tb.items():
        if series:
            print(f'tb {tag}: last=({series[-1][0]}, {series[-1][1]:.6f})')


if __name__ == '__main__':
    main()
