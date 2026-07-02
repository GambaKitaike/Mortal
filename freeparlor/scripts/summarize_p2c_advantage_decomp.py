#!/usr/bin/env python3
"""Summarize P2c advantage_decomp / kyoku_reward_decomp from ppo_diag.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _weighted_mean(rows: list[dict | None], key: str) -> tuple[float | None, int]:
    num = 0.0
    den = 0
    for row in rows:
        if not row:
            continue
        n = row.get('n', 0)
        if not n:
            continue
        num += row[key] * n
        den += n
    if den == 0:
        return None, 0
    return num / den, den


def _aggregate_adv(rows: list[dict], scale: str, category: str) -> dict | None:
    num = 0.0
    var_num = 0.0
    den = 0
    for row in rows:
        block = row.get(scale, {}).get(category)
        if not block:
            continue
        n = block['n']
        mean = block['mean']
        std = block['std']
        num += mean * n
        var_num += (std ** 2 + mean ** 2) * n
        den += n
    if den == 0:
        return None
    mean = num / den
    var = var_num / den - mean ** 2
    return {'mean': mean, 'std': max(var, 0.0) ** 0.5, 'n': den}


def main() -> int:
    path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else '/home/gamba/mahjong/runs/ppo/smoke_p2c/logs/ppo_diag.jsonl'
    )
    adv_rows = []
    kyoku_rows = []
    action_mass = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            ev = row.get('event')
            if ev == 'advantage_decomp':
                adv_rows.append(row)
            elif ev == 'kyoku_reward_decomp':
                kyoku_rows.append(row)
            elif ev == 'action_mass':
                action_mass.append(row)

    print(f'path={path}')
    print(f'advantage_decomp rows={len(adv_rows)} kyoku_reward_decomp rows={len(kyoku_rows)}')

    for scale in ('raw', 'norm'):
        print(f'\n=== advantage ({scale}) ===')
        for cat in ('call_taken', 'call_declined', 'riichi_taken', 'riichi_declined'):
            agg = _aggregate_adv(adv_rows, scale, cat)
            if not agg:
                print(f'{cat}: n=0')
                continue
            print(
                f'{cat}: mean={agg["mean"]:.4f} std={agg["std"]:.4f} n={agg["n"]}'
            )

    print('\n=== kyoku reward by riichi ===')
    for label in ('yes', 'no'):
        s_rows = [r['by_riichi'].get(label) for r in kyoku_rows]
        s_mean, s_n = _weighted_mean(s_rows, 'sotensu_mean')
        g_mean, _ = _weighted_mean(s_rows, 'grp_mean')
        c_mean, _ = _weighted_mean(s_rows, 'chip_mean')
        print(
            f'riichi_{label}: n={s_n} '
            f'sotensu={s_mean if s_mean is not None else float("nan"):.4f} '
            f'grp={g_mean if g_mean is not None else float("nan"):.4f} '
            f'chip={c_mean if c_mean is not None else float("nan"):.4f}'
        )

    print('\n=== kyoku reward by call ===')
    for label in ('yes', 'no'):
        s_rows = [r['by_call'].get(label) for r in kyoku_rows]
        s_mean, s_n = _weighted_mean(s_rows, 'sotensu_mean')
        g_mean, _ = _weighted_mean(s_rows, 'grp_mean')
        c_mean, _ = _weighted_mean(s_rows, 'chip_mean')
        print(
            f'call_{label}: n={s_n} '
            f'sotensu={s_mean if s_mean is not None else float("nan"):.4f} '
            f'grp={g_mean if g_mean is not None else float("nan"):.4f} '
            f'chip={c_mean if c_mean is not None else float("nan"):.4f}'
        )

    if action_mass:
        first = action_mass[0]
        last = action_mass[-1]
        print('\n=== action_mass endpoints ===')
        print(
            f'step {first.get("trainer_step", "?")}: '
            f'pi_call={first.get("pi_call_given_possible")} n={first.get("n_call_possible")} '
            f'pi_riichi={first.get("pi_riichi_given_possible")} n={first.get("n_riichi_possible")}'
        )
        print(
            f'step {last.get("trainer_step", "?")}: '
            f'pi_call={last.get("pi_call_given_possible")} n={last.get("n_call_possible")} '
            f'pi_riichi={last.get("pi_riichi_given_possible")} n={last.get("n_riichi_possible")}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
