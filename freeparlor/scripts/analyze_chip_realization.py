#!/usr/bin/env python3
"""チップの実現率 — 「稼いだ量」ではなく「どう稼いだか」を分解する（read-only・GPU 不要）。

動機
----
既存の `analyze_freeparlor_pnl_1v3.py` はチップの**総量**（枚/半荘）を出すが、
その内訳を出さない。総量が同じでも
「和了回数は少ないが1回あたりが厚い」と「薄く広く」では方策の性質が違う。
また**被チップ**（相手の和了で払う側）は総量に埋もれて見えない。

チップ規則（`freeparlor/scripts/preprocess_chips.py:24`）:

    chip_base = num_aka + num_ura + ippatsu + 5 * (yakuman >= 1)

ツモなら和了者 +3×base / 他家 −base、ロンなら和了者 +base / 放銃者 −base。
本スクリプトは牌譜の `hora` イベントに記録済みの `meta.chip_delta`（4人分）を
そのまま読む（規則の再実装はしない）。

測る量（すべて challenger 視点）
------------------------------
  - チップ/半荘・チップ/局（既存指標との突き合わせ用）
  - **チップ獲得率** = チップを伴った和了 / 全和了
  - **和了1回あたりの平均チップ**（獲得側のみ）
  - **被チップ率** = チップを払った局 / 全局
  - **払い1回あたりの平均チップ**
  - 収支の分解: 獲得合計 − 支払合計

使い方
------
    PYTHONPATH=mortal python freeparlor/scripts/analyze_chip_realization.py \\
        --logs <game_logs dir> [--logs <dir> ...] [--label NAME ...]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_freeparlor_pnl_1v3 import seat_from_filename  # noqa: E402


def analyze_log(path: Path) -> dict:
    seat = seat_from_filename(path)
    r = {
        'n_kyoku': 0, 'n_hanchan': 1,
        'n_agari': 0, 'n_agari_with_chip': 0, 'chip_gained': 0,
        'n_paid': 0, 'chip_paid': 0,
    }
    for line in gzip.open(path, 'rt', encoding='utf-8'):
        ev = json.loads(line)
        t = ev.get('type')
        if t == 'start_kyoku':
            r['n_kyoku'] += 1
        elif t == 'hora':
            delta = (ev.get('meta') or {}).get('chip_delta')
            if ev.get('actor') == seat:
                r['n_agari'] += 1
                if delta and delta[seat] > 0:
                    r['n_agari_with_chip'] += 1
                    r['chip_gained'] += delta[seat]
            elif delta and delta[seat] < 0:
                r['n_paid'] += 1
                r['chip_paid'] += -delta[seat]
    return r


def ratio_and_se(rows, num, den):
    tot_d = sum(x[den] for x in rows)
    if tot_d == 0:
        return float('nan'), float('nan')
    tot_n = sum(x[num] for x in rows)
    p = tot_n / tot_d
    m = len(rows)
    if m < 2:
        return p, float('nan')
    resid = [x[num] - p * x[den] for x in rows]
    s2 = sum(v * v for v in resid) / (m - 1)
    return p, math.sqrt(m * s2) / tot_d


def report(label: str, rows: list[dict]) -> None:
    n_h = len(rows)
    n_k = sum(x['n_kyoku'] for x in rows)
    gained = sum(x['chip_gained'] for x in rows)
    paid = sum(x['chip_paid'] for x in rows)
    n_ag = sum(x['n_agari'] for x in rows)
    n_ag_chip = sum(x['n_agari_with_chip'] for x in rows)
    n_paid = sum(x['n_paid'] for x in rows)

    p_get, se_get = ratio_and_se(rows, 'n_agari_with_chip', 'n_agari')
    p_pay, se_pay = ratio_and_se(rows, 'n_paid', 'n_kyoku')

    print(f'[{label}] {n_h:,} 半荘 / {n_k:,} 局 / 和了 {n_ag:,}')
    print(f'  チップ 収支            {(gained - paid) / n_h:+.4f} 枚/半荘'
          f'  （獲得 {gained / n_h:+.4f} − 支払 {paid / n_h:.4f}）')
    print(f'  **チップ獲得率**       {p_get * 100:6.2f}% ±{se_get * 100:.2f}'
          f'   （チップを伴った和了 {n_ag_chip:,} / 全和了 {n_ag:,}）')
    print(f'  和了1回あたりチップ     {gained / max(n_ag_chip, 1):6.3f} 枚（獲得した和了のみ）'
          f' / {gained / max(n_ag, 1):.3f} 枚（全和了で均す）')
    print(f'  **被チップ率**         {p_pay * 100:6.2f}% ±{se_pay * 100:.2f}'
          f'   （払った局 {n_paid:,} / 全局 {n_k:,}）')
    print(f'  払い1回あたりチップ     {paid / max(n_paid, 1):6.3f} 枚')
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', action='append', required=True)
    ap.add_argument('--label', action='append', default=None)
    args = ap.parse_args()

    labels = args.label or [Path(d).name for d in args.logs]
    if len(labels) != len(args.logs):
        raise SystemExit('FATAL: --label の数が --logs と一致しない')

    for label, d in zip(labels, args.logs):
        files = sorted(Path(d).glob('*.json.gz'))
        if not files:
            raise SystemExit(f'FATAL: no *.json.gz under {d}')
        report(label, [analyze_log(p) for p in files])

    print('注: チップ規則は preprocess_chips.py:24 が正'
          '（chip_base = num_aka + num_ura + ippatsu + 5×yakuman）。')
    print('    本スクリプトは牌譜に記録済みの meta.chip_delta を読むだけで規則を再実装しない。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
