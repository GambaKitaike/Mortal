#!/usr/bin/env python3
"""西入（サドンデス）が結果に与える影響の実測。GPU 不要・読み取り専用。

動機
----
libriichi の arena は天鳳ルール準拠で**西入を実装している**
（`libriichi/src/arena/game.rs:79-86`: オーラス終了時に 30000 点以上の者が居なければ
`kyoku` が `length`(=8) を超えて西場へ進み、W4 で打ち切り）。一方
**フリー雀荘ルールでは西入を採らないのが一般的**（2026-07-30 Gamba 指摘、レンズ4 の副産物）。

ルール変更は環境そのものの変更＝新しい実験レジームであり、既存 run の判定値との比較可能性を
壊す。着手前に「どれだけ影響するか」を実測して裁定材料にするのが本スクリプト。

何をするか
----------
既存の 1v3 eval 牌譜について、**西1局の開始直前でログを打ち切った反実仮想**
（= 西入なしルールでの終局）を作り、challenger の3ストリーム
（素点 / 順位点 / チップ）を実際の結果と突き合わせる。

**経済ロジックは複製しない**（禁則）。`analyze_freeparlor_pnl_1v3.py` の
`reconstruct_final_scores` / `rank_points` / `seat_from_filename` と
`chip_from_log.load_kyoku_chip_deltas_from_log` をそのまま呼ぶ。
供託の未収精算・同点タイブレークもそれらの実装に従う。

**測れないこと（正直に明示）**: 西入の有無は**方策の意思決定そのものを変える**
（オーラスで 30000 点に届かせる価値が消える）。本スクリプトは既存牌譜の事後集計なので
その行動変化は測れない。出るのは「**ルールを消したときに動く量の目安**」であって、
学習への影響の推定値ではない。

使い方
------
    PYTHONPATH=mortal python freeparlor/scripts/analyze_west_round_impact.py \
        --logs <game_logs dir> [--label NAME]
"""
from __future__ import annotations

import argparse
import glob
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_freeparlor_pnl_1v3 import (  # noqa: E402
    CHIP_VALUE,
    RETURN_SCORE,
    load_events,
    rank_points,
    reconstruct_final_scores,
    seat_from_filename,
)
from chip_from_log import load_kyoku_chip_deltas_from_log  # noqa: E402


def first_west_index(events: list[dict]) -> tuple[int, int] | None:
    """(events 中の西1局 start_kyoku の位置, その局の 0-origin kyoku 番号) を返す。"""
    kyoku_idx = -1
    for i, ev in enumerate(events):
        if ev.get('type') == 'start_kyoku':
            kyoku_idx += 1
            if ev['bakaze'] == 'W':
                return i, kyoku_idx
    return None


def streams(scores: list[int], seat: int, chips: float) -> tuple[float, float, float, float]:
    sotensu = (scores[seat] - RETURN_SCORE) / 1000.0
    rpts = rank_points(scores)[seat]
    return sotensu, rpts, chips, sotensu + rpts + chips * CHIP_VALUE


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', action='append', required=True, help='game_logs ディレクトリ（複数可）')
    ap.add_argument('--label', default='')
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()

    out = open(args.out, 'w', encoding='utf-8') if args.out else sys.stdout
    try:
        for d in args.logs:
            files = sorted(glob.glob(str(Path(d) / '*.json.gz')))
            if not files:
                print(f'FATAL: no *.json.gz under {d}', file=out)
                return 1
            n = len(files)
            n_west = 0
            n_west_kyoku = 0
            rank_changed = 0
            d_sotensu, d_rank, d_chip, d_comb = [], [], [], []
            for f in files:
                path = Path(f)
                events = load_events(path)
                hit = first_west_index(events)
                if hit is None:
                    continue
                n_west += 1
                ev_i, kyoku_i = hit
                seat = seat_from_filename(path)
                n_kyoku = sum(1 for ev in events if ev.get('type') == 'start_kyoku')
                n_west_kyoku += n_kyoku - kyoku_i

                actual_scores = reconstruct_final_scores(events, path)
                cf_scores = reconstruct_final_scores(events[:ev_i], path)

                chip_per_kyoku = load_kyoku_chip_deltas_from_log(path, seat, n_kyoku)
                actual_chip = float(chip_per_kyoku.sum())
                cf_chip = float(chip_per_kyoku[:kyoku_i].sum())

                a = streams(actual_scores, seat, actual_chip)
                c = streams(cf_scores, seat, cf_chip)
                if a[1] != c[1]:
                    rank_changed += 1
                d_sotensu.append(a[0] - c[0])
                d_rank.append(a[1] - c[1])
                d_chip.append(a[2] - c[2])
                d_comb.append(a[3] - c[3])

            print(f'\n=== {args.label or Path(d).name} ===', file=out)
            print(f'  半荘数 = {n} / **西入した半荘 = {n_west} ({100 * n_west / n:.2f}%)**', file=out)
            if n_west == 0:
                continue
            print(f'  西場の局数 = {n_west_kyoku}（西入1回あたり {n_west_kyoku / n_west:.2f} 局）',
                  file=out)
            print(f'  西入した半荘のうち **challenger の順位点が変わった** = {rank_changed}/{n_west} '
                  f'({100 * rank_changed / n_west:.1f}%)'
                  f' → 全体では {100 * rank_changed / n:.2f}% の半荘', file=out)
            print('', file=out)
            print(f'  {"ストリーム":<12} | {"西入あり − 西入なし":>18} | {"全 n 半荘に均すと":>18}', file=out)
            for name, xs in (('素点', d_sotensu), ('順位点', d_rank),
                             ('チップ(枚)', d_chip), ('合算', d_comb)):
                print(f'  {name:<12} | {st.mean(xs):+18.3f} | {sum(xs) / n:+18.4f}', file=out)
            print('', file=out)
            print('  ※ 西入した半荘のみの平均（左列）と、全半荘に均した寄与（右列）。', file=out)
            print('  ※ 事後集計であり、西入を消すとオーラスの意思決定自体が変わる。'
                  '学習への影響の推定値ではない。', file=out)
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
