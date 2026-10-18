#!/usr/bin/env python3
"""役牌の対子・暗刻を自分から手放す率 — レンズ4 所見の指標化（read-only・GPU 不要）。

動機
----
2026-08-02 のレンズ4（L1 O3 の 1v3、`10000_8192_a`）で Gamba が同じ癖を **3 局**で指摘した:

  - 東3局 4巡目打西 — 「**暗刻の役牌を切る癖は変わってない。自風の概念を理解して
    いないのか**」（東3局の起家は西家 = 西が自風）
  - 東4局 4/5巡目打發打發（対子落とし）+ 8巡目に3枚目も打 —
    「**發鳴かないとこの手は間に合いません。対子落としを安易に選択しないでください**」
  - 東3局 1巡目打南・2巡目打東 — 「**役牌より価値のない牌は切ること**」（切り順の問題）

既存の指標では捕まらない:
  - `analyze_ankou_pon.py` は「暗刻**から**ポン」= 鳴いた側の異常で、**打った側**は見ない
  - `analyze_call_quality.py` は鳴き機会が発生した決定点しか見ないので、
    「鳴き機会が来る前に自分から役牌を捨てて機会を消した」ケースを取りこぼす

役牌の対子は「1 鳴きで確定役 + 打点」という自己対戦経済で最も安い価値源であり、
早巡でこれを手放すのは（降りでない限り）ほぼ純損。所見のまま置かず機械検出できる
指標にしたのが本スクリプト（`qualitative_review_protocol.md` §3(4)）。

測る量
------
**早巡・非降り**の打牌決定点に限定する（降り時の役牌落としは正当なので分母から外す）:

  - 自分が立直していない
  - **他家に立直者がいない**（降りの局面を除外）
  - `turn <= EARLY_TURN`（既定 6。所見の3件はすべて 1〜5 巡目）
  - 手牌（門前）に役牌 = 三元牌 ∪ 場風 ∪ 自風 を **2 枚以上**持っている

  分母: 上を満たす決定点（対子以上を持っている / 暗刻以上を持っている で別集計）
  分子: そのうち、実際に**その役牌を打った**決定点

「打った」の判定は決定点の**次のイベント**が自席の `dahai` かで行う
（`analyze_ankou_pon.py` と同じ規約）。手牌復元は共通部品 `hand_replay.replay`、
合法手は `PlayerState.last_cans` = libriichi 自身の判定を使う（自前で書かない）。

使い方
------
    PYTHONPATH=mortal python freeparlor/scripts/analyze_yakuhai_retention.py \\
        --init-logs <dir> --ckpt-logs <dir> --label NAME [--early-turn 6]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_freeparlor_pnl_1v3 import seat_from_filename  # noqa: E402
from hand_replay import TRANSPARENT_EVENTS, replay, tile_id  # noqa: E402

SANGEN = (31, 32, 33)   # 白 發 中


def analyze_log(path: Path, early_turn: int) -> dict:
    seat = seat_from_filename(path)
    r = {
        'n_opp_toitsu': 0, 'n_drop_toitsu': 0,
        'n_opp_ankou': 0, 'n_drop_ankou': 0,
        'n_decisions': 0,
    }
    pending = None
    for ev, dec in replay(path, seat):
        et = ev.get('type')
        if et in TRANSPARENT_EVENTS:
            continue

        if pending is not None:
            held_toitsu, held_ankou = pending
            if et == 'dahai' and ev.get('actor') == seat and ev.get('pai') is not None:
                d = tile_id(ev['pai'])
                if d in held_ankou:
                    r['n_drop_ankou'] += 1
                elif d in held_toitsu:
                    r['n_drop_toitsu'] += 1
            pending = None

        if dec is None:
            continue
        if dec.self_riichi or dec.others_riichi:
            continue
        if dec.turn > early_turn:
            continue
        # 鳴き機会の決定点（他家の打牌に対する応答）は打牌決定点ではないので除く
        if dec.target_tile is not None:
            continue
        r['n_decisions'] += 1

        yakuhai = set(SANGEN) | {dec.bakaze, dec.jikaze}
        held_toitsu = {y for y in yakuhai if dec.tehai[y] == 2}
        held_ankou = {y for y in yakuhai if dec.tehai[y] >= 3}
        if held_toitsu:
            r['n_opp_toitsu'] += 1
        if held_ankou:
            r['n_opp_ankou'] += 1
        if held_toitsu or held_ankou:
            pending = (held_toitsu, held_ankou)
    return r


def ratio_and_se(rows, num, den):
    tot_d = sum(x[den] for x in rows)
    if tot_d == 0:
        return float('nan'), float('nan'), 0
    tot_n = sum(x[num] for x in rows)
    p = tot_n / tot_d
    m = len(rows)
    if m < 2:
        return p, float('nan'), tot_d
    resid = [x[num] - p * x[den] for x in rows]
    s2 = sum(v * v for v in resid) / (m - 1)
    se = math.sqrt(m * s2) / tot_d
    return p, se, tot_d


def run(logs_dir: str, early_turn: int) -> list[dict]:
    files = sorted(Path(logs_dir).glob('*.json.gz'))
    if not files:
        raise SystemExit(f'FATAL: no *.json.gz under {logs_dir}')
    return [analyze_log(p, early_turn) for p in files]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--init-logs', required=True)
    ap.add_argument('--ckpt-logs', required=True)
    ap.add_argument('--label', default='ckpt')
    ap.add_argument('--early-turn', type=int, default=6)
    args = ap.parse_args()

    init_rows = run(args.init_logs, args.early_turn)
    ckpt_rows = run(args.ckpt_logs, args.early_turn)
    for label, rows in (('init', init_rows), (args.label, ckpt_rows)):
        print(f'[{label}] {len(rows)} hanchan / 早巡・非降りの打牌決定点 '
              f'{sum(x["n_decisions"] for x in rows):,} / '
              f'役牌対子あり {sum(x["n_opp_toitsu"] for x in rows):,} / '
              f'役牌暗刻あり {sum(x["n_opp_ankou"] for x in rows):,}')
    print()
    print(f'{"metric":<30}{"init":>12}{args.label:>14}{"diff":>10}{"SE":>8}{"z":>8}')
    for name, num, den in (
        ('役牌対子を落とした率', 'n_drop_toitsu', 'n_opp_toitsu'),
        ('役牌暗刻を落とした率', 'n_drop_ankou', 'n_opp_ankou'),
    ):
        p0, se0, _ = ratio_and_se(init_rows, num, den)
        p1, se1, _ = ratio_and_se(ckpt_rows, num, den)
        diff = (p1 - p0) * 100
        se = math.sqrt(se0 * se0 + se1 * se1) * 100
        z = diff / se if se and not math.isnan(se) and se > 0 else float('nan')
        print(f'{name:<30}{p0 * 100:>11.2f}%{p1 * 100:>13.2f}%{diff:>+10.2f}{se:>8.2f}{z:>+8.2f}')
    print()
    print(f'注: 分母は「その決定点で役牌の対子(暗刻)を持っていた」件数。turn <= {args.early_turn} かつ')
    print('    自他とも立直なしに限定してあるので、降りのための役牌落としは含まない。')
    print('    ゼロであるべき指標ではない（ホンイツ移行・字牌過多の整理など正当な例はある）。')
    print('    init との差分で読むこと。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
