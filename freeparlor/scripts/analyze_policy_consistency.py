#!/usr/bin/env python3
"""方策の一貫性の指標化（read-only）— 起票: `anchor_arm_k_result.md` §6-1 /
`qualitative_review_anchor_k_20260728.md` §4-1。

## 動機

Gamba のレビューが2つの arm で同じ型の所見を出している:

  - 「東4局 4巡目と6巡目で**同じ牌姿なのに違う選択**」（ArmK §2）
  - 「6m 切ってるのに 4m は残す。ちぐはぐ」（ArmK §5b）
  - Arm C の「降りを始めて中断する」（`analyze_genbutsu_discipline.py` の
    降りの中断率 49.29%）も**方針がぶれる**という同族の症状の可能性がある

いずれも「一つ一つの選択は説明できるが、選択の集合として筋が通っていない」という
指摘であり、既存のどの集計指標にも対応物が無い。

## 測る量

### 主指標: 孤立牌の切り順の逆転率（局内・高カバレッジ）

同じ局の中で、孤立した数牌を切る順序が牌理の価値順と逆になっている割合。

孤立牌 = 手牌に1枚だけあり、同じ色で ±2 以内に他の牌が無い牌（＝ターツにも
面子にも育っていない牌）。孤立牌の価値は中央ほど高いので、スコアを
`1 → 1/9, 2 → 2/8, 3 → 3/7, 4 → 4/6, 5 → 5` と置く。

  巡 i で孤立牌 X を切り、そのとき手牌にあった孤立牌 Y を巡 j > i で切った
  → **正しい順序は score(X) ≤ score(Y)**（価値の低い牌から切る）。
  score(X) > score(Y) なら**逆転**

これは `qualitative_review_anchor_k_20260728.md` §5c で役牌に対して行った
「初打巡目の順序が理論と逆」という測定を、数牌へ一般化したものである
（§5c は平均巡目、本指標はペア単位）。**字牌は対象外** — 役としての価値が
純粋な牌効率の順序とは別軸のため（字牌は §5c の測定が正）。

### 副指標: 同型牌姿での選択のばらつき（低カバレッジ・仮定ゼロ）

決定点 = 自分のツモ番の打牌で `--max-turn`（既定 3）以下の巡目。
**序盤ほど盤面の情報が少なく、選択の違いを「文脈依存」で正当化しにくい**。

  - **key = (門前手牌34カウント, 副露数)** — Gamba の「4巡目と6巡目で同じ牌姿」に対応
  - **key = (門前手牌34カウント, 副露数, 巡目)** — さらに巡目も揃えた厳しい版

出現が2回以上あるキーについて、そのキーの**最頻の打牌**と違う選択をした
出現の割合を「不一致率」とする。14枚手の完全一致は稀なので**カバレッジは数%**に
とどまる（対象決定数を必ず併記する）。主指標はこの弱さを埋めるために置いた。

## 限界（**これを外すと指標を誤読する**）

**不一致 = 誤り、ではない。** 方策は手牌だけでなく盤面全体（河・ドラ・点況・
他家の挙動）を見ているので、同じ手牌で違う打牌を選ぶこと自体は正当な文脈依存で
ありうる。argmax eval は決定論的なので、不一致は必ず「観測の他の部分の違い」に
由来する。

したがって本指標の**絶対値には意味がない**。意味があるのは init との差分で、
同じ局面クラスの母集団に対して不一致率が有意に上がっていれば、
**方策が手牌以外の文脈に対してより敏感（＝ぶれやすく）なった**と言える。
序盤に絞るのはこの解釈を成立しやすくするための設計である。

なお最頻打牌はリーグ全体から決まるため、半荘クラスタ SE は厳密には
クラスタ間の独立性をわずかに破る（800 半荘では影響は小さいと見なす）。
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / 'mortal'))

from analyze_freeparlor_pnl_1v3 import seat_from_filename  # noqa: E402
from hand_replay import TRANSPARENT_EVENTS, replay, tile_id  # noqa: E402


def isolated_score(hand: tuple[int, ...], t: int) -> int | None:
    """数牌 t が孤立牌なら牌理スコア（1..5、中央ほど高い）、そうでなければ None。"""
    if t >= 27 or hand[t] != 1:
        return None
    suit, rank = divmod(t, 9)
    lo, hi = suit * 9, suit * 9 + 8
    for o in range(max(lo, t - 2), min(hi, t + 2) + 1):
        if o != t and hand[o]:
            return None
    return min(rank, 8 - rank) + 1


def collect_log(path: Path, max_turn: int) -> tuple[list[tuple], list[tuple]]:
    """副指標用の (手牌, 副露数, 巡目, 打牌) と、主指標用の逆転ペア集計を返す。

    主指標の集計は (逆転ペア数, 全ペア数) の2要素。
    """
    seat = seat_from_filename(path)
    early = []
    # 局ごとに「自分の打牌決定時の (巡目, 手牌, 切った牌)」を貯める
    per_kyoku: dict[tuple, list[tuple]] = defaultdict(list)
    pending = None
    for ev, dec in replay(path, seat):
        if ev.get('type') in TRANSPARENT_EVENTS:
            continue
        if pending is not None:
            if ev.get('actor') == seat and ev.get('type') == 'dahai':
                kyoku, tehai, n_open, turn = pending
                pai = tile_id(ev['pai'])
                per_kyoku[kyoku].append((turn, tehai, pai))
                if turn <= max_turn:
                    early.append((tehai, n_open, turn, pai))
            pending = None
        if dec is None:
            continue
        cans = dec.cans
        if not cans.can_discard or ev.get('type') != 'tsumo':
            continue
        if dec.self_riichi:
            continue
        pending = (dec.kyoku_id, dec.tehai, dec.n_open, dec.turn)

    n_rev = n_pair = 0
    for rows in per_kyoku.values():
        for i in range(len(rows)):
            turn_i, hand_i, x = rows[i]
            sx = isolated_score(hand_i, x)
            if sx is None:
                continue
            for j in range(i + 1, len(rows)):
                _turn_j, _hand_j, y = rows[j]
                # Y は「巡 i の時点で手牌にあり、かつそのとき孤立していた」ものに限る。
                # 巡 i の後に引いてきた牌を混ぜると順序の議論が成立しない
                sy = isolated_score(hand_i, y)
                if sy is None or y == x:
                    continue
                n_pair += 1
                if sx > sy:
                    n_rev += 1
    return early, [(n_rev, n_pair)]


def disagreement(per_hanchan: list[list[tuple]], with_turn: bool):
    """不一致率と半荘クラスタ SE、カバレッジを返す。"""
    choices: dict[tuple, Counter] = defaultdict(Counter)
    for rows in per_hanchan:
        for tehai, n_open, turn, pai in rows:
            key = (tehai, n_open, turn) if with_turn else (tehai, n_open)
            choices[key][pai] += 1

    modal = {k: c.most_common(1)[0][0] for k, c in choices.items()}
    multi = {k for k, c in choices.items() if sum(c.values()) >= 2}

    num, den = [], []
    for rows in per_hanchan:
        n = d = 0
        for tehai, n_open, turn, pai in rows:
            key = (tehai, n_open, turn) if with_turn else (tehai, n_open)
            if key not in multi:
                continue
            d += 1
            if pai != modal[key]:
                n += 1
        num.append(n)
        den.append(d)

    tot_d = sum(den)
    if tot_d == 0:
        return None, None, 0, 0
    rr = sum(num) / tot_d
    var = sum((a - rr * b) ** 2 for a, b in zip(num, den)) / tot_d**2
    return rr, math.sqrt(var), len(multi), tot_d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--init-logs', required=True)
    ap.add_argument('--ckpt-logs', required=True)
    ap.add_argument('--label', default='ckpt')
    ap.add_argument('--limit', type=int, default=0, help='半荘数の上限（0=全部）')
    ap.add_argument('--max-turn', type=int, default=3,
                    help='この巡目までの打牌に限る（既定 3。盤面情報が少ない序盤ほど'
                         '不一致を文脈依存で正当化しにくい）')
    args = ap.parse_args()

    legs: dict[str, list] = {}
    revs: dict[str, list] = {}
    for name, d in [('init', args.init_logs), (args.label, args.ckpt_logs)]:
        paths = sorted(Path(d).glob('*.json.gz'))
        if not paths:
            raise RuntimeError(f'no logs in {d}')
        if args.limit:
            paths = paths[:args.limit]
        collected = [collect_log(p, args.max_turn) for p in paths]
        legs[name] = [c[0] for c in collected]
        revs[name] = [c[1][0] for c in collected]
        print(f'[{name}] {len(paths)} hanchan / 序盤(≤{args.max_turn}巡)の打牌決定 '
              f'{sum(len(r) for r in legs[name]):,} / 孤立牌の切り順ペア '
              f'{sum(p for _, p in revs[name]):,}')

    print(f"\n{'metric':<40}{'init':>11}{args.label:>13}{'diff':>10}{'SE':>8}{'z':>8}")

    def rev_rate(rows):
        tot_d = sum(p for _, p in rows)
        if not tot_d:
            return None, None
        rr = sum(n for n, _ in rows) / tot_d
        var = sum((n - rr * p) ** 2 for n, p in rows) / tot_d**2
        return rr, math.sqrt(var)

    a, sa = rev_rate(revs['init'])
    b, sb = rev_rate(revs[args.label])
    if a is not None and b is not None:
        diff = (b - a) * 100
        se = math.sqrt(sa**2 + sb**2) * 100
        print(f'{"孤立牌の切り順 逆転率":<40}{a*100:>10.2f}%{b*100:>12.2f}%{diff:>+10.2f}'
              f'{se:>8.2f}{diff/se if se else float("nan"):>+8.2f}')

    for with_turn, lab in [(False, '不一致率(手牌のみ一致)'),
                           (True, '不一致率(手牌+巡目 一致)')]:
        a, sa, ka, na = disagreement(legs['init'], with_turn)
        b, sb, kb, nb = disagreement(legs[args.label], with_turn)
        if a is None or b is None:
            print(f'{lab:<40}{"(分母0)":>11}')
            continue
        diff = (b - a) * 100
        se = math.sqrt(sa**2 + sb**2) * 100
        print(f'{lab:<40}{a*100:>10.2f}%{b*100:>12.2f}%{diff:>+10.2f}{se:>8.2f}'
              f'{diff/se if se else float("nan"):>+8.2f}')
        print(f'{"  (再出現キー数 / 対象決定数)":<40}'
              f'{f"{ka:,}/{na:,}":>11}{f"{kb:,}/{nb:,}":>13}')

    print('\n注: 不一致は誤りではない（方策は手牌以外の盤面も見ているので文脈依存は正当）。'
          '絶対値ではなく init との差分で読むこと。差が有意に正なら「手牌以外の文脈に'
          'より敏感になった＝ぶれやすくなった」と読める。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
