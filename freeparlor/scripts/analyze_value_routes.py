#!/usr/bin/env python3
"""打点を作りに行く手順の指標化（read-only）— 起票: `anchor_arm_k_result.md` §6-2 /
`qualitative_review_anchor_k_20260728.md` §4-2。

## 動機

Gamba のレビューが繰り返し指す症状は「和了**率**には現れないが手順に現れる」型で、
既存のどの指標にも対応物が無い:

  - 「南3局 4巡目 78m 落としてマンガンのホンイツ一向聴。**2000点あがっても
    しゃあなくない？**」（ArmK §2）— 打点ルートを取らず安手で決着させる
  - 「中ポン後にホンイツへ行かず 2000点で終える」「跳満テンパイを取らない」
    「ダブ東対子ありでトイトイを拒否して暗刻から切る」（ArmC §1）

`diagnose_agari_composition.py` は**和了した後**の打点を層別するが、
「そもそも打点ルートに乗らなかった」局は和了に現れないので捕まらない。
本書は**打牌の時点で**打点ルートを畳んだかを数える。

## 測る量

決定点 = 自分のツモ番の打牌（`can_discard` かつ直前イベントが `tsumo`）、
自分が立直しておらず、向聴 ≤ `--max-shanten`（既定 4。ルートがまだ生きている範囲）。

### 1. 染め手ルートの放棄率

**判定は 2026-07-29 に作り直した**（旧定義の欠陥は §「旧定義の問題」参照）。
色 q について:

  same_num[q] = q の数牌の枚数 / honors = 字牌の枚数 / other[q] = 他色の数牌の枚数

  対象の色 q* = same_num が最大の色（同点は色番号の小さい方。**順序依存を排除**）
  染め手ルートが現実的 = same_num[q*] >= `--flush-suit-min`（既定 5）
                        かつ same_num[q*] + honors >= `--flush-min`（既定 9）
                        かつ other[q*] > 0

`same_num >= 5` を課すのが要点で、これが無いと**字牌が多いだけの手**
（例: 字牌8枚 + 萬子1枚）が「萬子の染め手」と判定されてしまう。

  - **放棄** = q* の数牌を切った（かつ他色の数牌が残っていた
    ＝ 先に切るべき牌があったのに染め色を削った）
  - 放棄率 = 放棄 / 染め手ルートが現実的だった打牌

### 2. ドラ切り率（非ドラの選択肢がある局面で）

手にドラ（表ドラ + 赤）と非ドラが両方あるとき、**ドラのほうを切った**割合。
ドラ表示牌からのドラ牌は通常の順送り（9→1、北→東、中→白）で解決する。

**他家に立直者がいる局面は除外する**（2026-07-29 追加）。そこでのドラ切りは
打点放棄ではなく**守備**であり、混ぜると指標の意味が変わる。旧定義では分母の
21.3% が立直下で、しかもそこはドラ切り率が高かった（2.96% vs 立直なし 2.04%）
ため、「打点放棄」の読みを水増ししていた。`analyze_tile_efficiency.py` は
最初から除外しており、**そちらと定義が揃っていなかった**のも問題だった。

## 旧定義の問題（2026-07-29 に修正・記録として残す）

Gamba の指摘を受けて判定基準を実データで点検したところ、旧定義に3つの欠陥があった:

1. **複数色が同時に条件を満たしうる**のに `break` で萬→筒→索の順に最初の色だけを
   採用していた。実測で **2色該当 6.7% / 3色該当 0.3%**（init 脚 120 半荘、3,179 局面）。
   → same_num の argmax に変更して順序依存を排除
2. **字牌を無制限に同色側へ足していた**ので、字牌が多いだけの手が染め手と判定された。
   実測で **11.1%** の局面が「字牌のほうが同色数牌より多い」。
   → `same_num >= 5` を必須条件に追加
3. **他家の立直下を除外していなかった**（ドラ切り率）。実測で分母の **21.3%** が
   立直下で、そこはドラ切り率が高い（2.96% vs 2.04%）。→ 除外

いずれも init と ckpt の**両方に同じ基準**を当てていたので差分の符号は保たれるが、
局面クラスの定義としては誤っていた。

## 限界（**これを外すと誤読する**）

  - **放棄 = 誤り、ではない。** 染め手を畳むのもドラを切るのも、速度・テンパイ料など
    正当な理由がありうる。**絶対値ではなく init との差分**を読む
  - 染め手の「現実的」は枚数だけの代理指標で、ターツの形は見ていない
  - ドラ切りは打点放棄の一部でしかない（ドラを含むターツの解体は数えていない）
  - **頻度であって質ではない**。「ドラを持ち続ける」と「ドラを活かして打点を作る」は
    別で、前者だけなら抱えて和了れていない可能性がある
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / 'mortal'))

from analyze_freeparlor_pnl_1v3 import seat_from_filename  # noqa: E402
from hand_replay import TRANSPARENT_EVENTS, replay, tile_id  # noqa: E402
from shanten import shanten  # noqa: E402


def dora_from_marker(marker: int) -> int:
    """ドラ表示牌からドラ牌へ（9→1 / 北→東 / 中→白 の順送り）。"""
    if marker < 27:                       # 数牌
        base, rank = marker // 9 * 9, marker % 9
        return base + (rank + 1) % 9
    if marker <= 30:                      # 風牌 E S W N
        return 27 + (marker - 27 + 1) % 4
    return 31 + (marker - 31 + 1) % 3     # 三元牌 P F C


def analyze_log(path: Path, max_shanten: int, flush_min: int,
                flush_suit_min: int) -> dict:
    seat = seat_from_filename(path)
    r = {
        'n_flush': 0, 'n_flush_abandon': 0,
        'n_dora_choice': 0, 'n_dora_cut': 0, 'n_dora_skipped_riichi': 0,
        'n_dec': 0,
    }

    dora: set[int] = set()
    pending = None
    for ev, dec in replay(path, seat, shanten_fn=shanten):
        t = ev.get('type')
        if t == 'start_kyoku':
            dora = {dora_from_marker(tile_id(ev['dora_marker']))}
        elif t == 'dora':
            dora.add(dora_from_marker(tile_id(ev['dora_marker'])))
        if t in TRANSPARENT_EVENTS:
            continue

        if pending is not None:
            if ev.get('actor') == seat and ev.get('type') == 'dahai':
                tehai, cur_dora, aka_held, others_riichi = pending
                pai = ev['pai']
                cut = tile_id(pai)
                is_aka_cut = pai.endswith('r')
                r['n_dec'] += 1

                # --- 1. 染め手ルートの放棄 ---
                honors = sum(tehai[27:34])
                # 対象の色は same_num の argmax（同点は色番号の小さい方）。
                # 「最初に条件を満たした色」だと萬→筒→索の順序に依存する
                q = max(range(3), key=lambda i: sum(tehai[i * 9:i * 9 + 9]))
                lo, hi = q * 9, q * 9 + 9
                same_num = sum(tehai[lo:hi])
                other = sum(tehai[:lo]) + sum(tehai[hi:27])
                if same_num >= flush_suit_min and same_num + honors >= flush_min \
                        and other > 0:
                    r['n_flush'] += 1
                    if lo <= cut < hi:
                        r['n_flush_abandon'] += 1

                # --- 2. ドラ切り（非ドラの選択肢があるとき・**他家立直下は除外**）---
                held_dora = sum(tehai[d] for d in cur_dora) + aka_held
                held_total = sum(tehai)
                if 0 < held_dora < held_total:
                    if others_riichi:
                        r['n_dora_skipped_riichi'] += 1
                    else:
                        r['n_dora_choice'] += 1
                        if cut in cur_dora or is_aka_cut:
                            r['n_dora_cut'] += 1
            pending = None

        if dec is None:
            continue
        if not dec.cans.can_discard or t != 'tsumo':
            continue
        if dec.self_riichi or dec.shanten > max_shanten:
            continue
        pending = (dec.tehai, frozenset(dora), dec.n_aka, dec.others_riichi)

    return r


def ratio_and_se(rows, num, den):
    tot_d = sum(r[den] for r in rows)
    if tot_d == 0:
        return None, None
    tot_n = sum(r[num] for r in rows)
    rr = tot_n / tot_d
    var = sum((r[num] - rr * r[den]) ** 2 for r in rows) / tot_d**2
    return rr, math.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--init-logs', required=True)
    ap.add_argument('--ckpt-logs', required=True)
    ap.add_argument('--label', default='ckpt')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-shanten', type=int, default=4)
    ap.add_argument('--flush-min', type=int, default=9,
                    help='同色+字牌がこの枚数以上で「染め手が現実的」とみなす（既定 9）')
    ap.add_argument('--flush-suit-min', type=int, default=5,
                    help='同色の**数牌**の下限（既定 5）。字牌だけで条件を満たす手を除く')
    args = ap.parse_args()

    legs = {}
    for name, d in [('init', args.init_logs), (args.label, args.ckpt_logs)]:
        paths = sorted(Path(d).glob('*.json.gz'))
        if not paths:
            raise RuntimeError(f'no logs in {d}')
        if args.limit:
            paths = paths[:args.limit]
        legs[name] = [analyze_log(p, args.max_shanten, args.flush_min,
                                  args.flush_suit_min) for p in paths]
        print(f'[{name}] {len(paths)} hanchan / 対象打牌 '
              f'{sum(r["n_dec"] for r in legs[name]):,} / '
              f'染め手が現実的 {sum(r["n_flush"] for r in legs[name]):,} / '
              f'ドラ選択あり {sum(r["n_dora_choice"] for r in legs[name]):,} '
              f'(他家立直下で除外 {sum(r["n_dora_skipped_riichi"] for r in legs[name]):,})')

    print(f"\n{'metric':<40}{'init':>11}{args.label:>13}{'diff':>10}{'SE':>8}{'z':>8}")
    for lab, num, den in [
        ('染め手ルートの放棄率', 'n_flush_abandon', 'n_flush'),
        ('ドラ切り率(非ドラの選択肢あり)', 'n_dora_cut', 'n_dora_choice'),
    ]:
        a, sa = ratio_and_se(legs['init'], num, den)
        b, sb = ratio_and_se(legs[args.label], num, den)
        if a is None or b is None:
            print(f'{lab:<40}{"(分母0)":>11}')
            continue
        diff = (b - a) * 100
        se = math.sqrt(sa**2 + sb**2) * 100
        print(f'{lab:<40}{a*100:>10.2f}%{b*100:>12.2f}%{diff:>+10.2f}{se:>8.2f}'
              f'{diff/se if se else float("nan"):>+8.2f}')

    print('\n注: 放棄もドラ切りも守備・速度の観点では正当でありうる。'
          '絶対値ではなく init との差分で読むこと。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
