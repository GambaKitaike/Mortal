#!/usr/bin/env python3
"""牌効率そのものの誤りの指標化（read-only）— 起票: `qualitative_review_anchor_k_20260728.md` §6-4。

## 動機

レビューが拾った「牌理の純粋なミス」（ArmK §5b: 東2局2本場 4巡目 打8m、
6巡目 打3s でドラ受け消滅、東1局 3巡目 打6p）は、既存のどの集計指標にも対応物が無い。
放銃率・和了率は**結果**であり、打牌選択そのものの誤りは捉えない。

本書は各打牌を「その局面で選べた他の打牌」と機械的に比較する。比較の基準は
向聴と受け入れ枚数だけで、**人間の価値判断は一切入れない**（`shanten.py` の
計算器で全候補を列挙するだけ）。

## 測る量

決定点 = **自分のツモ番の打牌**（`can_discard` かつ直前イベントが `tsumo`）。

各候補打牌 d について (向聴(d), 受け入れ枚数(d)) を計算し、実際の選択と比べる:

  1. **向聴を外した率** — 向聴(選択) > min_d 向聴(d)。すなわち「もっと進む切り方が
     あったのに向聴を戻した/据え置いた」割合
  2. **受け入れで支配された率** — 向聴は最小だが、同じ向聴で受け入れがより多い
     候補が存在した割合
  3. **平均受け入れ損失（枚）** — 上記2 の局面での (最大受け入れ − 選択の受け入れ)
  4. （参考）最善手一致率 — 向聴最小かつ受け入れ最大の候補を選んだ割合

## 対象の絞り込み（牌効率が目的関数だと言える局面に限る）

  - **自分のツモ番の打牌のみ**（鳴き直後の打牌は除外）。食い替え制限を考慮せずに
    候補を列挙すると違法な打牌を「より良い代替」に数えてしまうため、
    その可能性がある局面を最初から外す
  - **自分が立直していない**（立直後はツモ切り強制で選択が存在しない）
  - **他家に立直者がいない**（守備が目的関数になる局面を混ぜない）
  - **向聴 ≤ `--max-shanten`（既定 3）**（深い局面は牌効率以外の要素が支配的。
    libriichi の SP 計算も向聴4以上では確率を出さない）

## 限界（明記する）

**「支配された打牌 = 誤り」ではない。** 打点（ドラ・役・赤）、安全度、他家の動向は
一切見ていないので、受け入れを捨てて打点を取る正当な選択も「支配された」に数える。
したがって本指標の絶対値は牌理の巧拙を意味しない。**意味があるのは init との差分**で、
差が有意なら「同じ局面クラスで、より受け入れを捨てる方向に動いた」と言える。
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
from shanten import shanten, ukeire  # noqa: E402


def analyze_log(path: Path, max_shanten: int) -> dict:
    seat = seat_from_filename(path)
    r = {
        'n_dec': 0, 'n_shanten_miss': 0,
        'n_same_shanten': 0, 'n_dominated': 0,
        'uke_loss': 0, 'n_best': 0,
        'skipped': {},
    }

    pending = None
    for ev, dec in replay(path, seat, shanten_fn=shanten):
        if ev.get('type') in TRANSPARENT_EVENTS:
            continue
        # 直前の決定点で選ばれた打牌を、次に来る dahai イベントで確定させる
        if pending is not None:
            # `can_discard` の局面への応答は打牌とは限らない（暗槓・加槓・ツモ和了・
            # 九種九牌）。それらは「打牌の選択」ではないので測定対象から外すが、
            # 黙って落とすとサンプルが静かに欠けるので種別ごとに数えて報告する。
            if ev.get('actor') != seat or ev.get('type') != 'dahai':
                kind = ev.get('type') if ev.get('actor') == seat else f'other:{ev.get("type")}'
                r['skipped'][kind] = r['skipped'].get(kind, 0) + 1
                pending = None
                continue
            cand, best_sh, best_uke_at_best_sh = pending
            chosen = tile_id(ev['pai'])
            if chosen not in cand:
                raise RuntimeError(f'{path}: discarded {ev["pai"]} not among candidates')
            sh_c, uke_c = cand[chosen]
            r['n_dec'] += 1
            if sh_c > best_sh:
                r['n_shanten_miss'] += 1
            else:
                r['n_same_shanten'] += 1
                if uke_c < best_uke_at_best_sh:
                    r['n_dominated'] += 1
                    r['uke_loss'] += best_uke_at_best_sh - uke_c
                else:
                    r['n_best'] += 1
            pending = None

        if dec is None:
            continue
        cans = dec.cans
        if not cans.can_discard or ev.get('type') != 'tsumo':
            continue
        if dec.self_riichi or dec.others_riichi:
            continue
        if cans.can_tsumo_agari or dec.shanten > max_shanten:
            continue

        lst = list(dec.tehai)
        cand: dict[int, tuple[int, int]] = {}
        for t in range(34):
            if not lst[t]:
                continue
            lst[t] -= 1
            hand = tuple(lst)
            sh = shanten(hand, dec.n_open)
            cand[t] = (sh, ukeire(hand, dec.n_open, dec.seen)[1])
            lst[t] += 1
        best_sh = min(v[0] for v in cand.values())
        best_uke = max(v[1] for v in cand.values() if v[0] == best_sh)
        pending = (cand, best_sh, best_uke)

    return r


def ratio_and_se(rows, num, den):
    tot_d = sum(r[den] for r in rows)
    if tot_d == 0:
        return None, None
    tot_n = sum(r[num] for r in rows)
    rr = tot_n / tot_d
    var = sum((r[num] - rr * r[den]) ** 2 for r in rows) / tot_d**2
    return rr, math.sqrt(var)


def mean_and_se(rows, num, den):
    """半荘クラスタの ratio-estimator（比率でなく平均量にも同じ式が使える）。"""
    return ratio_and_se(rows, num, den)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--init-logs', required=True)
    ap.add_argument('--ckpt-logs', required=True)
    ap.add_argument('--label', default='ckpt')
    ap.add_argument('--limit', type=int, default=0, help='半荘数の上限（0=全部）')
    ap.add_argument('--max-shanten', type=int, default=3)
    args = ap.parse_args()

    legs = {}
    for name, d in [('init', args.init_logs), (args.label, args.ckpt_logs)]:
        paths = sorted(Path(d).glob('*.json.gz'))
        if not paths:
            raise RuntimeError(f'no logs in {d}')
        if args.limit:
            paths = paths[:args.limit]
        legs[name] = [analyze_log(p, args.max_shanten) for p in paths]
        skipped: dict[str, int] = {}
        for r in legs[name]:
            for k, v in r['skipped'].items():
                skipped[k] = skipped.get(k, 0) + v
        skip_txt = ', '.join(f'{k}={v:,}' for k, v in sorted(skipped.items())) or 'なし'
        print(f'[{name}] {len(paths)} hanchan / 対象打牌 '
              f'{sum(r["n_dec"] for r in legs[name]):,} '
              f'/ 打牌以外の応答で除外: {skip_txt}')

    print(f"\n{'metric':<40}{'init':>11}{args.label:>13}{'diff':>10}{'SE':>8}{'z':>8}")
    specs = [
        ('向聴を外した率',            'n_shanten_miss', 'n_dec'),
        ('受け入れで支配された率',    'n_dominated',    'n_same_shanten'),
        ('(参考)最善手一致率',        'n_best',         'n_same_shanten'),
    ]
    for lab, num, den in specs:
        a, sa = ratio_and_se(legs['init'], num, den)
        b, sb = ratio_and_se(legs[args.label], num, den)
        if a is None or b is None:
            print(f'{lab:<40}{"(分母0)":>11}')
            continue
        diff = (b - a) * 100
        se = math.sqrt(sa**2 + sb**2) * 100
        print(f'{lab:<40}{a*100:>10.2f}%{b*100:>12.2f}%{diff:>+10.2f}{se:>8.2f}'
              f'{diff/se if se else float("nan"):>+8.2f}')

    a, sa = mean_and_se(legs['init'], 'uke_loss', 'n_dominated')
    b, sb = mean_and_se(legs[args.label], 'uke_loss', 'n_dominated')
    if a is not None and b is not None:
        diff = b - a
        se = math.sqrt(sa**2 + sb**2)
        print(f'{"平均受け入れ損失(枚/支配局面)":<40}{a:>11.2f}{b:>13.2f}{diff:>+10.2f}'
              f'{se:>8.2f}{diff/se if se else float("nan"):>+8.2f}')

    print('\n注: 打点・安全度・他家の動向を一切見ていない純粋な受け入れ比較。'
          '「支配された打牌」には打点を取る正当な選択も含まれるので、絶対値は牌理の'
          '巧拙を意味しない。読むべきは init との差分。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
