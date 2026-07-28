#!/usr/bin/env python3
"""鳴きの「質」の指標化（read-only）— 起票: `qualitative_review_anchor_k_20260728.md` §6-3。

## 動機

鳴きの**頻度**（fuuro_rate / 副露和了割合）は既に測れているが、
**その鳴きが手を前進させたか**は測れていない。Gamba のレビューは、鳴き判断が
**双方向に壊れている**ことを指している:

  - 鳴くべきで鳴かない: 「7p スルー。ポンしたらマンガンのテンパイ」（ArmK §5b）
    「8s チーをスルー。残りターツが全て良型なので鳴きたい」（同 §2）
  - 鳴くべきでないのに鳴く: 「7p ポンはかなり論外寄り。待ちが減りツモ番も失う」（同 §5b）
    「南を暗刻からポン → その南をそのまま切る」（ArmC §1）

副露率が下がったこと（init 17.33% → K 10.57%）は前者しか説明しない。
本書は鳴き機会を**取った側と見送った側の両方**で評価する。

## 測る量

鳴き機会 = `PlayerState.last_cans` が pon / chi / daiminkan を許した決定点
（合法性判定は libriichi 自身のものを使う。自分が立直済みの局面は鳴けないので対象外）。

**取った鳴き**（分母 = 実際に鳴いた回数）:
  1. 向聴前進率 — 鳴いた結果、最善の打牌後の向聴が鳴く前より小さい割合
  2. 鳴きテンパイ率 — 鳴いた結果テンパイに取れた割合
  3. 受け入れ変化 — 鳴き前の受け入れ枚数と、鳴き後（最善打牌後）の受け入れ枚数の差

**見送った鳴き機会**（分母 = その鳴きが有効だった機会）:
  4. テンパイ機会の見送り率 — 「鳴けばテンパイできた」機会のうち鳴かなかった割合
  5. 向聴前進機会の見送り率 — 「鳴けば向聴が進む」機会のうち鳴かなかった割合

3 以外は向聴だけで決まるので安い。3 のみ受け入れ計算（34 種の試し引き）が要る。

## 限界（明記する）

  - **向聴前進 = 良い鳴き、ではない**。役無し・守備・打点を無視した指標であり、
    「鳴かないほうが良い」局面で鳴かなかったことは 4/5 では評価できない。
    本指標は**方策間の差分**として読むもので、絶対値の良し悪しは判定しない
  - 役の有無は判定していない（役判定器が要る）。「役無しでテンパイに取れる鳴き」も
    テンパイ機会に数えている。`analyze_agari_shape.py` の役牌絡み指標と併読すること
  - 打点の変化は本書では測らない（和了しない限り確定しないため）。
    和了時の打点は `diagnose_agari_composition.py` が層別で扱う
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
from hand_replay import TRANSPARENT_EVENTS, call_variants, replay  # noqa: E402
from shanten import shanten, ukeire  # noqa: E402

CALL_EVENTS = ('pon', 'chi', 'daiminkan')


def _best_after_call(tehai, n_open, seen, want_ukeire):
    """鳴き後（3n+2）の最善打牌を選び、(向聴, 受け入れ枚数) を返す。

    最善 = 向聴最小、同着なら受け入れ枚数最大。`want_ukeire=False` なら
    受け入れは計算しない（None）。
    """
    lst = list(tehai)
    best_sh = 99
    best_uke = -1
    for t in range(34):
        if not lst[t]:
            continue
        lst[t] -= 1
        cand = tuple(lst)
        sh = shanten(cand, n_open)
        if sh < best_sh:
            best_sh = sh
            best_uke = -1
        if want_ukeire and sh == best_sh:
            _, tiles = ukeire(cand, n_open, seen)
            if tiles > best_uke:
                best_uke = tiles
        lst[t] += 1
    return best_sh, (best_uke if want_ukeire else None)


def analyze_log(path: Path, want_ukeire: bool) -> dict:
    seat = seat_from_filename(path)
    r = {
        'n_taken': 0, 'n_taken_advance': 0, 'n_taken_tenpai': 0,
        'uke_before': 0, 'uke_after': 0, 'n_uke': 0,
        'n_opp_tenpai': 0, 'n_opp_tenpai_declined': 0,
        'n_opp_advance': 0, 'n_opp_advance_declined': 0,
        'n_opp': 0, 'n_call_events': 0, 'n_opp_ron': 0,
    }

    pending = None  # 直前の決定点が鳴き機会だった場合のスナップショット
    for ev, dec in replay(path, seat, shanten_fn=shanten):
        if ev.get('type') in TRANSPARENT_EVENTS:
            continue
        # 直前の鳴き機会が「取られた」かどうかは、次に来るイベントで分かる
        if pending is not None:
            taken = ev.get('type') in CALL_EVENTS and ev.get('actor') == seat
            sh_before, best_sh, best_uke, uke_before = pending
            if not taken and ev.get('type') == 'hora' and ev.get('actor') == seat:
                # ロンで応じた機会は「鳴きを見送った」ではない（和了が優越する）。
                # 見送り率の分母から外す。黙って落とさず件数を残す
                r['n_opp_ron'] += 1
                pending = None
                continue
            if taken:
                r['n_taken'] += 1
                if best_sh < sh_before:
                    r['n_taken_advance'] += 1
                if best_sh == 0:
                    r['n_taken_tenpai'] += 1
                if want_ukeire and best_uke is not None and best_uke >= 0:
                    r['uke_before'] += uke_before
                    r['uke_after'] += best_uke
                    r['n_uke'] += 1
            if sh_before >= 1 and best_sh == 0:
                r['n_opp_tenpai'] += 1
                if not taken:
                    r['n_opp_tenpai_declined'] += 1
            if best_sh < sh_before:
                r['n_opp_advance'] += 1
                if not taken:
                    r['n_opp_advance_declined'] += 1
            pending = None

        if ev.get('type') in CALL_EVENTS and ev.get('actor') == seat:
            r['n_call_events'] += 1

        if dec is None:
            continue
        cans = dec.cans
        if dec.self_riichi or dec.target_tile is None:
            continue
        if not (cans.can_pon or cans.can_chi or cans.can_daiminkan):
            continue

        variants = call_variants(dec.tehai, dec.target_tile, cans)
        if not variants:
            # last_cans が鳴きを許したのに手牌から消費できない = 再生の破れ
            raise RuntimeError(
                f'{path}: call allowed but no variant (target={dec.target_tile} '
                f'tehai={dec.tehai})'
            )
        r['n_opp'] += 1
        best_sh, best_uke = 99, -1
        for _label, hand in variants:
            sh, uke = _best_after_call(hand, dec.n_open + 1, dec.seen, want_ukeire)
            if sh < best_sh or (sh == best_sh and uke is not None and uke > best_uke):
                best_sh, best_uke = sh, uke
        uke_before = ukeire(dec.tehai, dec.n_open, dec.seen)[1] if want_ukeire else 0
        pending = (dec.shanten, best_sh, best_uke, uke_before)

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
    ap.add_argument('--limit', type=int, default=0, help='半荘数の上限（0=全部）')
    ap.add_argument('--no-ukeire', action='store_true',
                    help='受け入れ計算を省く（向聴系の指標だけなら大幅に速い）')
    args = ap.parse_args()

    want_uke = not args.no_ukeire
    legs = {}
    for name, d in [('init', args.init_logs), (args.label, args.ckpt_logs)]:
        paths = sorted(Path(d).glob('*.json.gz'))
        if not paths:
            raise RuntimeError(f'no logs in {d}')
        if args.limit:
            paths = paths[:args.limit]
        rows = [analyze_log(p, want_uke) for p in paths]
        legs[name] = rows
        taken = sum(r['n_taken'] for r in rows)
        events = sum(r['n_call_events'] for r in rows)
        if taken != events:
            raise RuntimeError(
                f'{name}: taken calls {taken} != call events {events} '
                f'(鳴き機会と実イベントの対応が壊れている)'
            )
        print(f'[{name}] {len(paths)} hanchan / 鳴き機会 '
              f'{sum(r["n_opp"] for r in rows):,} / 実際に鳴いた {taken:,}')

    print(f"\n{'metric':<40}{'init':>11}{args.label:>13}{'diff':>10}{'SE':>8}{'z':>8}")
    specs = [
        ('取った鳴き: 向聴前進率',   'n_taken_advance',        'n_taken'),
        ('取った鳴き: テンパイ率',   'n_taken_tenpai',         'n_taken'),
        ('テンパイ機会の見送り率',   'n_opp_tenpai_declined',  'n_opp_tenpai'),
        ('向聴前進機会の見送り率',   'n_opp_advance_declined', 'n_opp_advance'),
    ]
    for lab, num, den in specs:
        a, sa = ratio_and_se(legs['init'], num, den)
        b, sb = ratio_and_se(legs[args.label], num, den)
        if a is None or b is None:
            print(f'{lab:<40}{"(分母0)":>11}')
            continue
        diff = (b - a) * 100
        se = math.sqrt(sa**2 + sb**2) * 100
        z = diff / se if se else float('nan')
        print(f'{lab:<40}{a*100:>10.2f}%{b*100:>12.2f}%{diff:>+10.2f}{se:>8.2f}{z:>+8.2f}')

    if want_uke:
        print()
        for name in ('init', args.label):
            rows = legs[name]
            n = sum(r['n_uke'] for r in rows)
            if not n:
                continue
            before = sum(r['uke_before'] for r in rows) / n
            after = sum(r['uke_after'] for r in rows) / n
            print(f'[{name}] 鳴き前後の受け入れ枚数: {before:.2f} -> {after:.2f} '
                  f'({after - before:+.2f}, n={n:,})')

    print('\n注: 分母が違う指標を並べている。上2本は「取った鳴き」、下2本は「その鳴きが'
          '有効だった機会」が分母。向聴前進は良い鳴きの必要条件であって十分条件ではなく'
          '（役・守備・打点を見ていない）、絶対値ではなく init との差分で読むこと。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
