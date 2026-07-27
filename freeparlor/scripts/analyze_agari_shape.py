#!/usr/bin/env python3
"""和了形の構造分類 — 七対子 / 対々和 / 染め手 / 役牌（read-only）。

動機: Gamba 定性レビュー（2026-07-26/27）の所見のうち、
  「七対子とメンツ手を同時並行で進めたりできないのか」
  「ダブ東対子ありでトイトイを拒否して暗刻から切る」
  「中ポンしたならホンイツダッシュしてほしかった。その2000点に意味ある？」
  「役牌の対子落としをむやみにしないでください」
は `analyze_fundamentals_1v3.py` / `diagnose_agari_composition.py` では測れなかった
（mjai の hora イベントは役を持たず、libriichi `Stat` も役別内訳を露出していない）。
本書は**和了形を構造的に判定**してこの穴を埋める。役判定器ではなく、
上記4分類のみを厳密に判定する。

## 判定の定義（構造のみ。点数計算や役の複合は扱わない）

牌種は赤ドラを正規化（`5mr`→`5m`）。副露は種別（chi/pon/kan）を保持する。

- **七対子**: 副露ゼロ、かつ 14 枚が 7 種 × 各2枚
- **対々和**: 副露に chi が無く、かつ門前部分が
  「ちょうど1種が2枚（雀頭）+ 残りは各3枚」＝ (4 − 副露数) 個の刻子 + 雀頭
  （槓は副露側に入るので枚数計算に影響しない）
- **染め手**: 全14枚（副露込み）が単一数牌スート + 字牌のみ。
  字牌を含まないものを清一色、含むものを混一色として内訳も出す
- **役牌絡み**: 三元牌 / 自風 / 場風 の刻子・槓を含む（門前・副露を問わない）。
  自風は `(seat − oya) % 4` → E/S/W/N、場風は `start_kyoku.bakaze`

**限界**: 平和・タンヤオ・一気通貫等は判定しない（本書の目的外）。
複合（例: 対々和かつ染め手）は各分類で独立に計上するので合計は 100% にならない。

## 和了牌の復元

hora イベントは和了牌を持たない（実測。`{'type','actor','target','deltas','ura_markers'}`）。
  - ロン（actor != target）: 直前の `dahai` の牌
  - ツモ（actor == target）: 直前の `tsumo` の牌（既に手牌にある）
手牌は `start_kyoku.tehais` から再構成し、整合が崩れたら**大声で落とす**。

SE は半荘クラスタの ratio-estimator、差の SE は独立2標本近似
（`analyze_fundamentals_1v3.py` と同じ）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_freeparlor_pnl_1v3 import seat_from_filename  # noqa: E402

WINDS = ['E', 'S', 'W', 'N']
DRAGONS = {'P', 'F', 'C'}          # 白 發 中
HONORS = set(WINDS) | DRAGONS


def norm(t: str) -> str:
    return t[:-1] if t.endswith('r') else t


def suit_of(t: str) -> str:
    return 'z' if t in HONORS else t[-1]


def is_chiitoi(concealed: list[str], melds: list[dict]) -> bool:
    if melds:
        return False
    c = Counter(concealed)
    return len(c) == 7 and all(v == 2 for v in c.values())


def is_toitoi(concealed: list[str], melds: list[dict]) -> bool:
    if any(m['kind'] == 'chi' for m in melds):
        return False
    c = Counter(concealed)
    pairs = [k for k, v in c.items() if v == 2]
    triplets = [k for k, v in c.items() if v == 3]
    if len(pairs) != 1:
        return False
    if len(pairs) + len(triplets) != len(c):
        return False
    return len(triplets) == 4 - len(melds)


def flush_kind(all_tiles: list[str]) -> str | None:
    suits = {suit_of(t) for t in all_tiles}
    numbered = suits - {'z'}
    if len(numbered) != 1:
        return None
    return 'chinitsu' if 'z' not in suits else 'honitsu'


def has_yakuhai(concealed: list[str], melds: list[dict], seat: int,
                oya: int, bakaze: str) -> bool:
    jikaze = WINDS[(seat - oya) % 4]
    targets = DRAGONS | {jikaze, bakaze}
    for m in melds:
        if m['kind'] in ('pon', 'kan') and m['tiles'][0] in targets:
            return True
    c = Counter(concealed)
    return any(c[t] >= 3 for t in targets)


def analyze_log(path: Path) -> dict:
    seat = seat_from_filename(path)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        events = [json.loads(line) for line in f if line.strip()]

    out = Counter()
    hand: list[str] = []
    melds: list[dict] = []
    oya = 0
    bakaze = 'E'
    last_dahai = None
    last_tsumo = None

    for ev in events:
        t = ev.get('type')

        if t == 'start_kyoku':
            hand = [norm(x) for x in ev['tehais'][seat]]
            melds = []
            oya = ev['oya']
            bakaze = ev['bakaze']
            last_dahai = last_tsumo = None
            continue

        if t == 'tsumo':
            if ev['actor'] == seat:
                hand.append(norm(ev['pai']))
                last_tsumo = norm(ev['pai'])
            continue

        if t == 'dahai':
            if ev['actor'] == seat:
                p = norm(ev['pai'])
                if p not in hand:
                    raise RuntimeError(f'{path}: dahai {p} not in hand {hand}')
                hand.remove(p)
            last_dahai = norm(ev['pai'])
            continue

        if t in ('pon', 'chi', 'daiminkan') and ev['actor'] == seat:
            tiles = [norm(x) for x in ev['consumed']] + [norm(ev['pai'])]
            for c in (norm(x) for x in ev['consumed']):
                if c not in hand:
                    raise RuntimeError(f'{path}: consumed {c} not in hand {hand}')
                hand.remove(c)
            melds.append({'kind': 'kan' if t == 'daiminkan' else t, 'tiles': tiles})
            continue

        if t == 'ankan' and ev['actor'] == seat:
            tiles = [norm(x) for x in ev['consumed']]
            for c in tiles:
                if c not in hand:
                    raise RuntimeError(f'{path}: ankan {c} not in hand {hand}')
                hand.remove(c)
            melds.append({'kind': 'kan', 'tiles': tiles})
            continue

        if t == 'kakan' and ev['actor'] == seat:
            p = norm(ev['pai'])
            if p not in hand:
                raise RuntimeError(f'{path}: kakan {p} not in hand {hand}')
            hand.remove(p)
            for m in melds:
                if m['kind'] == 'pon' and m['tiles'][0] == p:
                    m['kind'] = 'kan'
                    m['tiles'].append(p)
                    break
            continue

        if t != 'hora' or ev['actor'] != seat:
            continue

        # --- 和了形の確定 ---
        concealed = list(hand)
        if ev['actor'] != ev['target']:          # ロン: 和了牌は手牌に無い
            if last_dahai is None:
                raise RuntimeError(f'{path}: ron without preceding dahai')
            concealed.append(last_dahai)
        elif last_tsumo is None:
            raise RuntimeError(f'{path}: tsumo hora without preceding tsumo')

        expected = 14 - 3 * len(melds)
        if len(concealed) != expected:
            raise RuntimeError(
                f'{path}: concealed {len(concealed)} != {expected} '
                f'(melds={len(melds)}) hand={concealed}'
            )

        all_tiles = concealed + [x for m in melds for x in m['tiles']]
        out['agari'] += 1
        if is_chiitoi(concealed, melds):
            out['chiitoi'] += 1
        if is_toitoi(concealed, melds):
            out['toitoi'] += 1
        fk = flush_kind(all_tiles)
        if fk:
            out['flush'] += 1
            out[fk] += 1
        if has_yakuhai(concealed, melds, seat, oya, bakaze):
            out['yakuhai'] += 1

    return dict(out)


def ratio_and_se(rows, num, den):
    tot_d = sum(r.get(den, 0) for r in rows)
    if tot_d == 0:
        return None, None
    tot_n = sum(r.get(num, 0) for r in rows)
    rr = tot_n / tot_d
    var = sum((r.get(num, 0) - rr * r.get(den, 0)) ** 2 for r in rows) / tot_d**2
    return rr, math.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--init-logs', required=True)
    ap.add_argument('--ckpt-logs', required=True)
    ap.add_argument('--label', default='ckpt')
    args = ap.parse_args()

    legs = {}
    for name, d in [('init', args.init_logs), (args.label, args.ckpt_logs)]:
        paths = sorted(Path(d).glob('*.json.gz'))
        if not paths:
            raise RuntimeError(f'no logs in {d}')
        legs[name] = [analyze_log(p) for p in paths]
        print(f'[{name}] {len(paths)} hanchan / 和了 {sum(r.get("agari",0) for r in legs[name]):,} 件')

    print(f"\n和了に占める割合\n{'分類':<22}{'init':>11}{args.label:>13}{'diff':>10}{'SE':>8}{'z':>8}")
    for lab, key in [('七対子', 'chiitoi'), ('対々和', 'toitoi'),
                     ('染め手（混+清）', 'flush'), ('  うち清一色', 'chinitsu'),
                     ('  うち混一色', 'honitsu'), ('役牌絡み', 'yakuhai')]:
        a, sa = ratio_and_se(legs['init'], key, 'agari')
        b, sb = ratio_and_se(legs[args.label], key, 'agari')
        if a is None or b is None:
            continue
        diff = (b - a) * 100
        se = math.sqrt(sa**2 + sb**2) * 100
        print(f'{lab:<22}{a*100:>10.2f}%{b*100:>12.2f}%{diff:>+10.2f}{se:>8.2f}{diff/se:>+8.2f}')

    print('\n注: 複合（対々和かつ染め手 等）は各分類で独立計上するため合計は 100% にならない。'
          '平和・タンヤオ等は判定対象外。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
