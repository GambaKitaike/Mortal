#!/usr/bin/env python3
"""副露手の役の有無を判定する（read-only）— 鳴き指標の分母を締めるための部品。

## なぜ要るのか

`analyze_call_quality.py` の「テンパイ機会」は、鳴いてテンパイに取れる機会を
すべて数えている。しかし**役の無いテンパイに取れるだけの鳴きは、見送るのが正しい**
（副露手は役が無ければ和了れない）。そのため見送り率の絶対値が9割を超え、
`policy_quality_metrics_20260728.md` §2b で「絶対値ではなく差分を読め」という
但し書きを付ける羽目になっていた。本モジュールはその分母を
**「鳴けば有役テンパイに取れる機会」**へ締めるために使う。

## 判定の範囲（**門前役は扱わない**）

対象は必ず**副露済みの手**（ポン/チー/大明槓を取った直後）なので、
立直・門前ツモ・平和・一盃口・七対子などは構造的に成立しない。よって
open-hand で成立しうる役だけを見る:

  役牌（三元牌・場風・自風）/ タンヤオ / ホンイツ / チンイツ / トイトイ /
  チャンタ / ジュンチャン / 混老頭 / 小三元 / 大三元 / 三色同順 / 三色同刻 /
  一気通貫 / 三槓子 / 字一色 / 清老頭 / 小四喜 / 大四喜 / 緑一色

**意図的に除外**: 海底・河底・嶺上開花・搶槓。いずれも**巡目の運**であって
「その鳴きが手として和了れる形か」という問いに関係しないため。したがって
本モジュールが False を返す手でも、実戦ではこれらで和了れることがある
（検証ではこの4クラスの和了を母集団から除く）。

## 正しさの担保

`--validate` で eval 牌譜を再生し、**実際に起きた副露和了**をすべて拾って
本モジュールに掛ける。実際に和了れた以上その手には役があったはずなので、
**1件でも False が出れば役の取りこぼし**であり大声で落ちる。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / 'mortal'))

TERMINALS = frozenset({0, 8, 9, 17, 18, 26})
HONORS = frozenset(range(27, 34))
TERM_HONOR = TERMINALS | HONORS
DRAGONS = (31, 32, 33)          # 白 發 中
WINDS = (27, 28, 29, 30)        # E S W N
GREENS = frozenset({19, 20, 21, 23, 25, 32})  # 2s3s4s6s8s 發


def _suit(t: int) -> int:
    return 3 if t >= 27 else t // 9


def _decompose(counts: list[int], need: int):
    """concealed 部分を `need` 面子 + 雀頭1 に分解する全通りを yield する。

    yield されるのは (面子のタプル, 雀頭の牌)。面子は ('k', tile) = 刻子 /
    ('s', base) = 順子。
    """
    def sets_only(c: list[int], i: int, acc: tuple):
        while i < 34 and c[i] == 0:
            i += 1
        if i == 34:
            if len(acc) == need:
                yield acc
            return
        if len(acc) >= need:
            return
        if c[i] >= 3:
            c[i] -= 3
            yield from sets_only(c, i, acc + (('k', i),))
            c[i] += 3
        if i < 27 and i % 9 <= 6 and c[i + 1] and c[i + 2]:
            c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
            yield from sets_only(c, i, acc + (('s', i),))
            c[i] += 1; c[i + 1] += 1; c[i + 2] += 1

    for p in range(34):
        if counts[p] < 2:
            continue
        counts[p] -= 2
        seen = set()
        for acc in sets_only(counts, 0, ()):
            key = tuple(sorted(acc))
            if key not in seen:
                seen.add(key)
                yield acc, p
        counts[p] += 2


def open_hand_yaku(concealed: tuple[int, ...], melds: list[tuple[str, int]],
                   win_tile: int, bakaze: int, jikaze: int) -> set[str]:
    """副露手の和了形に成立しうる役の集合を返す（空集合 = 役無し）。

    `concealed` は門前部分の 34 カウント（**和了牌を含まない**）、
    `melds` は [('k'|'s'|'kan'|'ankan', tile), ...]。
    """
    full = list(concealed)
    full[win_tile] += 1
    need = 4 - len(melds)

    meld_sets = []
    n_kan = 0
    for kind, t in melds:
        if kind in ('kan', 'ankan'):
            n_kan += 1
            meld_sets.append(('k', t))
        else:
            meld_sets.append((kind, t))

    # 手牌全体（副露分も含む）の枚数
    all_counts = list(full)
    for kind, t in melds:
        if kind == 's':
            for d in range(3):
                all_counts[t + d] += 1
        else:
            all_counts[t] += 3

    present = {i for i, c in enumerate(all_counts) if c}
    suits = {_suit(t) for t in present}
    num_suits = suits - {3}

    found: set[str] = set()

    # --- 分解に依存しない役 ---
    if not (present & TERM_HONOR):
        found.add('tanyao')
    if present <= HONORS:
        found.add('tsuuiisou')
    if present <= TERMINALS:
        found.add('chinroutou')
    if present <= TERM_HONOR:
        found.add('honroutou')
    if present <= GREENS:
        found.add('ryuuiisou')
    if len(num_suits) == 1:
        found.add('chinitsu' if 3 not in suits else 'honitsu')

    # --- 分解が要る役 ---
    for acc, pair in _decompose(full, need):
        sets = tuple(acc) + tuple(meld_sets)
        koutsu = [t for k, t in sets if k == 'k']
        shuntsu = [t for k, t in sets if k == 's']

        for d in DRAGONS:
            if d in koutsu:
                found.add('yakuhai_dragon')
        if bakaze in koutsu:
            found.add('yakuhai_bakaze')
        if jikaze in koutsu:
            found.add('yakuhai_jikaze')

        n_dragon = sum(1 for d in DRAGONS if d in koutsu)
        if n_dragon == 3:
            found.add('daisangen')
        elif n_dragon == 2 and pair in DRAGONS:
            found.add('shousangen')
        n_wind = sum(1 for w in WINDS if w in koutsu)
        if n_wind == 4:
            found.add('daisuushii')
        elif n_wind == 3 and pair in WINDS:
            found.add('shousuushii')

        if len(koutsu) == 4:
            found.add('toitoi')
        if n_kan >= 3:
            found.add('sankantsu')

        blocks = [{t} for t in koutsu] + [{b, b + 1, b + 2} for b in shuntsu] + [{pair}]
        if all(b & TERM_HONOR for b in blocks):
            found.add('junchan' if not (present & HONORS) else 'chanta')

        for base in shuntsu:
            r = base % 9
            if all(any(s % 9 == r and _suit(s) == q for s in shuntsu) for q in range(3)):
                found.add('sanshoku_doujun')
                break
        for t in koutsu:
            if t < 27:
                r = t % 9
                if all(any(k % 9 == r and _suit(k) == q for k in koutsu) for q in range(3)):
                    found.add('sanshoku_doukou')
                    break
        for q in range(3):
            base = q * 9
            if all(b in shuntsu for b in (base, base + 3, base + 6)):
                found.add('ittsuu')
                break

    return found


# --------------------------------------------------------------------------
# 検証: 実際に起きた副露和了はすべて「役あり」と判定されねばならない
# --------------------------------------------------------------------------

SITUATIONAL = 'situational'  # 海底/河底/嶺上/搶槓 — 母集団から除く
WALL_DRAWS = 70              # 1局の生牌山（実測: 流局 222/222 件でちょうど 70）


def _validate(log_dirs: list[Path], limit: int) -> int:
    from libriichi.state import PlayerState
    from hand_replay import TRANSPARENT_EVENTS, tile_id

    checked = failed = skipped = menzen = 0
    fails = []
    for d in log_dirs:
        paths = sorted(d.glob('*.json.gz'))
        if limit:
            paths = paths[:limit]
        for path in paths:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                events = [json.loads(line) for line in f if line.strip()]
            states = [PlayerState(i) for i in range(4)]
            bakaze = jikaze = 0
            oya = 0
            prev_type = None
            last_was_kan_draw = [False] * 4
            tsumo_count = 0
            for ev in events:
                line = json.dumps(ev)
                for i in range(4):
                    states[i].update(line)
                t = ev.get('type')
                if t == 'start_kyoku':
                    bakaze = tile_id(ev['bakaze'])
                    oya = ev['oya']
                    tsumo_count = 0
                    last_was_kan_draw = [False] * 4
                elif t == 'tsumo':
                    tsumo_count += 1
                    # 嶺上ツモの判定は「直前が槓」だが、槓と嶺上ツモの間には
                    # `dora`（新ドラ表示）が挟まる。素朴に prev_type を見ると
                    # 嶺上開花を取り逃す（実例: anchor_c step16000 の
                    # 10032_8192_b、ankan -> dora -> tsumo -> hora）。
                    # reach_accepted の割り込みと同じクラスなので同じ集合で透過する。
                    last_was_kan_draw[ev['actor']] = \
                        prev_type in ('ankan', 'kakan', 'daiminkan')
                elif t == 'hora':
                    actor = ev['actor']
                    st = states[actor]
                    # **暗槓は門前を維持する**ので副露に数えない。暗槓だけの手は
                    # 立直・門前ツモ等の門前役が使えるため本モジュールの対象外
                    # （ここを取り違えて 79 件の偽陰性を出した。2026-07-29）
                    n_open_melds = len(st.pons) + len(st.chis) + len(st.minkans)
                    if n_open_melds == 0:
                        menzen += 1
                        if t not in TRANSPARENT_EVENTS:
                            prev_type = t
                        continue
                    # 状況役（海底/河底/嶺上/搶槓）は母集団から外す。
                    # 海底・河底は「生牌山を引き切った時点の和了」で、実測では
                    # 1局の tsumo イベントはちょうど 70（流局 222/222 件で確認。
                    # 残りは九種九牌の 2-3 件）なので tsumo_count==70 で判定する。
                    is_tsumo = ev.get('target') == actor
                    if (is_tsumo and last_was_kan_draw[actor]) or prev_type == 'kakan' \
                            or tsumo_count >= WALL_DRAWS:
                        skipped += 1
                        if t not in TRANSPARENT_EVENTS:
                            prev_type = t
                        continue
                    # mjai の hora イベントは和了牌を持たないので状態から引く。
                    # ツモ和了なら tehai に和了牌が入っているので取り除き、
                    # ロン和了なら tehai に入っていないのでそのまま使う。
                    conc = list(st.tehai)
                    if is_tsumo:
                        w = st.last_self_tsumo()
                        if w is None:
                            skipped += 1
                            prev_type = t
                            continue
                        win = tile_id(w)
                        if conc[win] == 0:
                            skipped += 1
                            prev_type = t
                            continue
                        conc[win] -= 1
                    else:
                        w = st.last_kawa_tile()
                        if w is None:
                            skipped += 1
                            prev_type = t
                            continue
                        win = tile_id(w)
                    melds = [('k', b) for b in st.pons] + [('s', b) for b in st.chis] \
                        + [('kan', b) for b in st.minkans] + [('ankan', b) for b in st.ankans]
                    jikaze = WINDS[(actor - oya) % 4]
                    y = open_hand_yaku(tuple(conc), melds, win, bakaze, jikaze)
                    checked += 1
                    if not y:
                        failed += 1
                        if len(fails) < 5:
                            fails.append((f'{path.parent.name}/{path.name}', conc, melds, win, bakaze, jikaze))
                if t not in TRANSPARENT_EVENTS:
                    prev_type = t

    print(f'副露和了 {checked:,} 件を判定 / 状況役で除外 {skipped:,} 件 / 門前和了 {menzen:,} 件は対象外')
    for f in fails:
        print(f'  FALSE NEGATIVE {f[0]}: melds={f[2]} win={f[3]} bakaze={f[4]} jikaze={f[5]}')
        print(f'    concealed={tuple(i for i, c in enumerate(f[1]) for _ in range(c))}')
    if failed:
        print(f'FAIL: {failed:,} 件の副露和了を「役無し」と誤判定した')
        return 1
    print('PASS: 実際に和了れた副露手はすべて有役と判定された')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', nargs='+', metavar='GAME_LOG_DIR', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    return _validate([Path(d) for d in args.validate], args.limit)


if __name__ == '__main__':
    sys.exit(main())
