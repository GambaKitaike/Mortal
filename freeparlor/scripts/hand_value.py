#!/usr/bin/env python3
"""和了形から**確定打点**（翻・符・点数）を計算する（read-only）。

## 位置づけ — 何が「確定」で何が確定しないか

打点は2つの成分に分かれる。**混ぜてはいけない**（報酬の3ストリームを混ぜないのと
同じ理屈で、混ぜると「どこまでが手作りの成果でどこからが運か」が見えなくなる）:

| 成分 | 性質 | 本モジュール |
|---|---|---|
| 役・符・表ドラ・赤ドラ | 和了形が決まれば**確定** | **計算する** |
| 裏ドラ | 確率だが**期待値は確定** | `expected_ura()` で別に返す |
| 一発・海底・河底・嶺上 | **打ち回しと相手の行動の結果**であって手の性質ではない | **計算しない**（`extra_han` で外から足すことは可能） |

一発を手のラベルに混ぜるのは「運を予測させる」ことになるので、設計として入れない。

## libriichi との関係（**移植であって独自実装ではない**）

役・符・点数の規則は `libriichi/src/algo/agari.rs`（`search_yakus` / `calc_fu`）と
`libriichi/src/algo/point.rs` を忠実に移植した。麻雀の一般論ではなく
**このプロジェクトの正典に合わせる**のが目的なので、「一般的にはこうだ」で
直さないこと（例: 20符の分岐、連風牌 4符、切り上げ満貫あり）。

libriichi 側は面子分解を事前計算テーブル（`AGARI_TABLE`）で引くが、本モジュールは
`shanten.py` と同じ列挙で代用する。**分解を全部試して最大点を取る**のは同じ。

## なぜ Python 側に要るのか

`PlayerState.agari_detail()` は**実際に起きた和了**しか評価できない。
「この手が和了ったら何点か」「この打牌をしたら打点はどう変わるか」という
**反実仮想**には使えない（`shanten.py` と同じ事情）。

## 検証

`--validate` で eval 牌譜の**実際の和了**を全件再計算し、
`PlayerState.agari_detail(is_ron, [])`（裏ドラ無しで呼ぶ）と突き合わせる。
一発の和了と状況役（海底/河底/嶺上/搶槓）は本モジュールの対象外なので母集団から除く。

**実測（3脚 全数、2026-07-29）: 17,152 件すべてで point / han / fu / yakuman が一致。**
除外は 3,045 件（一発・状況役）。

### 検証で判明した libriichi 側の性質（**移植時に踏んだ落とし穴**）

1. `agari_detail` は **hora イベントを適用する前**の状態でしか呼べない
   （適用後は "cannot agari"）。ダブロンでは2件目のために、連続する hora を
   まとめて評価してから適用する必要がある
2. `agari_detail` は `additional_hans` を**内部で数える**（`agent_helper.rs:497`）。
   立直・**ダブル立直**・一発・海底/河底・搶槓・門前ツモ・嶺上。
   このうち `is_w_riichi` は Python に露出していないので牌譜から再構成する
   （自分の第一打牌での立直宣言、かつそれ以前に誰の鳴きも無いこと）
3. **`AgariDetail.num_aka` は暗槓の中の赤を数えない** — `count_num_aka` は
   `akas_in_hand + fuuro_overview` を見るが、暗槓は `ankan_overview` に入るため
   （`update.rs:633`）。**翻の側は取得時に `doras_owned` へ加算済みなので正しい**。
   ずれるのは `num_aka` = **チップ計算の基**（`agari_detail.rs::chip_base`）のほう

### 3 の含意（**チップ経済への影響は実測で無視できる**）

`chip_base = num_aka + num_ura + ippatsu + yakuman*5` なので、
**暗槓に入った赤はチップとして支払われていない**。実雀荘では支払われるので
ルール再現性の欠落だが、規模を測ったところ:

| 脚 | 和了 | 赤入り暗槓 | 和了時にチップから漏れた赤 |
|---|---:|---:|---:|
| init | 6,847 | 36 回 | **4 枚** |
| anchor_k-16000 | 6,701 | 28 回 | **2 枚** |
| anchor_c-16000 | 6,649 | 34 回 | **4 枚** |

800 半荘あたり 2–4 枚 ≈ **0.005 枚/半荘**。判定で問題にしているチップ差
（+0.5 枚/半荘級）の **約 1/100** なので、既存の判定・結論には影響しない。
修正するかは経済プリセット（0b 議題2 / G3）を触るときに併せて裁定すればよい。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / 'mortal'))

from shanten import _kokushi_shanten  # noqa: E402

DRAGONS = (31, 32, 33)
WINDS = (27, 28, 29, 30)
GREEN = frozenset({19, 20, 21, 23, 25, 32})


def is_yaokyuu(t: int) -> bool:
    return t >= 27 or t % 9 in (0, 8)


# --------------------------------------------------------------------------
# 点数表（libriichi/src/algo/point.rs の忠実な移植）
# --------------------------------------------------------------------------

def _point_calc(is_oya: bool, fu: int, han: int) -> tuple[int, int, int]:
    """(ron, tsumo_ko, tsumo_oya) を返す。`point.rs::Point::calc` の移植。"""
    if is_oya:
        table = {
            (20, 2): (2000, 700, 0), (40, 1): (2000, 700, 0),
            (20, 3): (3900, 1300, 0), (40, 2): (3900, 1300, 0), (80, 1): (3900, 1300, 0),
            (20, 4): (7700, 2600, 0), (40, 3): (7700, 2600, 0), (80, 2): (7700, 2600, 0),
            (25, 2): (2400, 800, 0), (50, 1): (2400, 800, 0),
            (25, 3): (4800, 1600, 0), (50, 2): (4800, 1600, 0), (100, 1): (4800, 1600, 0),
            (25, 4): (9600, 3200, 0), (50, 3): (9600, 3200, 0), (100, 2): (9600, 3200, 0),
            (30, 1): (1500, 500, 0),
            (30, 2): (2900, 1000, 0), (60, 1): (2900, 1000, 0),
            (30, 3): (5800, 2000, 0), (60, 2): (5800, 2000, 0),
            (30, 4): (11600, 3900, 0), (60, 3): (11600, 3900, 0),
            (70, 1): (3400, 1200, 0), (70, 2): (6800, 2300, 0),
            (90, 1): (4400, 1500, 0), (90, 2): (8700, 2900, 0),
            (110, 1): (5300, 1800, 0), (110, 2): (10600, 3600, 0),
        }
        big = [(5, (12000, 4000, 0)), (7, (18000, 6000, 0)), (10, (24000, 8000, 0)),
               (12, (36000, 12000, 0))]
        huge = (48000, 16000, 0)
    else:
        table = {
            (20, 2): (1300, 400, 700), (40, 1): (1300, 400, 700),
            (20, 3): (2600, 700, 1300), (40, 2): (2600, 700, 1300), (80, 1): (2600, 700, 1300),
            (20, 4): (5200, 1300, 2600), (40, 3): (5200, 1300, 2600), (80, 2): (5200, 1300, 2600),
            (25, 2): (1600, 400, 800), (50, 1): (1600, 400, 800),
            (25, 3): (3200, 800, 1600), (50, 2): (3200, 800, 1600), (100, 1): (3200, 800, 1600),
            (25, 4): (6400, 1600, 3200), (50, 3): (6400, 1600, 3200), (100, 2): (6400, 1600, 3200),
            (30, 1): (1000, 300, 500),
            (30, 2): (2000, 500, 1000), (60, 1): (2000, 500, 1000),
            (30, 3): (3900, 1000, 2000), (60, 2): (3900, 1000, 2000),
            (30, 4): (7700, 2000, 3900), (60, 3): (7700, 2000, 3900),
            (70, 1): (2300, 600, 1200), (70, 2): (4500, 1200, 2300),
            (90, 1): (2900, 800, 1500), (90, 2): (5800, 1500, 2900),
            (110, 1): (3600, 900, 1800), (110, 2): (7100, 1800, 3600),
        }
        big = [(5, (8000, 2000, 4000)), (7, (12000, 3000, 6000)), (10, (16000, 4000, 8000)),
               (12, (24000, 6000, 12000))]
        huge = (32000, 8000, 16000)

    # 満貫以上（切り上げ満貫あり: 40符4翻 / 70符3翻 以上は満貫）
    if han >= 13:
        return huge
    if han >= 5 or (han == 4 and fu >= 40) or (han == 3 and fu >= 70):
        for lim, val in big:
            if han <= lim:
                return val
        return huge
    key = (fu, han)
    if key not in table:
        raise ValueError(f'impossible combination of {fu} fu and {han} han')
    return table[key]


def _yakuman_point(is_oya: bool, count: int) -> tuple[int, int, int]:
    return (48000 * count, 16000 * count, 0) if is_oya else \
           (32000 * count, 8000 * count, 16000 * count)


def _total(pt: tuple[int, int, int], is_oya: bool, is_ron: bool) -> int:
    if is_ron:
        return pt[0]
    return pt[1] * 3 if is_oya else pt[1] * 2 + pt[2]


# --------------------------------------------------------------------------
# 面子分解
# --------------------------------------------------------------------------

def _divisions(counts: list[int], need: int):
    """`need` 面子 + 雀頭1 への分解を全通り yield する（(面子, 雀頭)）。"""
    def sets_only(c, i, acc):
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


@dataclass(frozen=True)
class HandValue:
    han: int
    fu: int
    yakuman: int
    point: int                       # 和了者が受け取る合計（供託・本場は含まない）
    yaku: dict = field(default_factory=dict)

    @property
    def has_yaku(self) -> bool:
        return self.yakuman > 0 or self.han > 0


def _chuuren(full: list[int], suit: int) -> bool:
    """九蓮宝燈（門前・その色のみ・1112345678999 + 1枚）。"""
    if sum(full) != 14:
        return False
    lo = suit * 9
    if sum(full[lo:lo + 9]) != 14:
        return False
    base = [3, 1, 1, 1, 1, 1, 1, 1, 3]
    extra = [full[lo + i] - base[i] for i in range(9)]
    return all(x >= 0 for x in extra) and sum(extra) == 1


def hand_value(concealed: tuple[int, ...], melds, winning_tile: int,
               bakaze: int, jikaze: int, *, is_ron: bool, is_oya: bool,
               dora_indicators=(), n_aka: int = 0,
               extra_han: int = 0) -> HandValue:
    """和了形の確定打点を返す。

    `concealed` は門前手牌の 34 カウント（**和了牌を含まない**）。
    `melds` は [('chi'|'pon'|'minkan'|'ankan', tile), ...]（chi は順子の先頭）。
    `extra_han` は立直・門前ツモなど**呼び出し側が確定と判断した**翻の追加分。
    一発・海底の類はここに入れないこと（`--validate` はそれらを母集団から外す）。
    """
    full = list(concealed)
    full[winning_tile] += 1

    chis = [t for k, t in melds if k == 'chi']
    pons = [t for k, t in melds if k == 'pon']
    minkans = [t for k, t in melds if k == 'minkan']
    ankans = [t for k, t in melds if k == 'ankan']
    is_menzen = not (chis or pons or minkans)

    # ドラ（表 + 赤）。裏ドラは含めない（expected_ura() で別に扱う）
    all_counts = list(full)
    for t in chis:
        for d in range(3):
            all_counts[t + d] += 1
    for t in pons:
        all_counts[t] += 3
    for t in minkans + ankans:
        all_counts[t] += 4      # 槓は4枚。3枚で数えるとドラが1枚落ちる
    n_dora = sum(all_counts[_next_tile(i)] for i in dora_indicators) + n_aka

    # 国士は特別扱い（他の役と組めない）
    if is_menzen and _kokushi_shanten(tuple(full)) == -1:
        pt = _yakuman_point(is_oya, 1)
        return HandValue(han=0, fu=0, yakuman=1, point=_total(pt, is_oya, is_ron),
                         yaku={'kokushi': 'yakuman'})

    need = 4 - len(melds)
    best: HandValue | None = None

    cand_divs = list(_divisions(list(full), need))
    # 七対子（門前・7対子ちょうど）
    chitoi = is_menzen and sum(1 for x in full if x == 2) == 7
    if chitoi:
        cand_divs.append(('CHITOI', None))

    for div in cand_divs:
        v = _eval_div(div, full, chis, pons, minkans, ankans, is_menzen,
                      winning_tile, bakaze, jikaze, is_ron, is_oya,
                      n_dora, extra_han)
        if v is None:
            continue
        if best is None or (v.yakuman, v.point) > (best.yakuman, best.point):
            best = v
    if best is None:
        return HandValue(han=0, fu=0, yakuman=0, point=0)
    return best


def _next_tile(t: int) -> int:
    """ドラ表示牌 -> ドラ（9→1 / 北→東 / 中→白）。"""
    if t < 27:
        base, r = t // 9 * 9, t % 9
        return base + (r + 1) % 9
    if t <= 30:
        return 27 + (t - 27 + 1) % 4
    return 31 + (t - 31 + 1) % 3


def _eval_div(div, full, chis, pons, minkans, ankans, is_menzen,
              win, bakaze, jikaze, is_ron, is_oya, n_dora, extra_han):
    """1つの分解について翻・符を出す（`agari.rs::search_yakus` の移植）。"""
    has_chitoi = div[0] == 'CHITOI'
    if has_chitoi:
        pairs = [i for i, c in enumerate(full) if c == 2]
        mz_shuntsu, mz_kotsu, pair = [], [], None
    else:
        acc, pair = div
        mz_shuntsu = [t for k, t in acc if k == 's']
        mz_kotsu = [t for k, t in acc if k == 'k']
        pairs = []

    yaku: dict[str, int] = {}
    han = 0
    yakuman = 0

    # 和了牌が明刻を作るか（`winning_tile_makes_minkou` の移植）
    wt_minkou = False
    if is_ron and not has_chitoi and win in mz_kotsu:
        if win >= 27:
            wt_minkou = True
        else:
            kind, num = win // 9, win % 9
            lo = kind * 9 + max(num - 2, 0)
            hi = kind * 9 + min(num, 6)
            wt_minkou = not any(lo <= s <= hi for s in mz_shuntsu)

    all_kotsu = mz_kotsu + pons + minkans + ankans
    all_shuntsu = mz_shuntsu + chis
    all_mentsu = all_kotsu + all_shuntsu

    has_pinfu = (
        len(mz_shuntsu) == 4 and not has_chitoi
        and pair not in DRAGONS and pair != bakaze and pair != jikaze
        and any((s % 9 + 1 <= 6 and s == win) or (s % 9 + 1 >= 2 and s + 2 == win)
                for s in mz_shuntsu)
    )
    if has_pinfu:
        han += 1; yaku['pinfu'] = 1
    if has_chitoi:
        han += 2; yaku['chiitoi'] = 2

    # 一盃口 / 二盃口（この分解の門前順子から）
    if not has_chitoi:
        dup = len(mz_shuntsu) - len(set(mz_shuntsu))
        if dup >= 2 and is_menzen:
            han += 3; yaku['ryanpeikou'] = 3
        elif dup == 1 and is_menzen:
            han += 1; yaku['ipeikou'] = 1

    if is_menzen and not has_chitoi and any(_chuuren(full, q) for q in range(3)):
        yakuman += 1; yaku['chuuren'] = 'yakuman'

    # 断幺九
    if has_chitoi:
        has_tanyao = all(not is_yaokyuu(t) for t in pairs)
    else:
        has_tanyao = (all(0 < s % 9 < 6 for s in all_shuntsu)
                      and all(not is_yaokyuu(k) for k in all_kotsu + [pair]))
    if has_tanyao:
        han += 1; yaku['tanyao'] = 1

    has_toitoi = not has_chitoi and not mz_shuntsu and not chis
    if has_toitoi:
        han += 2; yaku['toitoi'] = 2

    # 字一色 / 混一色 / 清一色
    tiles_for_isou = pairs if has_chitoi else all_mentsu + [pair]
    suits = {t // 9 for t in tiles_for_isou if t < 27}
    has_jihai = any(t >= 27 for t in tiles_for_isou)
    if not suits:
        yakuman += 1; yaku['tsuuiisou'] = 'yakuman'
    elif len(suits) == 1:
        n = (2 if has_jihai else 5) + (1 if is_menzen else 0)
        han += n; yaku['honitsu' if has_jihai else 'chinitsu'] = n

    if not has_chitoi:
        # 一気通貫
        ittsuu_mz = any(all(b in mz_shuntsu for b in (q * 9, q * 9 + 3, q * 9 + 6))
                        for q in range(3))
        if is_menzen and ittsuu_mz:
            han += 2; yaku['ittsuu'] = 2
        elif not chis and ittsuu_mz:
            han += 1; yaku['ittsuu'] = 1
        elif len(mz_shuntsu) + len(chis) >= 3:
            if any(all(b in all_shuntsu for b in (q * 9, q * 9 + 3, q * 9 + 6))
                   for q in range(3)):
                han += 1; yaku['ittsuu'] = 1

        # 三色同順 / 三色同刻
        s_num = {}
        for s in all_shuntsu:
            s_num.setdefault(s % 9, set()).add(s // 9)
        if any(len(v) == 3 for v in s_num.values()):
            n = 2 if is_menzen else 1
            han += n; yaku['sanshoku'] = n
        else:
            k_num = {}
            for k in all_kotsu:
                if k < 27:
                    k_num.setdefault(k % 9, set()).add(k // 9)
            if any(len(v) == 3 for v in k_num.values()):
                han += 2; yaku['sanshoku_doukou'] = 2

        n_ankou = len(ankans) + len(mz_kotsu) - (1 if wt_minkou else 0)
        if n_ankou == 4:
            yakuman += 1; yaku['suuankou'] = 'yakuman'
        elif n_ankou == 3:
            han += 2; yaku['sanankou'] = 2

        n_kan = len(ankans) + len(minkans)
        if n_kan == 4:
            yakuman += 1; yaku['suukantsu'] = 'yakuman'
        elif n_kan == 3:
            han += 2; yaku['sankantsu'] = 2

        if all(k in GREEN for k in all_kotsu + [pair]) and all(s == 19 for s in all_shuntsu):
            yakuman += 1; yaku['ryuuiisou'] = 'yakuman'

        if not has_tanyao:
            jihai_kotsu = {k for k in all_kotsu if k >= 27}
            if bakaze in jihai_kotsu:
                han += 1; yaku['yakuhai_bakaze'] = 1
            if jikaze in jihai_kotsu:
                han += 1; yaku['yakuhai_jikaze'] = 1
            n_dragon = sum(1 for d in DRAGONS if d in jihai_kotsu)
            if n_dragon:
                han += n_dragon; yaku['yakuhai_dragon'] = n_dragon
                if n_dragon == 3:
                    yakuman += 1; yaku['daisangen'] = 'yakuman'
                elif n_dragon == 2 and pair in DRAGONS:
                    han += 2; yaku['shousangen'] = 2
            n_wind = sum(1 for w in WINDS if w in jihai_kotsu)
            if n_wind == 4:
                yakuman += 1; yaku['daisuushii'] = 'yakuman'
            elif n_wind == 3 and pair in WINDS:
                yakuman += 1; yaku['shousuushii'] = 'yakuman'

    if not has_tanyao:
        check = pairs if has_chitoi else all_kotsu + [pair]
        if all(is_yaokyuu(k) for k in check):
            jihai = any(k >= 27 for k in check)
            if has_chitoi or has_toitoi:
                if jihai:
                    han += 2; yaku['honroutou'] = 2
                else:
                    yakuman += 1; yaku['chinroutou'] = 'yakuman'
            elif all(s % 9 in (0, 6) for s in all_shuntsu):
                n = (1 if jihai else 2) + (1 if is_menzen else 0)
                han += n; yaku['junchan' if not jihai else 'chanta'] = n

    if yakuman > 0:
        pt = _yakuman_point(is_oya, yakuman)
        return HandValue(han=0, fu=0, yakuman=yakuman,
                         point=_total(pt, is_oya, is_ron), yaku=yaku)
    if han == 0 and extra_han == 0:
        # 形の役が無く、立直・門前ツモ等も無いなら和了れない（ドラは役にならない）
        return None

    # `agari.rs` は符を計算しない条件が2つある。**点数には効かないが
    # `agari_detail.fu` と突き合わせるので忠実に移植する**:
    #   1. 形の役だけで 5翻以上（`search_yakus` の make_return!）
    #   2. 形の役が無く、追加翻 + ドラが 5翻以上（`agari()` の else 分岐）
    if han >= 5 or (han == 0 and extra_han + n_dora >= 5):
        fu = 0
    else:
        fu = _calc_fu(has_chitoi, mz_kotsu, mz_shuntsu, pair, pons, minkans, ankans,
                      wt_minkou, win, bakaze, jikaze, is_menzen, is_ron, has_pinfu)
    total_han = han + extra_han + n_dora
    pt = _point_calc(is_oya, fu, total_han)
    return HandValue(han=total_han, fu=fu, yakuman=0,
                     point=_total(pt, is_oya, is_ron), yaku=yaku)


def _calc_fu(has_chitoi, mz_kotsu, mz_shuntsu, pair, pons, minkans, ankans,
             wt_minkou, win, bakaze, jikaze, is_menzen, is_ron, has_pinfu) -> int:
    """`agari.rs::calc_fu` の忠実な移植。**一般論で直さないこと**。"""
    if has_chitoi:
        return 25
    fu = 20
    for t in mz_kotsu:
        minkou = wt_minkou and t == win
        y = is_yaokyuu(t)
        if not minkou and y:
            fu += 8
        elif (not minkou and not y) or (minkou and y):
            fu += 4
        else:
            fu += 2
    fu += sum(4 if is_yaokyuu(t) else 2 for t in pons)
    fu += sum(32 if is_yaokyuu(t) else 16 for t in ankans)
    fu += sum(16 if is_yaokyuu(t) else 8 for t in minkans)

    if pair in DRAGONS:
        fu += 2
    else:
        if pair == bakaze:
            fu += 2
        if pair == jikaze:      # 連風牌は 4符（天鳳ルール）
            fu += 2

    if fu == 20:
        if not is_menzen:
            return 30
        if has_pinfu:
            return 30 if is_ron else 20
        return 40 if is_ron else 30

    if not is_ron:
        fu += 2
    elif is_menzen:
        fu += 10

    if not wt_minkou:
        if pair == win:
            fu += 2
        else:
            if any(s + 1 == win or (s % 9 == 0 and s + 2 == win)
                   or (s % 9 == 6 and s == win) for s in mz_shuntsu):
                fu += 2
    return ((fu - 1) // 10 + 1) * 10


def expected_ura(concealed: tuple[int, ...], melds, unseen: tuple[int, ...],
                 n_indicators: int = 1) -> float:
    """裏ドラ枚数の**期待値**。

    裏ドラ表示牌はプレイヤーから見て未見牌の一様分布なので、
    `E = Σ_t (手にある t の枚数) × (t を指す表示牌の未見残枚数) / (未見総数)`
    で正しい期待値になる。

    **何に依存するかを取り違えないこと**（実測で確認した）:

      - 未見分布が一様なら**手の形には依存しない**。13枚なら `13 × 4/136 = 0.382` 枚で、
        順子3+対子2 も 刻子3+対子2 も同じ値になる。期待値は**枚数**で決まる
      - 形依存が出るのは **指示牌の未見枚数が偏るとき**。例えば 1m/1p/1s の刻子を
        持つ手で 9m9p9s が全部場に見えていると 0.382 → **0.129 枚**まで落ちる
      - **槓は4枚**なので寄与も4枚ぶん（実測: 10枚 0.294 → 暗槓1つ追加で 0.412）
      - なお**分散**は形に依存する（3枚持ちの牌が当たれば一度に3枚乗る）。
        打点の分布が要るなら期待値だけでは足りない — libriichi の SP 計算は
        `calc.rs:685` 以降で枚数別の指示牌数を数えて**分布**を出している

    libriichi の SP 計算は立直時の上乗せを「表ドラ表示牌が1枚のときだけ厳密、
    それ以外は +2翻の定数」で近似している（`algo/sp/calc.rs`）ので、
    この量そのものは観測に入っていない。

    `unseen` は未見牌の 34 カウント、`n_indicators` はカン込みの表示牌枚数。
    """
    total = sum(unseen)
    if total <= 0:
        return 0.0
    held = list(concealed)
    for k, t in melds:
        if k == 'chi':
            for d in range(3):
                held[t + d] += 1
        elif k == 'pon':
            held[t] += 3
        else:
            held[t] += 4        # 槓は4枚
    exp = 0.0
    for t in range(34):
        if not held[t]:
            continue
        prev = _prev_tile(t)
        exp += held[t] * unseen[prev] / total
    return exp * n_indicators


def _prev_tile(t: int) -> int:
    """その牌を指す表示牌（`_next_tile` の逆）。"""
    if t < 27:
        base, r = t // 9 * 9, t % 9
        return base + (r - 1) % 9
    if t <= 30:
        return 27 + (t - 27 - 1) % 4
    return 31 + (t - 31 - 1) % 3


# --------------------------------------------------------------------------
# 検証: 実際の和了を libriichi の agari_detail と突き合わせる
# --------------------------------------------------------------------------

def _validate(log_dirs: list[Path], limit: int) -> int:
    from libriichi.state import PlayerState
    from hand_replay import TRANSPARENT_EVENTS, tile_id

    WALL = 70
    checked = bad = skipped = 0
    fails = []
    for d in log_dirs:
        paths = sorted(d.glob('*.json.gz'))
        if limit:
            paths = paths[:limit]
        for path in paths:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                events = [json.loads(line) for line in f if line.strip()]
            states = [PlayerState(i) for i in range(4)]
            bakaze = oya = 0
            prev = None
            kan_draw = [False] * 4
            ntsumo = 0
            dora_ind: list[int] = []
            pending_hora: list[str] = []
            # ダブル立直の検出。agari_detail は additional_hans に is_w_riichi を
            # 含める（agent_helper.rs:497）が、この値は Python に露出していないので
            # 牌譜から再構成する: 自分の第一打牌での立直宣言、かつそれ以前に
            # 誰の鳴きも入っていないこと
            n_dahai = [0] * 4
            any_call = False
            w_riichi = [False] * 4
            # **AgariDetail.num_aka は暗槓内の赤を数えない**
            # （count_num_aka は akas_in_hand + fuuro_overview を見るが、暗槓は
            # ankan_overview に入るため）。翻の側は取得時に doras_owned へ
            # 加算済みなので、突き合わせにはこの差分だけを補正する
            ankan_aka = [0] * 4
            for ev in events:
                line = json.dumps(ev)
                t = ev.get('type')
                # hora は**適用前**の状態で評価する。適用後は局が終了しており
                # agari_detail が "cannot agari" で落ちる
                if t != 'hora':
                    for i in range(4):
                        states[i].update(line)
                if t == 'start_kyoku':
                    bakaze = tile_id(ev['bakaze'])
                    oya = ev['oya']
                    ntsumo = 0
                    kan_draw = [False] * 4
                    dora_ind = [tile_id(ev['dora_marker'])]
                    n_dahai = [0] * 4
                    any_call = False
                    w_riichi = [False] * 4
                    ankan_aka = [0] * 4
                elif t == 'dora':
                    dora_ind.append(tile_id(ev['dora_marker']))
                elif t == 'tsumo':
                    ntsumo += 1
                    kan_draw[ev['actor']] = prev in ('ankan', 'kakan', 'daiminkan')

                elif t == 'reach':
                    if n_dahai[ev['actor']] == 0 and not any_call:
                        w_riichi[ev['actor']] = True
                elif t == 'dahai':
                    n_dahai[ev['actor']] += 1
                elif t in ('chi', 'pon', 'daiminkan', 'ankan', 'kakan'):
                    any_call = True
                    if t == 'ankan':
                        ankan_aka[ev['actor']] += sum(
                            1 for c in ev.get('consumed', []) if c.endswith('r'))
                elif t == 'hora':
                    a = ev['actor']
                    st = states[a]
                    is_tsumo = ev.get('target') == a
                    detail = st.agari_detail(not is_tsumo, [])
                    # 一発・状況役は本モジュールの対象外なので母集団から外す
                    if detail.ippatsu or ntsumo >= WALL or prev == 'kakan' \
                            or (is_tsumo and kan_draw[a]):
                        skipped += 1
                        pending_hora.append(line)
                        prev = t
                        continue
                    conc = list(st.tehai)
                    if is_tsumo:
                        w = st.last_self_tsumo()
                        if w is None:
                            skipped += 1
                            pending_hora.append(line)
                            prev = t; continue
                        win = tile_id(w)
                        if conc[win] == 0:
                            skipped += 1
                            pending_hora.append(line)
                            prev = t; continue
                        conc[win] -= 1
                    else:
                        w = st.last_kawa_tile()
                        if w is None:
                            skipped += 1
                            pending_hora.append(line)
                            prev = t; continue
                        win = tile_id(w)
                    melds = ([('pon', b) for b in st.pons] + [('chi', b) for b in st.chis]
                             + [('minkan', b) for b in st.minkans]
                             + [('ankan', b) for b in st.ankans])
                    is_menzen = not (st.pons or st.chis or st.minkans)
                    # 立直 / 門前ツモ は確定として外から足す（一発は足さない）
                    extra = 0
                    if st.self_riichi_accepted:
                        extra += 1
                        if w_riichi[a]:
                            extra += 1      # ダブル立直
                    if is_tsumo and is_menzen:
                        extra += 1
                    n_aka = int(detail.num_aka) + ankan_aka[a]
                    jikaze = WINDS[(a - oya) % 4]
                    hv = hand_value(tuple(conc), melds, win, bakaze, jikaze,
                                    is_ron=not is_tsumo, is_oya=(a == oya),
                                    dora_indicators=tuple(dora_ind), n_aka=n_aka,
                                    extra_han=extra)
                    checked += 1
                    ok = (hv.point == detail.point
                          and (hv.yakuman > 0) == (detail.yakuman > 0)
                          and (hv.yakuman > 0 or (hv.han == detail.han and hv.fu == detail.fu)))
                    if not ok:
                        bad += 1
                        if len(fails) < 6:
                            fails.append((f'{path.parent.name}/{path.name}',
                                          hv, detail.point, detail.han, detail.fu,
                                          detail.yakuman, melds, win, conc,
                                          not is_tsumo, a == oya, extra, n_aka, dora_ind))
                if t == 'hora':
                    pending_hora.append(line)
                else:
                    # hora の連続が途切れたところでまとめて適用する。
                    # ダブロンでは2件目を先に適用すると局が終了して
                    # agari_detail が "cannot agari" で落ちるため
                    for h in pending_hora:
                        for i in range(4):
                            states[i].update(h)
                    pending_hora = []
                if t not in TRANSPARENT_EVENTS:
                    prev = t

    print(f'和了 {checked:,} 件を照合 / 一発・状況役で除外 {skipped:,} 件')
    for f in fails:
        print(f'  MISMATCH {f[0]}')
        print(f'    ours  point={f[1].point} han={f[1].han} fu={f[1].fu} '
              f'yakuman={f[1].yakuman} yaku={f[1].yaku}')
        print(f'    libri point={f[2]} han={f[3]} fu={f[4]} yakuman={f[5]}')
        print(f'    melds={f[6]} win={f[7]} is_ron={f[9]} is_oya={f[10]} '
              f'extra={f[11]} aka={f[12]} dora_ind={f[13]}')
        print(f'    concealed={tuple(i for i, c in enumerate(f[8]) for _ in range(c))}')
    if bad:
        print(f'FAIL: {bad:,}/{checked:,} 件で不一致 ({bad / max(checked, 1):.2%})')
        return 1
    print('PASS: 実際の和了はすべて libriichi の agari_detail と一致')
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
