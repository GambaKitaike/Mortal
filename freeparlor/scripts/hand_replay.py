#!/usr/bin/env python3
"""牌譜再生の共通部品 — challenger 席の決定点を状態つきで列挙する（read-only）。

`analyze_call_quality.py` / `analyze_tile_efficiency.py` /
`analyze_policy_consistency.py` が共通で必要とする「1半荘を再生して、
challenger の各決定点で手牌・副露・合法手・場に見えている牌を取り出す」処理を
1箇所に置く（3本に複製するとドリフトするため）。

合法手判定（ポン/チー/カンが可能か）は**自前で書かず `PlayerState.last_cans` を使う**
— libriichi 自身の判定であり、再実装すると本家挙動との乖離が入る。

## 「見えている牌」(`seen`) の定義

受け入れ計算で「山に残っている枚数」を出すために使う。judge 時点で観測できる:

  - 自分の門前手牌
  - 全員の副露（晒された牌）
  - 全員の河（打牌）
  - ドラ表示牌（裏は見えないので含めない）

含めない: 他家の手牌（見えない）、王牌（嶺上・裏ドラ）。
したがって `4 - seen[t]` は**自分視点の残り枚数の上界**であり、真の山残枚数ではない。
受け入れ「枚数」はこの上界で数える（麻雀の標準的な数え方と同じ）。
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

TILE_ORDER = [f'{n}{s}' for s in 'mps' for n in range(1, 10)] + \
             ['E', 'S', 'W', 'N', 'P', 'F', 'C']
TILE_INDEX = {t: i for i, t in enumerate(TILE_ORDER)}

# 決定と、その決定への応答の**間に割り込みうる**イベント。
# 応答を「次のイベント」で判定する解析は、これらを透過させないと取りこぼす。
#   - dahai(立直宣言牌) -> reach_accepted -> pon   （鳴きが1つ後ろにずれる）
#   - tsumo -> reach -> dahai                      （立直宣言時の打牌がずれる）
#   - ankan -> dora -> tsumo                       （新ドラ表示）
# 実例: 10000_8192_b.json.gz の pon が reach_accepted に隠れて未対応になっていた。
TRANSPARENT_EVENTS = frozenset({'reach', 'reach_accepted', 'dora'})


def tile_id(pai: str) -> int:
    """mjai の牌文字列を 0-33 の索引へ（赤ドラは通常牌に畳む）。"""
    if pai.endswith('r'):
        pai = pai[:-1]
    return TILE_INDEX[pai]


@dataclass
class Decision:
    """challenger の1決定点のスナップショット。"""
    kyoku_id: tuple[int, int]      # (kyoku, honba)
    turn: int                      # PlayerState.at_turn
    tehai: tuple[int, ...]         # 門前手牌の 34 カウント（副露分は含まない）
    n_open: int                    # 副露数（暗槓・明槓を含む）
    shanten: int                   # 本モジュールの規約による向聴（和了 = -1）
    seen: tuple[int, ...]          # 見えている枚数の 34 カウント
    cans: object                   # PlayerState.last_cans（合法手）
    self_riichi: bool
    others_riichi: bool
    event: dict                    # この決定点を生んだイベント
    target_tile: int | None        # 鳴き対象牌（鳴き機会でないときは None）
    melds: tuple                   # 既存の副露 [('k'|'s'|'kan'|'ankan', tile), ...]
    bakaze: int                    # 場風の牌 index
    jikaze: int                    # 自風の牌 index
    n_aka: int                     # 手中の赤ドラ枚数（チップ経済の主因なので分離して持つ）


@dataclass
class _Board:
    seen: list[int] = field(default_factory=lambda: [0] * 34)
    riichi: set[int] = field(default_factory=set)
    kyoku: int = 0
    honba: int = 0
    bakaze: int = 27
    jikaze: int = 27


def replay(path: Path, seat: int, *, shanten_fn=None):
    """1半荘を再生し、**全イベント**を `(event, decision|None)` として yield する。

    `decision` は challenger が行動できる局面（`last_cans.can_act`）でのみ非 None。
    イベントも一緒に流すのは、**決定への応答が次のイベントに現れる**ため
    （打牌は自分の手番の dahai、鳴きは pon/chi/daiminkan）。決定点だけを流すと、
    自分の打牌イベントは `can_act` が立たず観測できない。

    `shanten_fn` を渡すと各決定点の向聴を計算する（不要な解析では None にして
    計算コストを払わない）。
    """
    from libriichi.state import PlayerState

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        events = [json.loads(line) for line in f if line.strip()]

    states = [PlayerState(i) for i in range(4)]
    board = _Board()

    for ev in events:
        line = json.dumps(ev)
        for i in range(4):
            states[i].update(line)
        t = ev.get('type')

        if t == 'start_kyoku':
            # 自風 = 親からの相対位置（E S W N の順）
            winds = (27, 28, 29, 30)
            board = _Board(
                kyoku=ev['kyoku'], honba=ev['honba'],
                bakaze=tile_id(ev['bakaze']),
                jikaze=winds[(seat - ev['oya']) % 4],
            )
            for pai in ev['tehais'][seat]:
                board.seen[tile_id(pai)] += 1
            board.seen[tile_id(ev['dora_marker'])] += 1
            continue

        if t == 'dora':
            board.seen[tile_id(ev['dora_marker'])] += 1
        elif t == 'tsumo' and ev['actor'] == seat:
            board.seen[tile_id(ev['pai'])] += 1
        elif t == 'reach_accepted':
            board.riichi.add(ev['actor'])

        st = states[seat]
        cans = st.last_cans
        # 鳴き対象牌: 他家の打牌に反応できる状態でのみ意味を持つ
        target = None
        if cans.can_pon or cans.can_chi or cans.can_daiminkan or cans.can_ron_agari:
            last = st.last_kawa_tile()  # メソッド（Option<String> を返す）
            if last is not None:
                target = tile_id(last)

        dec = None
        if cans.can_act:
            tehai = tuple(st.tehai)
            n_open = len(st.pons) + len(st.chis) + len(st.minkans) + len(st.ankans)
            dec = Decision(
                kyoku_id=(board.kyoku, board.honba),
                turn=st.at_turn,
                tehai=tehai,
                n_open=n_open,
                shanten=shanten_fn(tehai, n_open) if shanten_fn else 0,
                seen=tuple(board.seen),
                cans=cans,
                self_riichi=st.self_riichi_accepted,
                others_riichi=bool(board.riichi - {seat}),
                event=ev,
                target_tile=target,
                melds=tuple([('k', b) for b in st.pons] + [('s', b) for b in st.chis]
                            + [('kan', b) for b in st.minkans]
                            + [('ankan', b) for b in st.ankans]),
                bakaze=board.bakaze,
                jikaze=board.jikaze,
                n_aka=sum(1 for a in st.akas_in_hand if a),
            )
        yield ev, dec

        # 打牌・副露で「見えた」牌を後追いで加算する。自分の手牌由来の牌は
        # start_kyoku / tsumo で既に数えているので二重計上しないよう他家分のみ。
        if t == 'dahai' and ev['actor'] != seat:
            board.seen[tile_id(ev['pai'])] += 1
        elif t in ('pon', 'chi', 'daiminkan', 'ankan', 'kakan') and ev['actor'] != seat:
            for c in ev.get('consumed', []):
                board.seen[tile_id(c)] += 1
            if t == 'kakan':
                board.seen[tile_id(ev['pai'])] += 1


def call_variants(tehai: tuple[int, ...], target: int, cans) -> list[tuple[str, tuple[int, ...], tuple]]:
    """鳴きを取った場合の門前手牌を、鳴きの種類ごとに返す。

    返り値は (種別, 消費後の手牌 34 カウント, 面子) のリスト。副露数は呼び出し側で +1 する。
    面子は `yaku.open_hand_yaku` と同じ表現で ('k', 牌) / ('kan', 牌) / ('s', 順子の先頭)。
    **チーの面子は対象牌ではなく順子の先頭**なので、low/mid/high で先頭が変わる。
    チーは `last_cans` が許した形だけを出す。
    """
    out: list[tuple[str, tuple[int, ...], tuple]] = []
    h = list(tehai)

    def take(idxs: list[int], label: str, meld: tuple) -> None:
        for i in idxs:
            if h[i] <= 0:
                return
        for i in idxs:
            h[i] -= 1
        out.append((label, tuple(h), meld))
        for i in idxs:
            h[i] += 1

    if cans.can_pon:
        take([target, target], 'pon', ('k', target))
    if cans.can_daiminkan:
        take([target, target, target], 'daiminkan', ('kan', target))
    if target < 27:  # チーは数牌のみ
        rank = target % 9
        if cans.can_chi_low and rank <= 6:
            take([target + 1, target + 2], 'chi', ('s', target))
        if cans.can_chi_mid and 1 <= rank <= 7:
            take([target - 1, target + 1], 'chi', ('s', target - 1))
        if cans.can_chi_high and rank >= 2:
            take([target - 2, target - 1], 'chi', ('s', target - 2))
    return out
