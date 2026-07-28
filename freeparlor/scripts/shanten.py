#!/usr/bin/env python3
"""向聴数（シャンテン）計算 — 仮想手牌を評価するための共通部品（read-only）。

## なぜ自前で持つのか

`libriichi.state.PlayerState.shanten` は**実際に起きた盤面の向聴数**しか返さない
（`update` がイベント駆動なので、任意の手牌を問い合わせられない）。一方、
本プロジェクトが測りたい指標は**反実仮想の手牌**を必要とする:

  - 受け入れ枚数 = 「その牌を引いたら向聴が進むか」を 34 種すべてで試す
  - ポンテン見送り = 「その鳴きを取っていたらテンパイできたか」
  - 牌効率の誤り = 「別の牌を切っていたら受け入れが何枚多かったか」

いずれも「今そこに無い手牌」の向聴数が要る。よって計算器を Python 側に持つ。

## 正しさの担保（サイレントに間違えないための設計）

自前実装は**バグりうる**。したがって本モジュールは単独では信用せず、
`--validate` で牌譜を `PlayerState` で再生して検証してから使う:

    python shanten.py --validate <game_logs ディレクトリ> [--limit N]

検証は2本立て:

  1. **3n+1（打牌待ちの手牌）を libriichi と全件照合** — これが本モジュールの
     プリミティブであり、`PlayerState.shanten` が正。**1件でも食い違えば FAIL**
  2. **3n+2（ツモ後・鳴き後）の規約を自己整合で固定** — 本モジュールの 3n+2 の値は
     「最善の打牌をした後の向聴数」（= 打牌候補全部の最小値）と定義する。
     和了形のみ例外で −1

この検証を通したコミットでのみ、本モジュールを使う指標スクリプトを回してよい。

## libriichi との既知の規約差（**バグではない。合わせに行かないこと**）

  - **和了形**: 本モジュールは −1 を返す。libriichi の `PlayerState.shanten` は
    `calc_all(...).max(0)` で **0 にクランプ**する（`update.rs:876`。内部で
    `_shanten_discards` を計算するための都合）。和了とテンパイを区別したい本用途では
    −1 のほうが正しい
  - **3n+2 の値**: libriichi のテーブル実装は 3n+2 で本モジュールと異なる値を返す
    （実測 25% で +1 側にずれる）。libriichi 側のその値は内部利用専用であり、
    **受け入れ計算に必要なのは「打牌後の最小向聴」のほう**なので本モジュールの規約を採る。
    実測（3脚 全数、2026-07-29）: 3n+1 は **2,259,173/2,259,173 で完全一致**。
    最小値を与えた形の内訳は normal 2,067,273 / chiitoi 190,822 / kokushi 1,078 で、
    **3形とも実データ上で libriichi と一致している**（成分ごとの裏取り）

## 定義

向聴数 = テンパイまでの最小手数（テンパイ = 0、和了形 = -1）。

  - 通常手: `8 - 2*(面子) - (搭子+対子)`、ブロック数（面子+搭子）は 5 まで。
    5 ブロックで雀頭が無い場合は +1（頭無しの補正）
  - 七対子 / 国士無双は**副露が無い場合のみ**評価し、通常手との最小値を取る

副露（`n_open`）は完成面子として数える。暗槓・明槓も面子1つ（枚数は 3 として扱う
= 向聴計算上の扱いは面子と同じ）。

## 形ごとの分解 API（補助タスクのラベル用）

`shanten()` は3つの形の**最小値**しか返さない。`shanten_breakdown()` は
**通常手 / 七対子 / 国士をそれぞれ**返す:

    bd = shanten_breakdown(tehai34, n_open)
    bd.normal    # 常に定義される（副露あり手でも）
    bd.chiitoi   # 門前のみ。副露ありでは None
    bd.kokushi   # 門前のみ。副露ありでは None
    bd.combined  # = shanten() と同値
    bd.best_form # 'normal' | 'chiitoi' | 'kokushi'

### なぜ分解に意味があるのか（obs との重複監査、2026-07-29 一次ソース確認）

`teacherfree_training_candidates.md` §2b は「自手の事実は既に観測の入力にある」と
監査しており、**素の向聴数についてはそのとおり**である（`obs_repr.rs:395` が
`state.shanten` を 0–6 の one-hot で符号化している）。

しかし `state.shanten` は `update.rs:876` のとおり
**`shanten::calc_all(...).max(0)` = 3形の最小値**であり、
**分解は入力に入っていない**（`obs_repr.rs` に chitoi / kokushi の符号化は無い）。
つまりネットは「最小値がいくつか」だけを与えられ、
「どの形で最小なのか」「他の形なら何向聴か」は自力で推論する必要がある。

→ **分解ラベルは §2b の3層のうち「下（既に入力にある）」ではなく、
入力に無い情報**である。素の向聴数を予測させるより marginal value が高い。
ただし oracle 層（相手の待ち・危険牌）ほどではない — 分解は自手だけから
決定的に計算できる量なので、**中間の層**に位置づけるのが正確。

### 補助タスクのラベルとして使うときの注意

  - **副露ありでは chiitoi / kokushi は None**（成立しない形なので値が無い）。
    予測対象から外すマスクを立てること。0 埋めすると「副露手の七対子向聴は 0」
    という嘘を教えることになる
  - 値域: normal `-1..8` / chiitoi `-1..6` / kokushi `-1..13`。
    分類ヘッドにするならクリップ範囲を形ごとに変えること
  - **eval/配備経路に漏らさない**規律は補助タスク全般に掛かる
    （`teacherfree_training_candidates.md` §2f）
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# 34 種の並び: 0-8 = 萬子, 9-17 = 筒子, 18-26 = 索子, 27-33 = 字牌
SUIT_RANGES = ((0, 9), (9, 18), (18, 27), (27, 34))
TERMINALS_HONORS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


def _pareto(triples: set[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    """支配されている分解を落とす（組み合わせ爆発を抑えるためだけの最適化）。

    (sets, partials, has_pair) が全項目で他に劣る分解は最終的な min に寄与しない。
    """
    out = []
    for t in triples:
        if not any(
            o != t and o[0] >= t[0] and o[1] >= t[1] and o[2] >= t[2]
            for o in triples
        ):
            out.append(t)
    return tuple(sorted(out, reverse=True))


@lru_cache(maxsize=None)
def _suit_blocks(counts: tuple[int, ...], honor: bool) -> tuple[tuple[int, int, int], ...]:
    """1 スートの分解候補 (面子数, 搭子数, 対子を含むか) を列挙する。

    スート単位でメモ化するので、同じ形が何度出てきても計算は1回だけ。
    """
    n = len(counts)
    c = list(counts)
    out: set[tuple[int, int, int]] = set()

    def dfs(i: int, sets: int, partials: int, pair: int) -> None:
        while i < n and c[i] == 0:
            i += 1
        if i == n:
            out.add((sets, partials, pair))
            return
        # 刻子
        if c[i] >= 3:
            c[i] -= 3
            dfs(i, sets + 1, partials, pair)
            c[i] += 3
        # 順子
        if not honor and i + 2 < n and c[i + 1] and c[i + 2]:
            c[i] -= 1
            c[i + 1] -= 1
            c[i + 2] -= 1
            dfs(i, sets + 1, partials, pair)
            c[i] += 1
            c[i + 1] += 1
            c[i + 2] += 1
        # 対子
        if c[i] >= 2:
            c[i] -= 2
            dfs(i, sets, partials + 1, 1)
            c[i] += 2
        # 両面・辺張
        if not honor and i + 1 < n and c[i + 1]:
            c[i] -= 1
            c[i + 1] -= 1
            dfs(i, sets, partials + 1, pair)
            c[i] += 1
            c[i + 1] += 1
        # 嵌張
        if not honor and i + 2 < n and c[i + 2]:
            c[i] -= 1
            c[i + 2] -= 1
            dfs(i, sets, partials + 1, pair)
            c[i] += 1
            c[i + 2] += 1
        # 孤立牌として捨てる
        c[i] -= 1
        dfs(i, sets, partials, pair)
        c[i] += 1

    dfs(0, 0, 0, 0)
    return _pareto(out)


def _normal_shanten(counts: tuple[int, ...], n_open: int) -> int:
    profiles = [
        _suit_blocks(counts[lo:hi], lo == 27) for lo, hi in SUIT_RANGES
    ]
    best = 99
    for a in profiles[0]:
        for b in profiles[1]:
            for c_ in profiles[2]:
                for d in profiles[3]:
                    sets = n_open + a[0] + b[0] + c_[0] + d[0]
                    partials = a[1] + b[1] + c_[1] + d[1]
                    pair = a[2] or b[2] or c_[2] or d[2]
                    if sets > 4:  # 副露込みで5面子はあり得ない（和了形は4面子1雀頭）
                        sets = 4
                    if sets + partials > 5:
                        partials = 5 - sets
                    sh = 8 - 2 * sets - partials
                    if sets + partials == 5 and not pair:
                        sh += 1
                    if sh < best:
                        best = sh
    return best


def _chiitoi_shanten(counts: tuple[int, ...]) -> int:
    pairs = sum(1 for x in counts if x >= 2)
    kinds = sum(1 for x in counts if x >= 1)
    sh = 6 - pairs
    if kinds < 7:
        sh += 7 - kinds
    return sh


def _kokushi_shanten(counts: tuple[int, ...]) -> int:
    kinds = sum(1 for i in TERMINALS_HONORS if counts[i] >= 1)
    has_pair = any(counts[i] >= 2 for i in TERMINALS_HONORS)
    return 13 - kinds - (1 if has_pair else 0)


def _check(counts: tuple[int, ...]) -> None:
    if len(counts) != 34:
        raise ValueError(f'counts must be length 34, got {len(counts)}')


@lru_cache(maxsize=1 << 20)
def shanten_normal(counts: tuple[int, ...], n_open: int = 0) -> int:
    """**通常手（メンツ手）だけ**の向聴数。副露の有無によらず常に定義される。"""
    _check(counts)
    return _normal_shanten(counts, n_open)


@lru_cache(maxsize=1 << 18)
def shanten_chiitoi(counts: tuple[int, ...], n_open: int = 0) -> int | None:
    """**七対子だけ**の向聴数。副露があると成立しないので `None` を返す。

    `None` は「値が無い」であって 0 ではない。補助タスクのラベルにするときは
    マスクを立てること（0 埋めは嘘を教えることになる）。
    """
    _check(counts)
    if n_open:
        return None
    return _chiitoi_shanten(counts)


@lru_cache(maxsize=1 << 18)
def shanten_kokushi(counts: tuple[int, ...], n_open: int = 0) -> int | None:
    """**国士無双だけ**の向聴数。副露があると成立しないので `None` を返す。"""
    _check(counts)
    if n_open:
        return None
    return _kokushi_shanten(counts)


@dataclass(frozen=True)
class ShantenBreakdown:
    """形ごとの向聴数と、その最小値。

    `chiitoi` / `kokushi` は副露ありで `None`（成立しない形）。
    `combined` は `shanten()` と同値であることを不変条件とする。
    """
    normal: int
    chiitoi: int | None
    kokushi: int | None
    combined: int
    best_form: str

    def as_label_dict(self) -> dict[str, object]:
        """補助タスクのラベル用に、値とマスクを分けて返す。"""
        return {
            'normal': self.normal,
            'chiitoi': self.chiitoi if self.chiitoi is not None else 0,
            'kokushi': self.kokushi if self.kokushi is not None else 0,
            'chiitoi_valid': self.chiitoi is not None,
            'kokushi_valid': self.kokushi is not None,
            'combined': self.combined,
            'best_form': self.best_form,
        }


def shanten_breakdown(counts: tuple[int, ...], n_open: int = 0) -> ShantenBreakdown:
    """通常手 / 七対子 / 国士を**それぞれ**計算して返す。

    `obs` v4 は3形の**最小値しか**符号化していない（`obs_repr.rs:395` の
    `state.shanten` = `calc_all(...).max(0)`）ので、この分解は観測に無い情報。
    詳細はモジュール docstring の「形ごとの分解 API」節。
    """
    n = shanten_normal(counts, n_open)
    c = shanten_chiitoi(counts, n_open)
    k = shanten_kokushi(counts, n_open)
    best, form = n, 'normal'
    if c is not None and c < best:
        best, form = c, 'chiitoi'
    if k is not None and k < best:
        best, form = k, 'kokushi'
    return ShantenBreakdown(normal=n, chiitoi=c, kokushi=k,
                            combined=best, best_form=form)


@lru_cache(maxsize=1 << 20)
def shanten(counts: tuple[int, ...], n_open: int = 0) -> int:
    """手牌 34 種カウント + 副露数から向聴数を返す（テンパイ 0、和了 -1）。

    3形の最小値。**形ごとの内訳が要るなら `shanten_breakdown()`**。
    `counts` は**門前手牌のみ**（副露で晒した牌は含めない）。tuple で渡すこと
    （メモ化のため）。
    """
    _check(counts)
    best = _normal_shanten(counts, n_open)
    if n_open == 0:
        best = min(best, _chiitoi_shanten(counts), _kokushi_shanten(counts))
    return best


def ukeire(counts: tuple[int, ...], n_open: int, seen: tuple[int, ...]) -> tuple[int, int]:
    """受け入れを返す: (有効牌の種類数, 残り枚数の合計)。

    `seen` は場に見えている枚数（自分の手牌・河・副露・ドラ表示）の 34 カウント。
    残り枚数 = 4 - seen[t] で、0 以下の牌は受け入れに数えない
    （「山に無い牌の受け入れ」を数えると指標が嘘になる）。
    """
    base = shanten(counts, n_open)
    kinds = 0
    tiles = 0
    lst = list(counts)
    for t in range(34):
        remaining = 4 - seen[t]
        if remaining <= 0 or lst[t] >= 4:
            continue
        lst[t] += 1
        adv = shanten(tuple(lst), n_open) < base
        lst[t] -= 1
        if adv:
            kinds += 1
            tiles += remaining
    return kinds, tiles


# --------------------------------------------------------------------------
# 検証: libriichi の実測値と全決定点で突き合わせる
# --------------------------------------------------------------------------

def _validate(log_dir: Path, limit: int) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / 'mortal'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from libriichi.state import PlayerState  # noqa: E402

    from analyze_freeparlor_pnl_1v3 import seat_from_filename  # noqa: E402

    paths = sorted(log_dir.glob('*.json.gz'))
    if not paths:
        raise RuntimeError(f'no logs in {log_dir}')
    if limit:
        paths = paths[:limit]

    n_31 = n_31_bad = 0        # 3n+1: libriichi と全件一致すべき
    n_32 = n_32_bad = 0        # 3n+2: 自己整合（打牌後の最小 = その値）
    n_32_agari = 0             # 3n+2 のうち和了形（−1 規約で自己整合の例外）
    n_32_libdiff = 0           # 3n+2 で libriichi と値が違った件数（既知の規約差）
    n_predeal = 0              # 配牌前（start_game 直後）
    by_form: dict[str, int] = {}   # argmin がどの形だったかの内訳（成分の裏取り）
    for path in paths:
        seat = seat_from_filename(path)
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            events = [json.loads(line) for line in f if line.strip()]
        states = [PlayerState(i) for i in range(4)]
        for ev in events:
            line = json.dumps(ev)
            for i in range(4):
                states[i].update(line)
            st = states[seat]
            tehai = tuple(st.tehai)
            n_open = len(st.pons) + len(st.chis) + len(st.minkans) + len(st.ankans)
            total = sum(tehai)
            if total == 0:
                n_predeal += 1
                continue
            # 有効な状態は「打牌待ち(3n+1)」か「ツモ/鳴き直後(3n+2)」のみ。
            # それ以外は再生の破れなので大声で落とす（サイレントスキップ禁止）。
            if total == 13 - 3 * n_open:
                n_31 += 1
                mine = shanten(tehai, n_open)
                # 形ごとの分解も同時に検証する。最小値を与えた形（argmin）については
                # libriichi の値がその形の値と一致するはずなので、**どの形が
                # 何件検証されたか**を数える（成分ごとの裏取りになる）。
                bd = shanten_breakdown(tehai, n_open)
                if bd.combined != mine:
                    raise RuntimeError(
                        f'{path}: breakdown.combined {bd.combined} != shanten {mine}'
                    )
                if mine == st.shanten:
                    by_form[bd.best_form] = by_form.get(bd.best_form, 0) + 1
                if mine != st.shanten:
                    n_31_bad += 1
                    if n_31_bad <= 5:
                        print(f'MISMATCH(3n+1) {path.name}: ours={mine} '
                              f'libriichi={st.shanten} melds={n_open} tehai={tehai}')
            elif total == 14 - 3 * n_open:
                n_32 += 1
                mine = shanten(tehai, n_open)
                if mine != st.shanten:
                    n_32_libdiff += 1
                if mine == -1:
                    n_32_agari += 1
                else:
                    lst = list(tehai)
                    best = 99
                    for t in range(34):
                        if lst[t]:
                            lst[t] -= 1
                            best = min(best, shanten(tuple(lst), n_open))
                            lst[t] += 1
                    if mine != best:
                        n_32_bad += 1
                        if n_32_bad <= 5:
                            print(f'CONVENTION BREAK(3n+2) {path.name}: value={mine} '
                                  f'min_over_discards={best} melds={n_open} tehai={tehai}')
            else:
                raise RuntimeError(
                    f'{path}: concealed={total} with melds={n_open} is neither '
                    f'3n+1 ({13 - 3 * n_open}) nor 3n+2 ({14 - 3 * n_open})'
                )

    print(f'{len(paths)} hanchan / pre-deal states skipped: {n_predeal:,}')
    print(f'[1] 3n+1 vs libriichi : {n_31 - n_31_bad:,}/{n_31:,} agree')
    forms = ' / '.join(f'{k}={v:,}' for k, v in sorted(by_form.items()))
    print(f'    うち最小値を与えた形の内訳（成分ごとの裏取り）: {forms or "なし"}')
    print(f'[2] 3n+2 self-consistency: {n_32 - n_32_agari - n_32_bad:,}/'
          f'{n_32 - n_32_agari:,} agree (agari hands excluded: {n_32_agari:,})')
    print(f'    INFO: 3n+2 differs from libriichi in {n_32_libdiff:,}/{n_32:,} '
          f'({n_32_libdiff / max(n_32, 1):.1%}) — known convention gap, see module docstring')
    print(f'    cache: suit_blocks={_suit_blocks.cache_info().currsize:,} '
          f'shanten={shanten.cache_info().currsize:,}')
    if n_31_bad or n_32_bad:
        print(f'FAIL: 3n+1 mismatches={n_31_bad:,} / 3n+2 convention breaks={n_32_bad:,}')
        return 1
    print('PASS')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validate', metavar='GAME_LOG_DIR',
                    help='牌譜ディレクトリを再生し libriichi の shanten と全件照合する')
    ap.add_argument('--limit', type=int, default=0, help='照合する半荘数の上限（0=全部）')
    args = ap.parse_args()
    if not args.validate:
        ap.error('--validate is required (this module is a library otherwise)')
    return _validate(Path(args.validate), args.limit)


if __name__ == '__main__':
    sys.exit(main())
