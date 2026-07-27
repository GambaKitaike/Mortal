#!/usr/bin/env python3
"""現物（安全牌）に対する打牌規律の測定 — 「降りの過程」の指標化（read-only）。

動機: `anchor_arm_c_result.md` §3-2 で、放銃増（+3.60pp、z=+6.57）の主因が
**門前・非立直の局面**（+3.06pp、z=+7.04）だと分かった。立直後・副露後の放銃「率」は
不変であり、壊れているのは「まだ何も晒していない状態での押し引き」である。
放銃率は**結果**指標なので、降り方の失敗そのものは捉えられない。本書はその過程を測る。

Gamba 定性レビュー（2026-07-26/27）の該当所見:
  「2件立直に両者への現物2sがあるのにそれを切らない！ベタオリも失敗しとるやんけ」
  「途中まで現物を切っていたのに突然危険牌を切って放銃するパターンを頻繁に目視」

---

## 現物の定義（本書の正。将来この量を補助タスクの予測対象にする場合もこの定義を使う）

立直者 X に対する現物 genbutsu(X) は、判断時点で以下の和集合:

  (a) **X の河にある牌種** — X はその牌でロンできない（フリテン）
  (b) **X の立直宣言後に場に打たれ、X がロンしなかった牌種** — 通過牌。
      X が見逃せば以後フリテンになるため恒久的に安全

牌種は赤ドラを区別しない（`5mr` は `5m` と同一視）。安全性は牌種の問題であり
赤の有無は無関係。

**本定義に含めない**（意図的な限界。過小評価側＝保守的に働く）:
  - スジ・壁・ワンチャンス等の**推定**安全牌。現物は「確実に当たらない」集合のみ
  - 同巡内フリテン（立直前の見逃し）。恒久的でないため
  - 立直していない他家に対する安全性。本書は立直者のみを対象にする

複数の立直者がいる場合、`safe_all` = 全立直者の現物の**積集合**（全員に通る牌）。

## 測定する量

決定点 = 「立直者が1人以上いる状態での、自分（challenger）の打牌」。

  1. **危険牌選択率** = 非 safe_all を切った打牌 / 全打牌
  2. **押し込み率（現物保持時）** = 現物を持ちながら非現物を切った / 現物を持っていた打牌
     ← 「安全な選択肢があったのに危険牌を選んだ」割合。押しは正当な戦術でもあるので
       この値が高いこと自体は劣化の証明にならない（下記3と併せて読む）
  3. **降りの中断率** = 直前の自分の打牌が現物だった局面のうち、
     現物を持ちながら非現物を切った割合
     ← **一度降りを選んだのに押し返した**割合。Gamba 所見「途中まで現物を切っていたのに
       突然危険牌」を直接測る。押し引きの一貫性の指標であり、3 が高いのは
       「方針がぶれている」ことを意味する（純粋な押し戦術なら最初から押す）

SE は半荘クラスタの ratio-estimator（`analyze_fundamentals_1v3.py` と同じ）。
差の SE は独立2標本近似。

## 手牌の再構成

`start_kyoku.tehais[seat]` を起点に tsumo/dahai/pon/chi/ankan/kakan/daiminkan を適用する。
副露で晒した牌は手牌から除く（切れないため）。整合が崩れたら**大声で落とす**
（サイレントフォールバック禁止）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_freeparlor_pnl_1v3 import seat_from_filename  # noqa: E402


def norm(tile: str) -> str:
    """赤ドラを通常牌に正規化する（安全性は牌種の問題で赤の有無は無関係）。"""
    return tile[:-1] if tile.endswith('r') else tile


def analyze_log(path: Path) -> dict:
    seat = seat_from_filename(path)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        events = [json.loads(line) for line in f if line.strip()]

    n_dec = 0          # 立直者ありでの自分の打牌
    n_danger = 0       # うち非現物を切った
    n_had_safe = 0     # うち現物を手に持っていた
    n_push_w_safe = 0  # うち現物を持ちながら非現物を切った
    n_after_fold = 0   # 直前の自分の打牌が現物だった局面
    n_fold_broken = 0  # うち現物を持ちながら非現物を切った

    hand: list[str] = []
    river: dict[int, set[str]] = {}
    riichi: set[int] = set()
    passed: dict[int, set[str]] = {}
    prev_was_safe = False

    for ev in events:
        t = ev.get('type')

        if t == 'start_kyoku':
            hand = list(ev['tehais'][seat])
            river = {i: set() for i in range(4)}
            passed = {i: set() for i in range(4)}
            riichi = set()
            prev_was_safe = False
            continue

        if t == 'tsumo':
            if ev['actor'] == seat:
                hand.append(ev['pai'])
            continue

        if t == 'reach_accepted':
            riichi.add(ev['actor'])
            continue

        if t in ('pon', 'chi', 'daiminkan') and ev['actor'] == seat:
            for c in ev['consumed']:
                if c not in hand:
                    raise RuntimeError(f'{path}: consumed {c} not in hand {hand}')
                hand.remove(c)
            continue

        if t == 'ankan' and ev['actor'] == seat:
            for c in ev['consumed']:
                if c not in hand:
                    raise RuntimeError(f'{path}: ankan {c} not in hand {hand}')
                hand.remove(c)
            continue

        if t == 'kakan' and ev['actor'] == seat:
            if ev['pai'] not in hand:
                raise RuntimeError(f'{path}: kakan {ev["pai"]} not in hand {hand}')
            hand.remove(ev['pai'])
            continue

        if t != 'dahai':
            continue

        actor = ev['actor']
        pai = ev['pai']

        if actor == seat:
            others_riichi = riichi - {seat}
            if others_riichi:
                safe_all = set.intersection(
                    *[river[x] | passed[x] for x in others_riichi]
                )
                if pai not in hand:
                    raise RuntimeError(f'{path}: dahai {pai} not in hand {hand}')
                held = {norm(x) for x in hand}
                had_safe = bool(held & safe_all)
                chose_safe = norm(pai) in safe_all

                n_dec += 1
                if not chose_safe:
                    n_danger += 1
                if had_safe:
                    n_had_safe += 1
                    if not chose_safe:
                        n_push_w_safe += 1
                if prev_was_safe and had_safe:
                    n_after_fold += 1
                    if not chose_safe:
                        n_fold_broken += 1
                prev_was_safe = chose_safe
            else:
                prev_was_safe = False
            hand.remove(pai)

        # 河への追加と、立直者に対する通過牌の記録
        river[actor].add(norm(pai))
        for x in riichi:
            if x != actor:
                passed[x].add(norm(pai))

    return {
        'n_dec': n_dec,
        'n_danger': n_danger,
        'n_had_safe': n_had_safe,
        'n_push_w_safe': n_push_w_safe,
        'n_after_fold': n_after_fold,
        'n_fold_broken': n_fold_broken,
    }


def ratio_and_se(rows, num, den):
    tot_d = sum(r[den] for r in rows)
    if tot_d == 0:
        return None, None
    tot_n = sum(r[num] for r in rows)
    rr = tot_n / tot_d
    var = sum((r[num] - rr * r[den]) ** 2 for r in rows) / tot_d**2
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
        tot = sum(r['n_dec'] for r in legs[name])
        print(f'[{name}] {len(paths)} hanchan / 立直者ありでの打牌 {tot:,} 回')

    print(f"\n{'metric':<34}{'init':>11}{args.label:>13}{'diff':>10}{'SE':>8}{'z':>8}")
    specs = [
        ('危険牌選択率',              'n_danger',      'n_dec'),
        ('押し込み率(現物保持時)',    'n_push_w_safe', 'n_had_safe'),
        ('降りの中断率',              'n_fold_broken', 'n_after_fold'),
        ('(参考)現物保持率',          'n_had_safe',    'n_dec'),
    ]
    for lab, num, den in specs:
        a, sa = ratio_and_se(legs['init'], num, den)
        b, sb = ratio_and_se(legs[args.label], num, den)
        if a is None or b is None:
            print(f'{lab:<34}{"(分母0)":>11}')
            continue
        diff = (b - a) * 100
        se = math.sqrt(sa**2 + sb**2) * 100
        print(f'{lab:<34}{a*100:>10.2f}%{b*100:>12.2f}%{diff:>+10.2f}{se:>8.2f}{diff/se:>+8.2f}')

    print('\n注: 現物 = 立直者の河 ∪ 立直後の通過牌（赤は同一視）。スジ・壁等の推定安全牌は'
          '含めない（保守的）。押しは正当な戦術なので「押し込み率」単独では劣化を示さない — '
          '「降りの中断率」と併せて読むこと。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
