#!/usr/bin/env python3
"""クラスタ単位の取り違えによる SE の過小/過大評価を測る（read-only・診断）。

## 動機

eval ハーネスは **1 seed = 1つの山を4回打たせ、challenger の座席だけ 0→1→2→3 と
回す**（`eval_grp_baseline_1v3.py:171-179`、実データでも配牌完全一致を確認）。
つまり `n=800 半荘` の実体は **200 seed × 4 split** であり、
**独立な乱数は 200 個しかない**可能性がある。

しかし既存の集計（`analyze_fundamentals_1v3.py` ほか）は
**半荘を1クラスタ**として cluster-robust SE を出している。もし同一 seed の
4 split が相関していれば、この SE は誤っている:

  - 正の相関 → 現行 SE は**過小評価**（z が過大に出る）
  - 負の相関 → 現行 SE は**過大評価**（座席ローテーションは分散削減の設計なので
    こちらもありうる。同じ配牌を4席から打てば「良い席を引いた運」が相殺される）

本スクリプトは同じ量を **半荘クラスタ / seed クラスタ**の2通りで計算して比べる。
design effect = (SE_seed / SE_hanchan)² が 1 より大きければ現行は過小評価。

## もう1つの近似 — 両脚の対応

差の SE は現行 `SE(差)² = SE₁² + SE₂²`（独立2標本近似）。しかし init 脚と
ckpt 脚は**同一 seed 集合**を使うので対応がある。対応を使った差の SE も併記する
（`analyze_fundamentals_1v3.py` の docstring は「正の相関 → 過大（保守的）」と
述べているが、実測値は置かれていない）。

**本スクリプトは判定を変更しない。** 判定条件は
`anchored_ppo_design.md` §6 が「`analyze_fundamentals_1v3.py` による」と凍結して
いるので、その実装の出力が正である。ここで測るのは**解釈に添える留保**の大きさ。
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / 'mortal'))

from analyze_fundamentals_1v3 import scan_hanchan  # noqa: E402
from analyze_freeparlor_pnl_1v3 import process_hanchan  # noqa: E402


def seed_of(path: Path) -> str:
    """ファイル名 `<seed>_<key>_<split>.json.gz` から seed を取る。"""
    return path.name.split('_')[0]


def collect(d: Path) -> list[dict]:
    rows = []
    for p in sorted(d.glob('*.json.gz')):
        r = scan_hanchan(p)
        pnl = process_hanchan(p)
        r['seed'] = seed_of(p)
        r['split'] = p.name.split('_')[2].split('.')[0]
        r['chip'] = float(pnl.chip_total)
        r['one'] = 1
        rows.append(r)
    return rows


def by_seed(rows: list[dict], keys: list[str]) -> list[dict]:
    """同一 seed の 4 split を1クラスタへ畳む。"""
    agg: dict[str, dict] = defaultdict(lambda: {k: 0.0 for k in keys})
    for r in rows:
        for k in keys:
            agg[r['seed']][k] += r[k]
    return list(agg.values())


def ratio_se(rows: list[dict], num: str, den: str) -> tuple[float, float]:
    tot_d = sum(r[den] for r in rows)
    rr = sum(r[num] for r in rows) / tot_d
    var = sum((r[num] - rr * r[den]) ** 2 for r in rows) / tot_d**2
    return rr, math.sqrt(var)


def paired_diff_se(a: list[dict], b: list[dict], num: str, den: str,
                   cluster_seed: bool) -> tuple[float, float]:
    """対応のある差の SE。クラスタ内で (num - r·den) の**差**の分散を取る。

    比率の差なので、各脚の pooled 比率 r_a, r_b を使って
    d_c = (num_b - r_b·den_b)/D_b − (num_a - r_a·den_a)/D_a を積む
    （linearization。両脚の同一 seed を対応付ける）。
    """
    def index(rows):
        if cluster_seed:
            m = defaultdict(lambda: {num: 0.0, den: 0.0})
            for r in rows:
                m[r['seed']][num] += r[num]
                m[r['seed']][den] += r[den]
            return m
        return {f"{r['seed']}_{r['split']}": {num: r[num], den: r[den]} for r in rows}

    ia, ib = index(a), index(b)
    common = sorted(set(ia) & set(ib))
    ra = sum(v[num] for v in ia.values()) / sum(v[den] for v in ia.values())
    rb = sum(v[num] for v in ib.values()) / sum(v[den] for v in ib.values())
    Da = sum(v[den] for v in ia.values())
    Db = sum(v[den] for v in ib.values())
    ds = [(ib[k][num] - rb * ib[k][den]) / Db - (ia[k][num] - ra * ia[k][den]) / Da
          for k in common]
    return rb - ra, math.sqrt(sum(x * x for x in ds))


def report(label: str, init_rows, ckpt_rows, num: str, den: str) -> None:
    keys = ['n_kyoku', 'n_agari', 'n_houjuu', 'chip', 'one']
    ia_s, ib_s = by_seed(init_rows, keys), by_seed(ckpt_rows, keys)

    a_h, sa_h = ratio_se(init_rows, num, den)
    b_h, sb_h = ratio_se(ckpt_rows, num, den)
    a_s, sa_s = ratio_se(ia_s, num, den)
    b_s, sb_s = ratio_se(ib_s, num, den)

    diff = (b_h - a_h)
    se_h = math.sqrt(sa_h**2 + sb_h**2)          # 現行（半荘クラスタ・非対応）
    se_s = math.sqrt(sa_s**2 + sb_s**2)          # seed クラスタ・非対応
    _, se_hp = paired_diff_se(init_rows, ckpt_rows, num, den, cluster_seed=False)
    _, se_sp = paired_diff_se(init_rows, ckpt_rows, num, den, cluster_seed=True)

    sc = 100 if den == 'n_kyoku' else 1
    print(f'\n[{label}] {num}/{den}   init={a_h*sc:.4f} ckpt={b_h*sc:.4f} '
          f'diff={diff*sc:+.4f}')
    print(f'  {"手法":<34}{"SE(差)":>10}{"z":>9}{"vs 現行":>10}')
    for name, se in [
        ('現行: 半荘クラスタ・非対応', se_h),
        ('seed クラスタ・非対応', se_s),
        ('半荘クラスタ・対応あり', se_hp),
        ('seed クラスタ・対応あり', se_sp),
    ]:
        z = diff / se if se else float('nan')
        print(f'  {name:<34}{se*sc:>10.4f}{z:>+9.2f}{se/se_h:>9.2f}x')
    deff_i = (sa_s / sa_h) ** 2 if sa_h else float('nan')
    deff_c = (sb_s / sb_h) ** 2 if sb_h else float('nan')
    icc_i = (deff_i - 1) / 3
    icc_c = (deff_c - 1) / 3
    print(f'  design effect (seed/半荘)²: init {deff_i:.3f} / ckpt {deff_c:.3f}'
          f'   → 級内相関 ICC ≈ {icc_i:+.3f} / {icc_c:+.3f}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--init-logs', required=True)
    ap.add_argument('--ckpt-logs', required=True)
    ap.add_argument('--label', default='ckpt')
    args = ap.parse_args()

    init_rows = collect(Path(args.init_logs))
    ckpt_rows = collect(Path(args.ckpt_logs))
    n_seed = len({r['seed'] for r in init_rows})
    print(f'init {len(init_rows):,} 半荘 / {n_seed:,} seed'
          f'   ckpt {len(ckpt_rows):,} 半荘 / {len({r["seed"] for r in ckpt_rows}):,} seed')

    report(args.label, init_rows, ckpt_rows, 'n_houjuu', 'n_kyoku')
    report(args.label, init_rows, ckpt_rows, 'n_agari', 'n_kyoku')
    report(args.label, init_rows, ckpt_rows, 'chip', 'one')

    print('\n注: design effect > 1 なら現行 SE は過小評価（z が過大）、< 1 なら過大評価。')
    print('    ICC は 4 split の級内相関の近似（deff = 1 + (m−1)·ICC、m=4）。')
    print('    判定条件は analyze_fundamentals_1v3.py の出力が正であり本書は変更しない。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
