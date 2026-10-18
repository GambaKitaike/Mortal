#!/usr/bin/env python3
"""和了の構成分解 + カン頻度の診断 (1v3 eval game_logs、read-only)。

動機: 2026-07-26 の Gamba 定性レビュー (レンズ4) の所見と測定値の食い違い。
所見は「打点を作らない」(跳満テンパイ拒否・ホンイツ不行・トイトイ拒否・加カン見送り・
暗刻/対子を壊す) に集中しているのに、`analyze_fundamentals_1v3.py` の測定は
平均和了打点 +1137.8点 (z=+7.72) の**上昇**を示した。

作業仮説: **打点上昇は立直由来であって手作り由来ではない**。立直率が 24.3%->37.6% に
上がっており、裏ドラ・一発・立直棒で打点が嵩上げされていれば「手役を作らない」と
「平均打点が上がる」は両立する。

検証方法: 和了を 立直 / ダマ (門前非立直) / 副露 に層別し、それぞれの平均打点を
init と比較する。仮説が正しければ **立直和了の打点だけが上がり、ダマ・副露の打点は
上がらない (むしろ下がる)**。手作り能力が落ちていないなら全層で上がるはず。

打点・和了内訳は libriichi `Stat` をログ単位で呼んで取得する (定義の複製禁止)。
カンは Stat に無いので mjai イベント (ankan/kakan/daiminkan) を直接数える。

**七対子・対々和の判定は本スクリプトでは出せない** — mjai の hora イベントは役を
持たず、Stat も役別内訳を露出していない。Gamba 所見の「対子系に弱い」は本書の
対象外で、和了形の再構成が要る別タスク。

SE は半荘クラスタ。差の SE は独立2標本近似 (analyze_fundamentals_1v3.py と同じ)。
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

KAN_TYPES = ('ankan', 'kakan', 'daiminkan')


def scan_log(path: Path) -> dict:
    from libriichi.stat import Stat

    seat = seat_from_filename(path)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        text = f.read()
    st = Stat.from_log(text, seat)

    kan = {k: 0 for k in KAN_TYPES}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get('type') in KAN_TYPES and ev.get('actor') == seat:
            kan[ev['type']] += 1

    return {
        'n_kyoku': int(st.round),
        'agari': int(st.agari),
        # 和了の層別 (件数と合計打点)
        'riichi_agari': int(st.riichi_agari),
        'riichi_agari_point': float(st.riichi_agari_point),
        'dama_agari': int(st.dama_agari),
        'dama_agari_point': float(st.dama_agari_point),
        'fuuro_agari': int(st.fuuro_agari),
        'fuuro_agari_point': float(st.fuuro_agari_point),
        'agari_point': float(st.agari_point_ko + st.agari_point_oya),
        # 参考
        'riichi': int(st.riichi),
        'fuuro': int(st.fuuro),
        'n_kan': sum(kan.values()),
        **{f'n_{k}': v for k, v in kan.items()},
    }


def ratio_and_se(rows, num_key: str, den_key: str):
    tot_d = sum(r[den_key] for r in rows)
    if tot_d == 0:
        return None, None
    tot_n = sum(r[num_key] for r in rows)
    ratio = tot_n / tot_d
    var = sum((r[num_key] - ratio * r[den_key]) ** 2 for r in rows) / tot_d**2
    return ratio, math.sqrt(var)


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
        legs[name] = [scan_log(p) for p in paths]
        n_k = sum(r['n_kyoku'] for r in legs[name])
        print(f'[{name}] {len(paths)} hanchan / {n_k} kyoku')

    def emit(title, specs):
        print(f'\n=== {title} ===')
        print(f"{'metric':<26} {'init':>16} {args.label:>16} {'diff':>10} {'SE':>9} {'z':>7}")
        for label, num, den, scale, unit in specs:
            v0, s0 = ratio_and_se(legs['init'], num, den)
            v1, s1 = ratio_and_se(legs[args.label], num, den)
            if v0 is None or v1 is None:
                print(f'{label:<26} {"(分母0)":>16}')
                continue
            diff = (v1 - v0) * scale
            se = math.sqrt(s0**2 + s1**2) * scale
            z = diff / se if se else float('nan')
            print(f'{label:<26} {v0*scale:>14.2f}{unit} {v1*scale:>14.2f}{unit} '
                  f'{diff:>+10.2f} {se:>9.2f} {z:>+7.2f}')

    emit('和了の層別 平均打点 (点/和了) — 作業仮説の本体', [
        ('全和了',        'agari_point',        'agari',        1.0, ' '),
        ('立直和了',      'riichi_agari_point', 'riichi_agari', 1.0, ' '),
        ('ダマ和了',      'dama_agari_point',   'dama_agari',   1.0, ' '),
        ('副露和了',      'fuuro_agari_point',  'fuuro_agari',  1.0, ' '),
    ])

    emit('和了の構成比 (件/全和了)', [
        ('立直和了 割合', 'riichi_agari', 'agari', 100.0, '%'),
        ('ダマ和了 割合', 'dama_agari',   'agari', 100.0, '%'),
        ('副露和了 割合', 'fuuro_agari',  'agari', 100.0, '%'),
    ])

    emit('カン頻度 (件/局)', [
        ('カン全体',   'n_kan',       'n_kyoku', 100.0, '%'),
        ('暗カン',     'n_ankan',     'n_kyoku', 100.0, '%'),
        ('加カン',     'n_kakan',     'n_kyoku', 100.0, '%'),
        ('大明カン',   'n_daiminkan', 'n_kyoku', 100.0, '%'),
    ])

    print('\n注: 七対子・対々和の判定は mjai ログに役情報が無いため本書では出せない '
          '(和了形の再構成が要る別タスク)。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
