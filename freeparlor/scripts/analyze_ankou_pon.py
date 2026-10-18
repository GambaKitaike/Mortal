#!/usr/bin/env python3
"""暗刻からのポン率 — レンズ4 所見の指標化（read-only・GPU 不要）。

動機
----
2026-07-30 のレンズ4（b04 自己対戦）で Gamba が
「**暗刻から南ポン。そしてまた南切り。直らないねこれ**」と指摘した。
既に3枚持っている牌をポンすると:

  - 向聴は1ミリも進まない（面子は既に完成している）
  - 手が門前でなくなる（立直・ツモ・三暗刻を失う）
  - 手に残った4枚目は**完全な死に牌**になる
  - 相手に情報を与える

つまり**ほぼ常に純損**であり、方策の健全性の強いシグナルになる。所見のまま置かず
機械検出できる指標にしたのが本スクリプト（`qualitative_review_protocol.md` §3(4) の
「所見 → 定量指標への突き合わせ」）。

実装
----
手牌復元は `hand_replay.replay`（`analyze_call_quality.py` 等と共通の部品）を使う。
合法手判定は `PlayerState.last_cans` = libriichi 自身の判定。**自前で書かない**。
「ポンしたか」の判定は `analyze_call_quality.py` と同じく「鳴き機会の**次の**イベントが
自席の pon か」で行う。

出力
----
  - ポン総数
  - うち暗刻ポン（対象牌を既に3枚持っていた）の数と率
  - うち暗刻ポン後に**同じ局で**その牌（4枚目）を切った数 = 死に牌の確認
  - 参考: 対子ポン（2枚保持＝正常）の数

使い方
------
    PYTHONPATH=mortal python freeparlor/scripts/analyze_ankou_pon.py \
        --init-logs <dir> --ckpt-logs <dir> --label NAME
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_freeparlor_pnl_1v3 import seat_from_filename  # noqa: E402
from hand_replay import TRANSPARENT_EVENTS, replay, tile_id  # noqa: E402


def analyze_log(path: Path) -> dict:
    seat = seat_from_filename(path)
    r = {'n_pon': 0, 'n_pon_ankou': 0, 'n_pon_toitsu': 0, 'n_ankou_then_discard': 0}
    pending_target = None   # 直前の決定点が鳴き機会だったときの (対象牌, 手牌枚数)
    watch_dead_tile = None  # 暗刻ポン直後、4枚目を切るかを見る
    for ev, dec in replay(path, seat):
        et = ev.get('type')
        if et in TRANSPARENT_EVENTS:
            continue

        if watch_dead_tile is not None:
            # 4枚目（死に牌）を**その局のうちに**切ったかを見る。ポン直後の打牌に
            # 限定すると、安牌として抱えてから後で切るケースを取りこぼす
            if et == 'dahai' and ev.get('actor') == seat and ev.get('pai') is not None:
                if tile_id(ev['pai']) == watch_dead_tile:
                    r['n_ankou_then_discard'] += 1
                    watch_dead_tile = None
            elif et == 'start_kyoku':
                watch_dead_tile = None

        if pending_target is not None:
            target, held = pending_target
            if et == 'pon' and ev.get('actor') == seat:
                r['n_pon'] += 1
                if held >= 3:
                    r['n_pon_ankou'] += 1
                    watch_dead_tile = target
                elif held == 2:
                    r['n_pon_toitsu'] += 1
            pending_target = None

        if dec is None:
            continue
        if dec.target_tile is None or dec.self_riichi:
            continue
        if not dec.cans.can_pon:
            continue
        pending_target = (dec.target_tile, dec.tehai[dec.target_tile])
    return r


def ratio_and_se(rows, num, den):
    tot_d = sum(x[den] for x in rows)
    if tot_d == 0:
        return float('nan'), float('nan'), 0
    tot_n = sum(x[num] for x in rows)
    p = tot_n / tot_d
    # 半荘クラスタの ratio-estimator SE（他の診断スクリプトと同一の扱い）
    m = len(rows)
    if m < 2:
        return p, float('nan'), tot_d
    resid = [x[num] - p * x[den] for x in rows]
    s2 = sum(v * v for v in resid) / (m - 1)
    se = math.sqrt(m * s2) / tot_d
    return p, se, tot_d


def run(logs_dir: str) -> list[dict]:
    files = sorted(Path(logs_dir).glob('*.json.gz'))
    if not files:
        raise SystemExit(f'FATAL: no *.json.gz under {logs_dir}')
    return [analyze_log(p) for p in files]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--init-logs', required=True)
    ap.add_argument('--ckpt-logs', required=True)
    ap.add_argument('--label', default='ckpt')
    args = ap.parse_args()

    init_rows = run(args.init_logs)
    ckpt_rows = run(args.ckpt_logs)
    print(f'[init] {len(init_rows)} hanchan / ポン {sum(x["n_pon"] for x in init_rows)}')
    print(f'[{args.label}] {len(ckpt_rows)} hanchan / ポン {sum(x["n_pon"] for x in ckpt_rows)}')
    print()
    print(f'{"metric":<34}{"init":>12}{args.label:>14}{"diff":>10}{"SE":>8}{"z":>8}')
    for name, num, den in (('暗刻からのポン率', 'n_pon_ankou', 'n_pon'),):
        p0, se0, d0 = ratio_and_se(init_rows, num, den)
        p1, se1, d1 = ratio_and_se(ckpt_rows, num, den)
        diff = (p1 - p0) * 100
        se = math.sqrt(se0 * se0 + se1 * se1) * 100
        z = diff / se if se and not math.isnan(se) and se > 0 else float('nan')
        print(f'{name:<34}{p0 * 100:>11.2f}%{p1 * 100:>13.2f}%{diff:>+10.2f}{se:>8.2f}{z:>+8.2f}')
    print()
    for label, rows in (('init', init_rows), (args.label, ckpt_rows)):
        n_ank = sum(x['n_pon_ankou'] for x in rows)
        n_dead = sum(x['n_ankou_then_discard'] for x in rows)
        print(f'  [{label}] 暗刻ポン {n_ank} 件 / うち同じ局で4枚目を切った {n_dead} 件 '
              f'/ 対子ポン {sum(x["n_pon_toitsu"] for x in rows)} 件')
    print()
    print('注: 暗刻ポンは向聴を進めず門前性・暗刻役を失い4枚目が死に牌になるため、ほぼ常に純損。')
    print('    ただし本指標は「絶対にゼロであるべき」とまでは主張しない（現物切りのための')
    print('    形式的な鳴き等、極稀に説明可能な局面はある）。init との差分で読むこと。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
