#!/usr/bin/env python3
"""off-policy staleness の分解（L1 O1 の事前登録用計測器）。GPU 不要・読み取り専用。

なぜ別スクリプトか
------------------
`analyze_ppo_optimization_health.py` は L1 層の**横断診断**（8 run の構造的事実の提示）で、
commit 済み `ppo_optimization_health_20260725.md` の再現器でもあるため出力書式を変えられない
（同書 §7 が数値を回帰基準として固定している）。本スクリプトは L1 O1
（`submit_every` 引き下げ）の**事前登録に必要な2つの量**だけを出す新設の計測器:

  1. **staleness の分解** — 「1 バッチが学習される時点で、その `logp_old` を記録した
     パラメータは何 optimizer step 古いか」を step 単位で復元し、
     **submit_every で動く成分（量子化）**と**動かない成分（session+queue の transit）**に分ける。
     `batch_lag.lag` は param_version 単位（= submit_every 粒度）なので、
     lag だけを見ると「submit_every を 1/5 にすれば staleness も 1/5」と誤読する。
     実際に動くのは量子化成分のみ。

  2. **発進ゲート（--gate）** — `submit_every` が実際に効いているかの機械的検査。
     `trainer_param_version` が step の関数として設定値どおりに刻まれているかを assert する。
     `kl_ref_mean > 0` 型の「環境で偽陽性/偽陰性になるゲート」を避け
     （`anchored_ppo_design.md` §5-a1 の教訓）、**機械的に保証される量のみ**を合否条件にする。

staleness の復元方法
--------------------
trainer は step が `submit_every` の倍数になるたびに param を submit し
`trainer_param_version` を +1 する（`mortal/train_ppo.py:605-608`）。起動時に
version 1 を idle submit する（`:150-151`）ので、

    version v (v>=1) のスナップショット = trainer step (v-1) * submit_every

client は session 開始時に server の最新 version を1回だけ取得し（`mortal/client.py:217-234`）、
その session 中の全半荘に同じ `param_version` を刻む。よって diag の `batch_lag` から

    staleness_steps = trainer_step - (param_version - 1) * submit_every
                    = 量子化成分 (0 .. submit_every-1) + transit 成分 (session + queue)
    transit ≈ submit_every * lag        (lag = trainer_param_version - param_version)
    量子化 ≈ staleness - submit_every * lag   → 期待値 (submit_every-1)/2

⚠ `save_every` が `submit_every` の倍数でない run では追加 submit が入り
（`train_ppo.py:619-622`）version と step の対応がずれる。その場合は WARNING を出し、
分解値を「未算出」として明示する（0 埋めしない）。

使い方
------
    # 完走 run の分解（複数指定可）
    python freeparlor/scripts/analyze_staleness_decomposition.py \
        --run /home/gamba/mahjong/runs/ppo/anchor_k_20260727_000805

    # 発進ゲート（step 1-200 で submit_every の刻みを assert）
    python freeparlor/scripts/analyze_staleness_decomposition.py \
        --run <RUN_DIR> --gate --gate-window 1 200

    # submit 1 回の実測オーバーヘッド（trainer.log の timestamp 差）
    python freeparlor/scripts/analyze_staleness_decomposition.py --run <RUN_DIR> --submit-cost
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_WINDOWS = ((1, 100), (101, 200), (201, 500), (501, 2000), (2001, 16000))


def load_toml(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        import tomli as tomllib  # type: ignore
    with path.open('rb') as f:
        return tomllib.load(f)


def read_batch_lag(diag: Path) -> list[tuple[int, int, int]]:
    """(trainer_step, param_version, trainer_param_version) を返す。"""
    rows = []
    with diag.open(encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('event') != 'batch_lag':
                continue
            t = d.get('trainer_step')
            pv = d.get('param_version')
            tpv = d.get('trainer_param_version')
            if t is None or pv is None or tpv is None:
                continue
            rows.append((int(t), int(pv), int(tpv)))
    return rows


def read_clip_epoch1(diag: Path) -> dict[int, float]:
    out = {}
    with diag.open(encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('event') == 'ppo_epoch' and d.get('epoch') == 1:
                t = d.get('trainer_step')
                c = d.get('clip_fraction')
                if t is not None and c is not None:
                    out[int(t)] = float(c)
    return out


def infer_base_step(rows, submit_every: int) -> int | None:
    """version 1 に対応する trainer step（resume run では resume 開始 step）を復元する。

    非 resume run では 0。resume run では checkpoint の step から採番が 1 に戻るため
    （`train_ppo.py:150-151` の idle submit）base != 0 になる。base を仮定せずに
    `t - (pv-1)*submit_every` を使うと staleness が resume 分だけ水増しされるので、
    ここで復元し、恒等式が全レコードで成立しない場合は None（= 未算出）を返す。
    """
    cands = [t - (tpv - 1) * submit_every for t, _, tpv in rows]
    base = (min(cands) // submit_every) * submit_every
    if base < 0:
        return None
    for t, _, tpv in rows:
        if tpv != (t - base) // submit_every + 1:
            return None
    return base


def decompose(rows, submit_every: int, windows, out=sys.stdout) -> None:
    print(f'  submit_every = {submit_every}', file=out)
    neg = [r for r in rows if r[1] < 0]
    if neg:
        print(f'  WARNING: param_version < 0 が {len(neg)} 件（client 未受信）→ 除外', file=out)
    rows = [r for r in rows if r[1] >= 0]
    if not rows:
        print('  no batch_lag rows（未算出）', file=out)
        return
    base = infer_base_step(rows, submit_every)
    if base is None:
        print('  WARNING: trainer_param_version と trainer_step の恒等式が成立しない '
              '（版採番の不整合。resume 境界・追加 submit を疑う）', file=out)
        print('  → staleness の step 復元は不能。**未算出**（0 埋め・近似はしない）', file=out)
        return
    if base:
        print(f'  NOTE: resume run（version 1 = step {base}）。staleness は base 補正込みで算出。'
              f'step 窓のラベルは元の trainer_step のまま', file=out)
    bad_lag = [r for r in rows if r[2] - r[1] < 0]
    if bad_lag:
        print(f'  WARNING: lag < 0 が {len(bad_lag)} 件（版採番の不整合。resume 境界を疑う）', file=out)
    print(f'  {"step 窓":>16} | {"n":>6} | {"staleness mean":>14} | {"min":>4} | {"max":>4} '
          f'| {"lag mean":>8} | {"transit":>7} | {"量子化":>6}', file=out)
    for lo, hi in windows:
        w = [r for r in rows if lo <= r[0] <= hi]
        if not w:
            print(f'  {f"{lo}-{hi}":>16} | (no rows)', file=out)
            continue
        stal = [t - base - (pv - 1) * submit_every for t, pv, _ in w]
        lag = [tpv - pv for _, pv, tpv in w]
        transit = submit_every * st.mean(lag)
        print(f'  {f"{lo}-{hi}":>16} | {len(w):6d} | {st.mean(stal):14.1f} | {min(stal):4d} '
              f'| {max(stal):4d} | {st.mean(lag):8.3f} | {transit:7.1f} | {st.mean(stal) - transit:6.1f}',
              file=out)
    stal_all = [t - base - (pv - 1) * submit_every for t, pv, _ in rows]
    lag_all = [tpv - pv for _, pv, tpv in rows]
    transit_all = submit_every * st.mean(lag_all)
    quant_all = st.mean(stal_all) - transit_all
    print(f'  全体: staleness mean = {st.mean(stal_all):.1f} step '
          f'= transit {transit_all:.1f} + 量子化 {quant_all:.1f}', file=out)
    print(f'        （量子化の理論値 (submit_every-1)/2 = {(submit_every - 1) / 2:.1f}）', file=out)
    print(f'  ⇒ submit_every を S に変えた場合の予測 staleness = transit {transit_all:.1f} + (S-1)/2', file=out)
    for s in (1, 10, 25):
        if s == submit_every:
            continue
        pred = transit_all + (s - 1) / 2
        print(f'       S={s:3d} → {pred:6.1f} step '
              f'（現行比 {100 * (pred / st.mean(stal_all) - 1):+.1f}%）', file=out)


def clip_by_lag(rows, clip1, min_step: int, out=sys.stdout) -> None:
    groups: dict[int, list[float]] = {}
    for t, pv, tpv in rows:
        if t < min_step or pv < 0:
            continue
        c = clip1.get(t)
        if c is None:
            continue
        groups.setdefault(tpv - pv, []).append(c)
    if not groups:
        print(f'  no ppo_epoch(epoch=1) rows for step >= {min_step}（未算出）', file=out)
        return
    print(f'  clip_fraction@epoch1 を lag で条件付け（step >= {min_step}）:', file=out)
    for lag in sorted(groups):
        v = groups[lag]
        if len(v) < 5:
            print(f'    lag={lag} n={len(v)} （n<5 のため統計を出さない）', file=out)
            continue
        print(f'    lag={lag} n={len(v):6d} mean={st.mean(v):.4f} median={st.median(v):.4f}', file=out)
    if len(groups.get(1, [])) >= 5 and len(groups.get(2, [])) >= 5:
        a, b = groups[1], groups[2]
        diff = st.mean(b) - st.mean(a)
        se = (st.variance(a) / len(a) + st.variance(b) / len(b)) ** 0.5
        print(f'    lag2 - lag1 = {diff:+.5f} (SE {se:.5f}, z={diff / se:+.2f})', file=out)
        print('    ⚠ これは横断（同一 run 内で lag が違うバッチの比較）であり因果ではない。'
              'run 間で符号が一致しないことが実測されている', file=out)


TS = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)\s')


def _ts(line: str) -> float | None:
    m = TS.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').timestamp() + int(m.group(2)) / 1000


def submit_cost(trainer_log: Path, submit_every: int, out=sys.stdout) -> None:
    step_re = re.compile(r'ppo step (\d+): epoch1_clip')
    steps: list[tuple[int, float]] = []
    n_submit = 0
    with trainer_log.open(encoding='utf-8', errors='replace') as f:
        for line in f:
            t = _ts(line)
            if t is None:
                continue
            m = step_re.search(line)
            if m:
                steps.append((int(m.group(1)), t))
            elif 'param has been submitted' in line:
                n_submit += 1
    if len(steps) < 100:
        print(f'  ppo step 行が {len(steps)} 件しかない（未算出）', file=out)
        return
    on, off = [], []
    for (n1, t1), (n2, t2) in zip(steps, steps[1:]):
        if n2 != n1 + 1:
            continue
        gap = t2 - t1
        if gap > 120:  # drain 待ち・停滞は外れ値として除外（median で読むが念のため）
            continue
        (on if n1 % submit_every == 0 else off).append(gap)
    if len(on) < 20 or len(off) < 20:
        print(f'  submit 直後 n={len(on)} / それ以外 n={len(off)} — 標本不足（未算出）', file=out)
        return
    print(f'  submit 回数（log 行）= {n_submit}', file=out)
    print(f'  step 間隔 median: submit を挟む = {st.median(on):.3f}s / 挟まない = {st.median(off):.3f}s',
          file=out)
    over = st.median(on) - st.median(off)
    total = steps[-1][1] - steps[0][1]
    nstep = steps[-1][0] - steps[0][0]
    print(f'  ⇒ submit 1 回のオーバーヘッド ≈ {over:.3f}s', file=out)
    print(f'  run 全体: {total / 3600:.2f}h / {nstep} step = {total / nstep:.3f} s/step', file=out)
    for s in (10, 25):
        extra = (nstep / s - nstep / submit_every) * over
        print(f'       submit_every={s} にした場合の追加コスト ≈ {extra / 60:.1f} 分 '
              f'（run 全体の {100 * extra / total:+.2f}%）', file=out)


def gate(rows, submit_every: int, lo: int, hi: int, out=sys.stdout) -> bool:
    """機械ゲート: trainer_param_version が submit_every の刻みどおりか。"""
    w = [r for r in rows if lo <= r[0] <= hi]
    print(f'  ゲート窓 step [{lo}, {hi}]: n={len(w)}', file=out)
    if not w:
        print('  FAIL: 窓が空', file=out)
        return False
    base = infer_base_step([r for r in rows if r[1] >= 0], submit_every)
    if base is None:
        print('  FAIL: trainer_param_version と trainer_step の恒等式が run 全体で成立しない', file=out)
        return False
    if base:
        print(f'  NOTE: resume run（version 1 = step {base}）', file=out)
    bad = [(t, tpv, (t - base) // submit_every + 1)
           for t, _, tpv in w if tpv != (t - base) // submit_every + 1]
    if bad:
        print(f'  FAIL: trainer_param_version が期待値と不一致 {len(bad)} 件 '
              f'(例: step={bad[0][0]} 実測={bad[0][1]} 期待={bad[0][2]})', file=out)
        return False
    print(f'  PASS: 全 {len(w)} 件で trainer_param_version == (step - {base}) // {submit_every} + 1',
          file=out)
    stal = [t - base - (pv - 1) * submit_every for t, pv, _ in w if pv >= 0]
    if stal:
        print(f'  INFO（合否条件ではない）: 窓内の staleness mean = {st.mean(stal):.1f} step '
              f'(min {min(stal)} / max {max(stal)})', file=out)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, help='run dir（複数指定可）')
    ap.add_argument('--submit-every', type=int, default=None,
                    help='既定は run の config.toml の [control] submit_every')
    ap.add_argument('--gate', action='store_true', help='発進ゲート判定を行う')
    ap.add_argument('--gate-window', nargs=2, type=int, default=(1, 200), metavar=('LO', 'HI'))
    ap.add_argument('--submit-cost', action='store_true', help='trainer.log から submit コストを実測')
    ap.add_argument('--clip-min-step', type=int, default=500)
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()

    out = open(args.out, 'w', encoding='utf-8') if args.out else sys.stdout
    rc = 0
    try:
        for run in args.run:
            R = Path(run)
            print(f'\n=== {R.name} ===', file=out)
            diag = R / 'logs' / 'ppo_diag.jsonl'
            if not diag.is_file():
                print(f'  FATAL: {diag} が無い', file=out)
                rc = 1
                continue
            cfg = load_toml(R / 'config.toml')
            if cfg is None:
                print('  WARNING: config.toml が無い（submit_every は --submit-every 指定が必須）',
                      file=out)
            se = args.submit_every
            if se is None:
                if cfg is None:
                    print('  FATAL: submit_every を決定できない（未算出）', file=out)
                    rc = 1
                    continue
                # 注: [control] にある。`analyze_ppo_optimization_health.py` は
                # ('online','submit_every') を見ているため拾えていない（同書の値は
                # config を人手で読んだもの。ここでは正しい節を見る）
                se = int(cfg['control']['submit_every'])
            save_every = int(cfg['control']['save_every']) if cfg else None
            if save_every is not None and save_every % se != 0:
                print(f'  WARNING: save_every={save_every} が submit_every={se} の倍数でない '
                      f'→ 追加 submit で version/step 対応がずれる。分解値は未算出扱い', file=out)
                continue
            rows = read_batch_lag(diag)
            decompose(rows, se, DEFAULT_WINDOWS, out=out)
            clip_by_lag(rows, read_clip_epoch1(diag), args.clip_min_step, out=out)
            if args.submit_cost:
                tl = R / 'logs' / 'trainer.log'
                if tl.is_file():
                    submit_cost(tl, se, out=out)
                else:
                    print(f'  WARNING: {tl} が無い → submit コストは未算出', file=out)
            if args.gate:
                ok = gate(rows, se, args.gate_window[0], args.gate_window[1], out=out)
                if not ok:
                    rc = 1
    finally:
        if out is not sys.stdout:
            out.close()
    return rc


if __name__ == '__main__':
    sys.exit(main())
