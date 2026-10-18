#!/usr/bin/env python3
"""PPO 最適化衛生（L1 層）の横断診断。

目的
----
`fundamentals_degradation_diagnosis_20260725.md` §4 の原因仮説（主犯=アンカー不在 /
増幅=均衡退化・スパース報酬）は L2/L3/報酬設計を覆うが、**最適化そのものの健全性**を
仮説に含めていない。本スクリプトは既存 run の `logs/ppo_diag.jsonl` を横断集計して、
その空白を埋める一次証拠を出す。GPU 不要・読み取り専用。

`summarize_ppo_diag.py` との関係
--------------------------------
`summarize_ppo_diag.py` は P2 期の単一 run 用サマライザで、commit 済み `ppo_p2_diag.md` の
再現器として機能している（既定パスがハードコード、扱う event は batch_lag / ppo_epoch のみ）。
出力書式を変えると当該成果物の再現性を壊すため**拡張せず**、本スクリプトを新設した。
集計対象の event と観点が異なる（batch_size / advantage_decomp / action_mass / lag 条件付け）。

出力する観点
------------
[A] batch_size 分布と `minibatch_size` に対する越え率
    → ミニバッチ分割が実際に発生しているか（1 optimizer step の実効サンプル数）
[B] clip_fraction を **param snapshot age（optimizer step 単位）で条件付けた**表 ← 本診断の中心
    → trust region が「学習」で消費されているか「off-policy staleness」で消費されているか。
      age は `trainer_step - min{trainer_step : 同一 param_version}` で推定する。
      注意: `batch_lag.lag` は **param_version 単位**（`submit_every` 粒度）なので
      staleness の代理変数としては粗すぎる（lag=0 でも最大 submit_every-1 step の
      ドリフトを含む）。lag 別の表も出すが、読みは age 側を正とする
[C] clip_fraction の epoch 別
    → epoch を重ねて clip が増えるなら学習由来。増えないなら batch 到着時点で
      既に trust region が消費されている（= staleness offset が支配）
[D] clip_fraction / ratio_std の step トレンド
[E] advantage_decomp の各カテゴリのサンプル数 n
    → 局末のみ非ゼロ報酬という構造下で、advantage 推定が何サンプルに乗っているか
[F] action_mass（立直 / 鳴き）の step トレンド（参考・判定非関与）
[G] 欠損 event の明示

禁則遵守: 欠損はサイレントに 0 埋めせず "no <event> rows" と明示する
（`CLAUDE.md` 実装の禁則: サイレントフォールバック禁止）。

使い方
------
    # 既定: /home/gamba/mahjong/runs/ppo/ 配下の diag を持つ全 run
    python freeparlor/scripts/analyze_ppo_optimization_health.py

    # run を明示
    python freeparlor/scripts/analyze_ppo_optimization_health.py \
        --run /home/gamba/mahjong/runs/ppo/anchor_c_20260725_164756

    # レポート用に保存
    python freeparlor/scripts/analyze_ppo_optimization_health.py -o out.txt
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path('/home/gamba/mahjong/runs/ppo')

# 集計対象の event。ここに無い event は素通しし、[G] で「未集計」として数だけ報告する。
TRACKED_EVENTS = (
    'batch_lag',
    'ppo_epoch',
    'advantage_decomp',
    'action_mass',
    'call_bonus',
    'kl_anchor',
    'kyoku_reward_decomp',
    'grp_calibration',
)

ADV_CATEGORIES = ('call_taken', 'call_declined', 'riichi_taken', 'riichi_declined')

# config.toml から拾う L1 関連キー。(セクション, キー) の順で探す。
CONFIG_KEYS = (
    ('ppo', 'lr'),
    ('ppo', 'minibatch_size'),
    ('ppo', 'ppo_epochs'),
    ('ppo', 'eps_clip'),
    ('ppo', 'gamma_disc'),
    ('ppo', 'gae_lambda'),
    ('ppo', 'c_ent'),
    ('ppo', 'kl_beta'),
    ('optim', 'max_grad_norm'),
    ('online', 'submit_every'),
    ('control', 'save_every'),
    ('opponent_pool', 'anchor_prob'),
)


def load_toml(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return None
    with path.open('rb') as f:
        return tomllib.load(f)


def config_summary(run_dir: Path) -> tuple[dict[str, object], str]:
    """run dir の config.toml から L1 関連パラメータを抜く。

    Returns (値の dict, 出所の説明). config が無い場合は空 dict と理由を返す
    （欠損をサイレントに埋めない）。
    """
    cfg = load_toml(run_dir / 'config.toml')
    if cfg is None:
        return {}, 'config.toml なし（または tomllib/tomli 不在）'
    out: dict[str, object] = {}
    for section, key in CONFIG_KEYS:
        node = cfg.get(section)
        if isinstance(node, dict) and key in node:
            out[f'{section}.{key}'] = node[key]
    return out, 'run dir の config.toml'


def quant(vals: list[float]) -> tuple[float, float, float]:
    """(p25, median, p75)。n<4 では quantiles が使えないので median で埋める。"""
    med = st.median(vals)
    if len(vals) < 4:
        return med, med, med
    q = st.quantiles(vals, n=4)
    return q[0], med, q[2]


def bucket_of(step: int, width: int) -> int:
    return step // width * width


class RunStats:
    def __init__(self, run_dir: Path, bucket: int):
        self.run_dir = run_dir
        self.name = run_dir.name
        self.bucket = bucket
        self.event_counts: dict[str | None, int] = defaultdict(int)
        self.bad_lines = 0
        self.steps: list[int] = []

        # [A]/[B]
        self.batch_sizes: list[int] = []
        self.lag_hist: dict[int, int] = defaultdict(int)
        self.neg_lag_steps: list[int] = []
        # step -> param_version（age の算出に使う。batch_lag と ppo_epoch の join キー）
        self.pv_by_step: dict[int, int] = {}
        # [B]/[C]/[D]
        self.clip_by_lag: dict[int, list[float]] = defaultdict(list)
        self.clip_by_epoch: dict[int, list[float]] = defaultdict(list)
        self.rstd_by_epoch: dict[int, list[float]] = defaultdict(list)
        self.clip_by_lag_epoch: dict[tuple[int, int], list[float]] = defaultdict(list)
        self.clip_trend: dict[int, list[float]] = defaultdict(list)
        self.rstd_trend: dict[int, list[float]] = defaultdict(list)
        self.max_epoch = 0
        # epoch1 の (step, clip)。finalize() で age に変換する
        self.e1_rows: list[tuple[int, float]] = []
        # finalize() の産物
        self.clip_by_age: dict[int, list[float]] = defaultdict(list)
        self.clip_at_step0: list[float] = []
        self.age_bucket = 10
        # [E]
        self.adv_n: dict[str, list[int]] = defaultdict(list)
        self.adv_absent: dict[str, int] = defaultdict(int)
        # [F]
        self.am_trend: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        # kl (anchor Arm K のみ)
        self.kl_mean: list[float] = []
        # call_bonus b の観測値（Stage3 系のみ非ゼロ）
        self.call_bonus_b: set[float] = set()

    # -- ingest ----------------------------------------------------------
    def feed(self, row: dict) -> None:
        event = row.get('event')
        self.event_counts[event] += 1
        step = row.get('trainer_step')
        if isinstance(step, int):
            self.steps.append(step)

        if event == 'batch_lag':
            bs = row.get('batch_size')
            lag = row.get('lag')
            pv = row.get('param_version')
            if isinstance(bs, int):
                self.batch_sizes.append(bs)
            if isinstance(lag, int):
                self.lag_hist[lag] += 1
                if lag < 0 and isinstance(step, int):
                    self.neg_lag_steps.append(step)
            if isinstance(pv, int) and isinstance(step, int):
                self.pv_by_step[step] = pv

        elif event == 'ppo_epoch':
            clip = row.get('clip_fraction')
            epoch = row.get('epoch')
            lag = row.get('param_lag')
            rstd = row.get('ratio_std')
            if clip is None or not isinstance(epoch, int):
                return
            self.max_epoch = max(self.max_epoch, epoch)
            self.clip_by_epoch[epoch].append(clip)
            if rstd is not None:
                self.rstd_by_epoch[epoch].append(rstd)
            if isinstance(lag, int):
                self.clip_by_lag[lag].append(clip)
                self.clip_by_lag_epoch[(lag, epoch)].append(clip)
            if isinstance(step, int):
                b = bucket_of(step, self.bucket)
                self.clip_trend[b].append(clip)
                if rstd is not None:
                    self.rstd_trend[b].append(rstd)
                if epoch == 1:
                    self.e1_rows.append((step, clip))

        elif event == 'advantage_decomp':
            raw = row.get('raw')
            if not isinstance(raw, dict):
                return
            for cat in ADV_CATEGORIES:
                node = raw.get(cat)
                if isinstance(node, dict) and isinstance(node.get('n'), int):
                    self.adv_n[cat].append(node['n'])
                else:
                    # null = そのバッチに該当決定が無い。0 埋めせず「不在」として数える
                    self.adv_absent[cat] += 1

        elif event == 'action_mass':
            if not isinstance(step, int):
                return
            b = bucket_of(step, self.bucket)
            for key in ('pi_riichi_given_possible', 'pi_call_given_possible',
                        'pi_call_given_possible_aka_held', 'pi_call_aka_over_no_aka'):
                v = row.get(key)
                if isinstance(v, (int, float)):
                    self.am_trend[b][key].append(float(v))

        elif event == 'kl_anchor':
            v = row.get('kl_ref_mean')
            if isinstance(v, (int, float)):
                self.kl_mean.append(float(v))

        elif event == 'call_bonus':
            v = row.get('b')
            if isinstance(v, (int, float)):
                self.call_bonus_b.add(float(v))

    # -- finalize --------------------------------------------------------
    def finalize(self) -> None:
        """epoch1 の clip を param snapshot age（optimizer step 単位）に割り付ける。

        age = trainer_step - min{trainer_step : 同じ param_version}
        client は param_version v を受け取った直後から v で対局を生成するので、
        v を持つ最初の batch が trainer に届いた step ≈ v が作られた step。
        よって age は「client のスナップショットから trainer が何 step 進んだか」の推定値。
        """
        if not self.pv_by_step:
            return
        first_step_of: dict[int, int] = {}
        for step, pv in self.pv_by_step.items():
            if pv not in first_step_of or step < first_step_of[pv]:
                first_step_of[pv] = step
        for step, clip in self.e1_rows:
            pv = self.pv_by_step.get(step)
            if pv is None:
                continue  # batch_lag が無い step。0 埋めせず単に集計外
            age = step - first_step_of[pv]
            self.clip_by_age[bucket_of(age, self.age_bucket)].append(clip)
            if step == 0:
                self.clip_at_step0.append(clip)

    # -- report ----------------------------------------------------------
    def header(self, out) -> None:
        cfg, cfg_src = config_summary(self.run_dir)
        lo = min(self.steps) if self.steps else None
        hi = max(self.steps) if self.steps else None
        print(f'\n{"=" * 78}', file=out)
        print(f'RUN {self.name}', file=out)
        print(f'{"=" * 78}', file=out)
        print(f'  trainer_step 範囲 : {lo} .. {hi}', file=out)
        print(f'  batch 数          : {self.event_counts.get("batch_lag", 0)}'
              f'  (ppo_epoch レコード {self.event_counts.get("ppo_epoch", 0)})', file=out)
        if self.bad_lines:
            print(f'  ⚠ parse 不能行     : {self.bad_lines}', file=out)
        if cfg:
            print(f'  config ({cfg_src}):', file=out)
            for k, v in cfg.items():
                print(f'      {k:28s} = {v}', file=out)
        else:
            print(f'  config: {cfg_src}', file=out)
        if self.call_bonus_b:
            print(f'  観測された call_bonus b : {sorted(self.call_bonus_b)}', file=out)

    def report_a(self, out) -> int | None:
        """[A] batch_size 分布。戻り値は config の minibatch_size（無ければ None）。"""
        print('\n[A] batch_size 分布（1 optimizer step の実効サンプル数）', file=out)
        if not self.batch_sizes:
            print('    no batch_lag rows', file=out)
            return None
        bs = self.batch_sizes
        p25, med, p75 = quant([float(x) for x in bs])
        print(f'    n={len(bs)} min={min(bs)} p25={p25:.0f} median={med:.0f} '
              f'p75={p75:.0f} max={max(bs)} mean={st.mean(bs):.1f}', file=out)
        cfg, _ = config_summary(self.run_dir)
        mb = cfg.get('ppo.minibatch_size')
        if isinstance(mb, int):
            over = sum(1 for x in bs if x > mb)
            print(f'    minibatch_size={mb} を超える batch: {over}/{len(bs)} '
                  f'= {over / len(bs):.4%}', file=out)
            if over == 0:
                print('    → ミニバッチ分割は一度も発生していない '
                      '（実効「full-batch × ppo_epochs」）', file=out)
            return mb
        print(f'    minibatch_size 不明（config なし）→ 越え率は算出しない。'
              f'参考: max={max(bs)}', file=out)
        return None

    def report_b(self, out) -> None:
        print('\n[B] clip_fraction × param snapshot age ← 本診断の中心', file=out)
        print('    age = trainer_step - min{trainer_step : 同一 param_version}', file=out)
        print('        = client のスナップショットから trainer が進んだ optimizer step 数', file=out)
        print('    clip は epoch 1 のみ（= このバッチに勾配を当てる前の値）', file=out)
        if not self.clip_by_age:
            print('    算出不能（batch_lag の param_version または ppo_epoch が欠損）', file=out)
        else:
            if self.clip_at_step0:
                print(f'    ※ 陽性対照 trainer_step=0（age=0・client と trainer が同一パラメータ）: '
                      f'clip={st.mean(self.clip_at_step0):.4f}', file=out)
            print(f'    {"age":>9} {"n":>8} {"clip mean":>10} {"clip med":>9} {"clip max":>9}',
                  file=out)
            for a in sorted(self.clip_by_age):
                v = self.clip_by_age[a]
                print(f'    {a:>4}-{a + self.age_bucket - 1:<4} {len(v):>8} {st.mean(v):>10.4f} '
                      f'{st.median(v):>9.4f} {max(v):>9.4f}', file=out)
            print('    読み: age=0 付近で既に clip が高いなら、trust region は', file=out)
            print('          「このバッチの学習」ではなく client/trainer 間の', file=out)
            print('          パラメータ非対称（logp_old は client が行動時に記録）で消費されている。', file=out)
            print('          age が平坦なら staleness は定常（下限が buffer 滞留 + submit_every で決まる）', file=out)
            print('          であり、age ではなく [B2] の lag 分布が実効 staleness を表す', file=out)

        print(f'\n[B2] clip_fraction × batch_lag.lag（param_version 単位 = submit_every step 粒度）',
              file=out)
        if self.lag_hist:
            tot = sum(self.lag_hist.values())
            dist = '  '.join(f'lag{k}={self.lag_hist[k]}({self.lag_hist[k] / tot:.1%})'
                             for k in sorted(self.lag_hist))
            print(f'    lag 分布（batch 単位, n={tot}）: {dist}', file=out)
            neg = sum(n for k, n in self.lag_hist.items() if k < 0)
            if neg:
                # 版採番の不整合。サイレントに丸めず大声で出す
                print(f'    ⚠ WARNING: lag < 0 のバッチが {neg}/{tot} ({neg / tot:.2%})。', file=out)
                print(f'      client の param_version が trainer より新しい = 版採番の不整合。', file=out)
                print(f'      steps {self.neg_lag_steps[0]}..{self.neg_lag_steps[-1]} '
                      f'（resume / trainer 再起動の境界を疑う）', file=out)
        if not self.clip_by_lag:
            print('    no ppo_epoch rows with param_lag', file=out)
            return
        print(f'    {"lag":>5} {"n":>8} {"clip mean":>10} {"clip med":>9} '
              f'{"clip max":>9}   epoch別 clip mean', file=out)
        for lag in sorted(self.clip_by_lag):
            v = self.clip_by_lag[lag]
            per_epoch = []
            for ep in range(1, self.max_epoch + 1):
                w = self.clip_by_lag_epoch.get((lag, ep))
                per_epoch.append(f'e{ep}={st.mean(w):.3f}' if w else f'e{ep}=--')
            print(f'    {lag:>5} {len(v):>8} {st.mean(v):>10.4f} {st.median(v):>9.4f} '
                  f'{max(v):>9.4f}   {" ".join(per_epoch)}', file=out)
        print('    ⚠ lag は param_version の差なので 1 単位 = submit_every step。', file=out)
        print('      lag=0 でも窓内で最大 submit_every-1 step 分ドリフトしており、', file=out)
        print('      staleness の代理変数としては [B] の age を正とする', file=out)

    def report_c(self, out) -> None:
        print('\n[C] clip_fraction / ratio_std の epoch 別（全 lag 込み）', file=out)
        if not self.clip_by_epoch:
            print('    no ppo_epoch rows', file=out)
            return
        for ep in sorted(self.clip_by_epoch):
            v = self.clip_by_epoch[ep]
            r = self.rstd_by_epoch.get(ep) or []
            rtxt = f'{st.mean(r):.4f}' if r else '--'
            print(f'    epoch {ep}: n={len(v):<7} clip mean={st.mean(v):.4f} '
                  f'median={st.median(v):.4f} max={max(v):.4f} | ratio_std mean={rtxt}',
                  file=out)
        eps = sorted(self.clip_by_epoch)
        if len(eps) >= 2:
            d = st.mean(self.clip_by_epoch[eps[-1]]) - st.mean(self.clip_by_epoch[eps[0]])
            print(f'    epoch{eps[0]} → epoch{eps[-1]} の clip mean 差: {d:+.4f}', file=out)
            print('    読み: 差が ~0 なら epoch を重ねても trust region 消費が増えない', file=out)
            print('          = [B] の staleness offset が支配している', file=out)

    def report_d(self, out) -> None:
        print(f'\n[D] clip_fraction / ratio_std の step トレンド（バケット幅 {self.bucket}）', file=out)
        if not self.clip_trend:
            print('    no ppo_epoch rows with trainer_step', file=out)
            return
        for b in sorted(self.clip_trend):
            v = self.clip_trend[b]
            r = self.rstd_trend.get(b) or []
            rtxt = f'{st.mean(r):.4f}' if r else '--'
            print(f'    {b:>6}-{b + self.bucket - 1:<6} n={len(v):<7} '
                  f'clip={st.mean(v):.4f}  ratio_std={rtxt}', file=out)

    def report_e(self, out) -> None:
        print('\n[E] advantage_decomp のカテゴリ別サンプル数 n（局末のみ非ゼロ報酬の帰結）', file=out)
        if not self.adv_n and not self.adv_absent:
            print('    no advantage_decomp rows', file=out)
            return
        for cat in ADV_CATEGORIES:
            ns = self.adv_n.get(cat) or []
            absent = self.adv_absent.get(cat, 0)
            if not ns:
                print(f'    {cat:16s} 全バッチで不在（n レコードなし・不在 {absent} 回）', file=out)
                continue
            p25, med, p75 = quant([float(x) for x in ns])
            print(f'    {cat:16s} バッチ数={len(ns):<7} n: min={min(ns)} p25={p25:.0f} '
                  f'median={med:.0f} p75={p75:.0f} max={max(ns)} mean={st.mean(ns):.2f}'
                  f'   （該当決定なしバッチ {absent} 回）', file=out)

    def report_f(self, out) -> None:
        print(f'\n[F] action_mass トレンド（sampled レンズ・判定非関与、バケット幅 {self.bucket}）',
              file=out)
        if not self.am_trend:
            print('    no action_mass rows', file=out)
            return
        print(f'    {"step":>12}  {"riichi":>7} {"call":>7} {"call|aka":>9} {"selectivity":>11}',
              file=out)
        for b in sorted(self.am_trend):
            r = self.am_trend[b]

            def m(key: str) -> str:
                vs = r.get(key) or []
                return f'{st.mean(vs):.3f}' if vs else '  --'

            print(f'    {b:>6}-{b + self.bucket - 1:<5}  '
                  f'{m("pi_riichi_given_possible"):>7} '
                  f'{m("pi_call_given_possible"):>7} '
                  f'{m("pi_call_given_possible_aka_held"):>9} '
                  f'{m("pi_call_aka_over_no_aka"):>11}', file=out)

    def report_g(self, out) -> None:
        print('\n[G] event の在不在（欠損は 0 埋めせず明示）', file=out)
        for ev in TRACKED_EVENTS:
            n = self.event_counts.get(ev, 0)
            mark = f'{n}' if n else 'no rows（この run では未出力）'
            print(f'    {ev:22s} {mark}', file=out)
        untracked = {k: v for k, v in self.event_counts.items() if k not in TRACKED_EVENTS}
        if untracked:
            print(f'    未集計 event: {untracked}', file=out)
        if self.kl_mean:
            print(f'    kl_ref_mean: n={len(self.kl_mean)} mean={st.mean(self.kl_mean):.6g} '
                  f'min={min(self.kl_mean):.6g} max={max(self.kl_mean):.6g}', file=out)

    def report(self, out) -> None:
        self.header(out)
        self.report_a(out)
        self.report_b(out)
        self.report_c(out)
        self.report_d(out)
        self.report_e(out)
        self.report_f(out)
        self.report_g(out)


def ingest(run_dir: Path, bucket: int) -> RunStats:
    stats = RunStats(run_dir, bucket)
    path = run_dir / 'logs' / 'ppo_diag.jsonl'
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # 走行中 run の末尾は書き込み途中の行があり得る。数えて可視化する
                stats.bad_lines += 1
                continue
            if isinstance(row, dict):
                stats.feed(row)
    stats.finalize()
    return stats


def cross_run_table(all_stats: list[RunStats], out) -> None:
    print(f'\n{"=" * 78}', file=out)
    print('横断サマリ（L1 指標）', file=out)
    print(f'{"=" * 78}', file=out)
    hdr = (f'{"run":36s} {"batch med":>9} {"over mb%":>8} {"clip@step0":>10} '
           f'{"clip e1":>7} {"clip eN":>7} {"e1→eN":>7} {"adv n(call_taken)":>17}')
    print(hdr, file=out)
    print('-' * len(hdr), file=out)
    for s in all_stats:
        bs_med = f'{st.median(s.batch_sizes):.0f}' if s.batch_sizes else '--'
        cfg, _ = config_summary(s.run_dir)
        mb = cfg.get('ppo.minibatch_size')
        if s.batch_sizes and isinstance(mb, int):
            over = f'{sum(1 for x in s.batch_sizes if x > mb) / len(s.batch_sizes):.2%}'
        else:
            over = '--'
        c0 = f'{st.mean(s.clip_at_step0):.4f}' if s.clip_at_step0 else '--'
        eps = sorted(s.clip_by_epoch)
        if eps:
            e1 = f'{st.mean(s.clip_by_epoch[eps[0]]):.4f}'
            en = f'{st.mean(s.clip_by_epoch[eps[-1]]):.4f}'
            de = (f'{st.mean(s.clip_by_epoch[eps[-1]]) - st.mean(s.clip_by_epoch[eps[0]]):+.4f}'
                  if len(eps) >= 2 else '--')
        else:
            e1 = en = de = '--'
        ct = s.adv_n.get('call_taken') or []
        adv = f'median={st.median(ct):.0f}' if ct else '--'
        print(f'{s.name:36s} {bs_med:>9} {over:>8} {c0:>10} {e1:>7} {en:>7} {de:>7} {adv:>17}',
              file=out)
    print('\n※ over mb%  = config の minibatch_size を超えた batch の割合。'
          '0% ならミニバッチ分割は発生していない', file=out)
    print('※ clip@step0 = trainer_step 0（client と trainer が同一パラメータ）の陽性対照。'
          'ここが ~0 で', file=out)
    print('              定常の clip e1 が大きいなら、差は staleness 由来', file=out)
    print('※ clip e1    = epoch 1 の clip_fraction 平均 = **このバッチに勾配を当てる前**の消費量', file=out)
    print('※ e1→eN     = epoch を重ねた増分。~0 なら 4 epoch が trust region を'
          'ほとんど動かしていない', file=out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description='PPO 最適化衛生（L1）の横断診断。読み取り専用・GPU 不要。')
    ap.add_argument('--run', action='append', default=[], metavar='RUN_DIR',
                    help='対象 run dir（複数指定可）。省略時は --root 配下を走査')
    ap.add_argument('--root', default=str(DEFAULT_ROOT),
                    help=f'run を走査する親 dir（既定: {DEFAULT_ROOT}）')
    ap.add_argument('--bucket', type=int, default=500,
                    help='step トレンドのバケット幅（既定 500）')
    ap.add_argument('-o', '--output', help='出力先ファイル（既定 stdout）')
    args = ap.parse_args()

    if args.run:
        run_dirs = [Path(r) for r in args.run]
    else:
        root = Path(args.root)
        if not root.is_dir():
            print(f'FATAL: root が dir でない: {root}', file=sys.stderr)
            return 2
        run_dirs = sorted(d for d in root.iterdir()
                          if (d / 'logs' / 'ppo_diag.jsonl').is_file())

    missing = [d for d in run_dirs if not (d / 'logs' / 'ppo_diag.jsonl').is_file()]
    if missing:
        # サイレントに skip しない
        print('FATAL: ppo_diag.jsonl が無い run が指定された:', file=sys.stderr)
        for d in missing:
            print(f'  {d}', file=sys.stderr)
        return 2
    if not run_dirs:
        print(f'FATAL: 対象 run が 0 件（root={args.root}）', file=sys.stderr)
        return 2

    out = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
    try:
        print('PPO 最適化衛生（L1 層）横断診断', file=out)
        print(f'対象 run: {len(run_dirs)} 件', file=out)
        for d in run_dirs:
            print(f'  {d}', file=out)
        all_stats = []
        for d in run_dirs:
            stats = ingest(d, args.bucket)
            stats.report(out)
            all_stats.append(stats)
        cross_run_table(all_stats, out)
    finally:
        if args.output:
            out.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
