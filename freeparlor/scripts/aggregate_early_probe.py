#!/usr/bin/env python3
"""初期損傷プローブの軌跡集計（read-only）— `early_damage_probe_design.md` §4。

10 個の diag checkpoint それぞれについて既存の解析スクリプトを走らせ、
`anchor_checkpoint_trajectory_20260728.md` と**同じ列**の1枚の表にまとめる。

なぜ集約スクリプトを置くのか: 10 checkpoint × 4 スクリプト = 40 回の実行結果を
人手で表に写すと、写し間違いが1つあっても誰も気づかない。ここを機械化する。

各指標は既存スクリプトの出力をパースして取る（**ロジックを複製しない** —
複製すると判定に使った経路と乖離する）。パースに失敗した指標は 0 埋めせず
**大声で落とす**（サイレントフォールバック禁止）。

注意: 本 run の n=400 は判定標準（n=800）の半分で、**SE は約 1.41 倍**。
`anchor_checkpoint_trajectory_20260728.md` の数値と同じ表に混ぜないこと。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def run_script(script: str, init_logs: Path, ckpt_logs: Path, label: str) -> str:
    out = subprocess.run(
        [sys.executable, str(_HERE / script),
         '--init-logs', str(init_logs), '--ckpt-logs', str(ckpt_logs), '--label', label],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f'{script} failed for {label}:\n{out.stdout}\n{out.stderr}')
    return out.stdout


def grab(text: str, metric: str, script: str, label: str) -> tuple[float, float]:
    """指標行から (ckpt 側の値, z) を取る。見つからなければ落とす。

    2つの表形式があるので**末尾から数える**（先頭からだと形式差でずれる）:
      - `analyze_fundamentals_1v3.py`: init±SE  ckpt±SE  diff  SE  z  → 数値7個
      - genbutsu / agari_composition : init     ckpt     diff  SE  z  → 数値5個
    どちらも末尾3つが diff/SE/z なので、z = nums[-1]、ckpt = nums[-5] または nums[-4]。
    """
    for line in text.splitlines():
        if not line.startswith(metric):
            continue
        nums = re.findall(r'[-+]?\d+\.\d+', line)
        idx = -5 if '±' in line else -4
        if len(nums) < abs(idx):
            continue
        return float(nums[idx]), float(nums[-1])
    raise RuntimeError(f'metric {metric!r} not found in {script} output for {label}')


def read_pnl(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise RuntimeError(f'pnl file missing: {path}')
    d = {}
    for line in path.read_text().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            d[k.strip()] = float(v)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--steps', default='200 400 600 800 1000 1200 1400 1600 1800 1900')
    args = ap.parse_args()

    results = Path(args.run_dir) / 'logs' / 'eval_grp_baseline'
    init_logs = results / 'game_logs_init'
    if not init_logs.is_dir():
        raise RuntimeError(f'init leg missing: {init_logs}')
    init_pnl = read_pnl(results / 'pnl_init.txt')

    rows = []
    for s in args.steps.split():
        label = f'step{s}'
        ckpt_logs = results / f'game_logs_{label}'
        if not ckpt_logs.is_dir():
            raise RuntimeError(f'missing eval logs for {label}: {ckpt_logs}')

        fund = run_script('analyze_fundamentals_1v3.py', init_logs, ckpt_logs, label)
        houjuu, houjuu_z = grab(fund, 'houjuu', 'fundamentals', label)
        agari, agari_z = grab(fund, 'agari ', 'fundamentals', label)
        rank, rank_z = grab(fund, 'avg_rank', 'fundamentals', label)

        gen = run_script('analyze_genbutsu_discipline.py', init_logs, ckpt_logs, label)
        fold, fold_z = grab(gen, '降りの中断率', 'genbutsu', label)

        comp = run_script('diagnose_agari_composition.py', init_logs, ckpt_logs, label)
        riichi_share, riichi_z = grab(comp, '立直和了 割合', 'agari_composition', label)

        pnl = read_pnl(results / f'pnl_{label}.txt')
        chip_diff = pnl['chip_mean'] - init_pnl['chip_mean']
        chip_se = (pnl['chip_se'] ** 2 + init_pnl['chip_se'] ** 2) ** 0.5
        rows.append({
            'step': int(s), 'houjuu': houjuu, 'houjuu_z': houjuu_z,
            'agari': agari, 'rank': rank, 'rank_z': rank_z,
            'fold': fold, 'fold_z': fold_z,
            'riichi_share': riichi_share, 'riichi_z': riichi_z,
            'chip': pnl['chip_mean'], 'chip_se': chip_se,
            'chip_ratio': chip_diff / chip_se if chip_se else float('nan'),
        })
        print(f'  collected {label}', file=sys.stderr)

    print(f"\n初期損傷プローブ 軌跡（run={Path(args.run_dir).name}、n=400、"
          f"SE は判定標準 n=800 の約 1.41 倍）\n")
    print(f"{'step':>5} {'放銃%':>7} {'z':>7} {'和了%':>7} {'avg_rank':>9} {'z':>7} "
          f"{'降り中断%':>9} {'z':>7} {'立直和了%':>9} {'z':>7} {'チップ':>8} {'/SE':>7}")
    print(f"{'0(init)':>5} {init_pnl.get('chip_mean', float('nan')):>7} "
          f"{'—':>7} {'—':>7} {'—':>9} {'—':>7} {'—':>9} {'—':>7} {'—':>9} {'—':>7} "
          f"{init_pnl['chip_mean']:>+8.3f} {'—':>7}")
    for r in rows:
        print(f"{r['step']:>5} {r['houjuu']:>7.2f} {r['houjuu_z']:>+7.2f} "
              f"{r['agari']:>7.2f} {r['rank']:>9.4f} {r['rank_z']:>+7.2f} "
              f"{r['fold']:>9.2f} {r['fold_z']:>+7.2f} "
              f"{r['riichi_share']:>9.2f} {r['riichi_z']:>+7.2f} "
              f"{r['chip']:>+8.3f} {r['chip_ratio']:>+7.2f}")

    print('\n注: z は init 基準（半荘クラスタ SE）。チップ /SE は init との差 / SE(差)。')
    print('    n=400 なので anchor_checkpoint_trajectory_20260728.md（n=800）と'
          '同じ表に混ぜないこと。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
