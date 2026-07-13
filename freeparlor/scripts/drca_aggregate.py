#!/usr/bin/env python3
"""DRCA プローブ: 集計 (drca_probe_design.md §5a-2/§5a-5).

drca_run_probe.py の出力 jsonl (1行=1ロールアウト、branch_index/arm/
reward_primary などを持つ) を消費し、分岐点単位で cluster-robust に
Delta-Q-hat = mean(R_call) - mean(R_nocall) を推定する。

本スクリプトは数値を報告するだけで判定は行わない (drca_probe_design.md
§3: 「層別分析は exploratory と事前明記。主判定は全体平均のみ」)。
閾値との比較・解釈 A/B/C/D への割り当ては設計監督側が §4/§5a-2 に基づき
別途行う。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_rollouts(path: str) -> list[dict]:
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_branch(rows: list[dict]) -> dict[int, dict[str, list[dict]]]:
    out: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r['branch_index']][r['arm']].append(r)
    return out


def branch_delta(call_rows: list[dict], nocall_rows: list[dict]) -> dict:
    call_vals = [r['reward_primary'] for r in call_rows]
    nocall_vals = [r['reward_primary'] for r in nocall_rows]
    if not call_vals or not nocall_vals:
        return None
    mean_call = statistics.fmean(call_vals)
    mean_nocall = statistics.fmean(nocall_vals)
    return {
        'n_call': len(call_vals),
        'n_nocall': len(nocall_vals),
        'mean_call': mean_call,
        'mean_nocall': mean_nocall,
        'delta_q': mean_call - mean_nocall,
    }


def weighted_mean_and_cluster_se(deltas: list[dict]) -> tuple[float, float, int]:
    """n-weighted pooled mean of per-branch-point Delta-Q-hat, with a
    cluster-robust SE computed at the branch-point level (drca_probe_design.md
    §5a-2: "分岐点単位の cluster-robust", since the 2K rollouts within one
    branch point are not independent). Weight per cluster = mean(n_call,
    n_nocall) (rollout count contributed by that branch point).
    """
    if not deltas:
        return float('nan'), float('nan'), 0
    weights = [(d['n_call'] + d['n_nocall']) / 2.0 for d in deltas]
    values = [d['delta_q'] for d in deltas]
    w_sum = sum(weights)
    mean = sum(w * v for w, v in zip(weights, values)) / w_sum
    n = len(values)
    if n < 2:
        return mean, float('nan'), n
    # Weighted variance across clusters (cluster is the independence unit),
    # then SE of the weighted mean via the standard weighted-variance /
    # effective-n formula.
    var = sum(w * (v - mean) ** 2 for w, v in zip(weights, values)) / w_sum
    # effective sample size correction (Kish) so unequal cluster weights
    # don't understate the SE
    eff_n = (w_sum ** 2) / sum(w ** 2 for w in weights)
    se = math.sqrt(var / max(eff_n - 1, 1))
    return mean, se, n


def sign_test(deltas: list[dict]) -> dict:
    pos = sum(1 for d in deltas if d['delta_q'] > 0)
    neg = sum(1 for d in deltas if d['delta_q'] < 0)
    zero = len(deltas) - pos - neg
    n = pos + neg
    if n == 0:
        p_value = float('nan')
    else:
        k = min(pos, neg)
        # exact two-sided binomial sign test, p=0.5
        total = sum(math.comb(n, i) for i in range(0, k + 1))
        p_value = min(1.0, 2 * total / (2 ** n))
    return {'n_pos': pos, 'n_neg': neg, 'n_zero': zero, 'p_value': p_value}


def shanten_bucket(s):
    if s is None:
        return 'unknown'
    if s <= 0:
        return 'tenpai(<=0)'
    if s == 1:
        return '1-shanten'
    return '2+_shanten'


def turn_bucket(t):
    if t is None:
        return 'unknown'
    if t <= 6:
        return 'early(<=6)'
    if t <= 12:
        return 'mid(7-12)'
    return 'late(13+)'


def call_type_label(call_types_available):
    labels = []
    for cid in call_types_available or []:
        labels.append({38: 'chi', 39: 'chi', 40: 'chi', 41: 'pon', 42: 'kan'}.get(cid, str(cid)))
    return '+'.join(sorted(set(labels))) or 'none'


def score_rank_label(rank):
    return f'rank{rank}' if rank is not None else 'unknown'


STRATIFIERS = {
    'shanten': lambda meta: shanten_bucket(meta.get('shanten')),
    'turn': lambda meta: turn_bucket(meta.get('at_turn')),
    'call_type': lambda meta: call_type_label(meta.get('call_types_available')),
    'score_rank': lambda meta: score_rank_label(meta.get('score_rank_at_branch')),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe-out', required=True, nargs='+',
                     help='drca_run_probe.py の --out ファイル (複数可、set(a)/(b) 別ファイルを一括集計する場合など)')
    ap.add_argument('--out', default=None, help='JSON サマリの出力先 (省略時は標準出力のみ)')
    ap.add_argument('--expect-branch-points', type=int, default=None,
                     help='安全弁: 両腕揃った分岐点数がこれ未満なら FATAL exit 1 '
                          '(省略時は検査なし)')
    args = ap.parse_args()

    rows = []
    for p in args.probe_out:
        rows.extend(load_rollouts(p))
    if not rows:
        print('FATAL: no rollout records loaded', file=sys.stderr)
        sys.exit(1)

    grouped = group_by_branch(rows)
    branch_meta = {}
    for r in rows:
        branch_meta.setdefault(r['branch_index'], {
            'shanten': r.get('shanten'),
            'at_turn': r.get('at_turn'),
            'at_kyoku': r.get('at_kyoku'),
            'call_types_available': r.get('call_types_available'),
            'game_key': r.get('game_key'),
            'score_rank_at_branch': r.get('score_rank_at_branch'),
        })

    deltas = []
    for bi, arms in sorted(grouped.items()):
        d = branch_delta(arms.get('call', []), arms.get('no_call', []))
        if d is None:
            print(f'WARNING: branch_index={bi} missing call or no_call rollouts, skipped', file=sys.stderr)
            continue
        d['branch_index'] = bi
        d.update(branch_meta[bi])
        deltas.append(d)

    print(f'branch points with both arms present: {len(deltas)} / {len(grouped)}')

    if args.expect_branch_points is not None and len(deltas) < args.expect_branch_points:
        print(
            f'FATAL: expected >= {args.expect_branch_points} branch points with both '
            f'arms present, got {len(deltas)}', file=sys.stderr,
        )
        sys.exit(1)

    mean, se, n = weighted_mean_and_cluster_se(deltas)
    sign = sign_test(deltas)

    print()
    print('=== PRIMARY (pooled, all branch points, canonical composite reward) ===')
    print(f'  n_branch_points = {n}')
    print(f'  Delta-Q-hat (n-weighted mean) = {mean:.4f} (千点)')
    print(f'  cluster-robust SE             = {se:.4f}')
    if not math.isnan(se) and se > 0:
        print(f'  |Delta-Q-hat| / SE            = {abs(mean) / se:.3f} (2SE threshold reference only, no verdict here)')
    print(f'  sign test: {sign["n_pos"]} positive / {sign["n_neg"]} negative / {sign["n_zero"]} zero '
          f'(two-sided exact p={sign["p_value"]:.4f})')

    summary = {
        'n_branch_points': n,
        'delta_q_hat': mean,
        'cluster_se': se,
        'sign_test': sign,
        'per_branch': deltas,
        'exploratory_strata': {},
    }

    print()
    print('=== EXPLORATORY stratifications (drca_probe_design.md §5a-5; NOT used for any verdict) ===')
    for strat_name, key_fn in STRATIFIERS.items():
        print(f'-- {strat_name} --')
        buckets = defaultdict(list)
        for d in deltas:
            buckets[key_fn(d)].append(d)
        strat_summary = {}
        for bucket, ds in sorted(buckets.items()):
            bmean, bse, bn = weighted_mean_and_cluster_se(ds)
            strat_summary[bucket] = {'n_branch_points': bn, 'delta_q_hat': bmean, 'cluster_se': bse}
            se_str = f'{bse:.4f}' if not math.isnan(bse) else 'n/a (n<2)'
            print(f'  {bucket:20s} n={bn:3d}  Delta-Q-hat={bmean:8.4f}  SE={se_str}')
        summary['exploratory_strata'][strat_name] = strat_summary

    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\nsummary written to {args.out}')


if __name__ == '__main__':
    main()
