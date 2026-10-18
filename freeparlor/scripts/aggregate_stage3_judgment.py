#!/usr/bin/env python3
"""Stage3 事前登録判定の集計 (stage3_design.md §4)。

使い方:
  python3 aggregate_stage3_judgment.py \
      /home/gamba/mahjong/runs/ppo/stage3_20260712_033403/logs/ppo_diag.jsonl

集計仕様(実行前固定、Stage2 判定と同一手法):
  - 単一 run・クラッシュなしのため境界規則は不要(全レコード採用)
  - 主指標: π(鳴き|可能∧赤保持)、n_call_possible_aka_held 加重
  - baseline: step 0-200 / 窓: step 8000-16000 (anneal 完了後、正典報酬のみ)
  - トレンド: 窓内 200-step バケット x 40、バケット中心 step に対する加重 OLS
  - 三分岐 (stage3_design.md §4):
      [1] 倍率 >= 2.0x かつ 有意下降トレンド無し (slope/SE > -3)
      [2] 倍率 < 1.0x、または 有意下降トレンドで減衰中 (slope/SE <= -3)
      [3] 倍率 [1.0, 2.0) かつ 有意下降トレンド無し
判定は集計値の印字まで。三分岐の裁定は設計監督側が行う。
"""
import json
import math
import sys

BASE_LO, BASE_HI = 0, 200
WIN_LO, WIN_HI = 8000, 16000
N_BUCKETS = 40
SLOPE_SIG = 3.0


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("event") == "action_mass" and d.get("trainer_step") is not None:
                recs.append(d)
    return recs


def wmean(recs, pi_key, n_key):
    num = den = 0.0
    for d in recs:
        n = d.get(n_key) or 0
        p = d.get(pi_key)
        if n > 0 and p is not None:
            num += p * n
            den += n
    return (num / den if den else float("nan")), int(den)


def in_range(recs, lo, hi):
    return [d for d in recs if lo <= d["trainer_step"] <= hi]


def weighted_ols(xs, ys, ws):
    W = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / W
    ybar = sum(w * y for w, y in zip(ws, ys)) / W
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    dof = len(xs) - 2
    s2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / dof
    se = math.sqrt(s2 / sxx)
    return b, se


def main():
    recs = load(sys.argv[1])
    print(f"action_mass レコード総数: {len(recs)}")

    base = in_range(recs, BASE_LO, BASE_HI)
    win = in_range(recs, WIN_LO, WIN_HI)
    b_mean, b_n = wmean(base, "pi_call_given_possible_aka_held", "n_call_possible_aka_held")
    w_mean, w_n = wmean(win, "pi_call_given_possible_aka_held", "n_call_possible_aka_held")
    ratio = w_mean / b_mean if b_mean else float("nan")
    print("\n=== 主指標 π(鳴き|可能∧赤保持) ===")
    print(f"baseline (step {BASE_LO}-{BASE_HI}): {b_mean:.4f} (n={b_n:,})")
    print(f"窓平均 (step {WIN_LO}-{WIN_HI}):    {w_mean:.4f} (n={w_n:,})")
    print(f"倍率: {ratio:.3f}x  (閾値: 分岐1 >=2.0x / 分岐2 <1.0x)")

    print("\n=== 窓内四半期 (n加重) ===")
    q_means = []
    for i in range(4):
        qlo = WIN_LO + i * 2000
        qhi = qlo + 2000 if i < 3 else WIN_HI
        qm, qn = wmean(in_range(win, qlo, qhi),
                       "pi_call_given_possible_aka_held", "n_call_possible_aka_held")
        q_means.append(qm)
        print(f"Q{i+1} (step {qlo}-{qhi}): {qm:.4f} (n={qn:,}, 対baseline {qm/b_mean:.3f}x)")

    buckets = {}
    for d in win:
        b = min((d["trainer_step"] - WIN_LO) // 200, N_BUCKETS - 1)
        buckets.setdefault(b, []).append(d)
    xs, ys, ws = [], [], []
    for b in sorted(buckets):
        m, n = wmean(buckets[b], "pi_call_given_possible_aka_held", "n_call_possible_aka_held")
        if n > 0 and not math.isnan(m):
            xs.append(WIN_LO + b * 200 + 100)
            ys.append(m)
            ws.append(n)
    slope, se = weighted_ols(xs, ys, ws)
    slope_1k, se_1k = slope * 1000, se * 1000
    t = slope_1k / se_1k
    print("\n=== トレンド (200-stepバケット x " + str(len(xs)) + ", 加重OLS) ===")
    print(f"傾き: {slope_1k:+.5f}/1000step, SE {se_1k:.5f}, slope/SE = {t:+.2f}")

    print("\n=== 三分岐判定材料 (stage3_design.md §4、裁定は監督側) ===")
    down = t <= -SLOPE_SIG
    print(f"[1] 倍率 >= 2.0x かつ 有意下降無し (slope/SE > -{SLOPE_SIG:.0f}): "
          f"倍率{ratio:.3f}x, slope/SE={t:+.2f} -> {'成立' if ratio >= 2.0 and not down else '不成立'}")
    print(f"[2] 倍率 < 1.0x または 有意下降で減衰中 (slope/SE <= -{SLOPE_SIG:.0f}): "
          f"-> {'成立' if ratio < 1.0 or down else '不成立'}")
    print(f"[3] 倍率 [1.0, 2.0) かつ 有意下降無し: "
          f"-> {'成立' if 1.0 <= ratio < 2.0 and not down else '不成立'}")

    rb, rbn = wmean(base, "pi_riichi_given_possible", "n_riichi_possible")
    rw, rwn = wmean(win, "pi_riichi_given_possible", "n_riichi_possible")
    print("\n=== 副次 π(立直|可能) ===")
    print(f"baseline: {rb:.4f} (n={rbn:,}) / 窓平均: {rw:.4f} (n={rwn:,}) / {rw/rb:.2f}x")

    cb, cbn = wmean(base, "pi_call_given_possible", "n_call_possible")
    cw, cwn = wmean(win, "pi_call_given_possible", "n_call_possible")
    print("\n=== 参考 π(鳴き|可能)全体 ===")
    print(f"baseline: {cb:.4f} (n={cbn:,}) / 窓平均: {cw:.4f} (n={cwn:,})")

    print("\n=== 参考: 全期間 500-step バケット推移 (n加重) ===")
    for lo in range(0, 16000, 500):
        m, n = wmean(in_range(recs, lo, lo + 499), "pi_call_given_possible_aka_held",
                     "n_call_possible_aka_held")
        if n:
            print(f"  [{lo:>5},{lo+499:>5}] {m:.4f} (n={n:,})")


if __name__ == "__main__":
    main()
