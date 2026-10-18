#!/usr/bin/env python3
"""Stage2 事前登録判定の集計 (stage2_design.md §4)。

使い方:
  python3 aggregate_stage2_judgment.py \
      /home/gamba/mahjong/runs/ppo/stage2_20260709_092541/logs/ppo_diag.jsonl \
      /home/gamba/mahjong/runs/ppo/stage2_20260709_194510_resume/logs/ppo_diag.jsonl

境界規則(実行前固定):
  - 旧 run: trainer_step <= 6000 を採用(6001-7339 はクラッシュ破棄枝として除外)
  - resume: trainer_step >= 6001 を採用(6000 は境界重複可能性のため除外・件数記録)
  - baseline: step 0-200 (旧 run 由来)、窓: step 8000-16000 (resume 由来)
  - バケット: 窓内 200-step x 40本、bucket = min((step-8000)//200, 39)
判定は集計値の印字まで。三分岐の裁定は設計監督側が行う。
"""
import json
import math
import sys

ORIG_MAX = 6000       # 旧 run はこの step まで採用(破棄枝除外)
RESUME_MIN = 6001     # resume はこの step から採用
BASE_LO, BASE_HI = 0, 200
WIN_LO, WIN_HI = 8000, 16000
N_BUCKETS = 40


def load(path, lo, hi):
    """[lo, hi] の action_mass レコードを返す。範囲外は excluded に計上。"""
    recs, excluded, samples = [], 0, {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            ev = d.get("event")
            if ev != "action_mass":
                # 未集計イベントのサンプルを各種1件だけ保存
                if ev not in samples and ev != "ppo_epoch" and ev != "batch_lag":
                    samples[ev] = d
                continue
            s = d.get("trainer_step")
            if s is None:
                continue
            if lo <= s <= hi:
                recs.append(d)
            else:
                excluded += 1
    return recs, excluded, samples


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
    """加重 OLS y = a + b x。slope b と SE(b) を返す。"""
    W = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / W
    ybar = sum(w * y for w, y in zip(ws, ys)) / W
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    # 加重残差分散 (dof = len - 2)
    dof = len(xs) - 2
    s2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / dof
    se = math.sqrt(s2 / sxx)
    return b, se


def main():
    orig_path, resume_path = sys.argv[1], sys.argv[2]

    orig, orig_excl, samples_o = load(orig_path, 0, ORIG_MAX)
    resume, resume_excl, samples_r = load(resume_path, RESUME_MIN, 10**9)
    print(f"旧run採用 action_mass: {len(orig)} (破棄枝など除外: {orig_excl})")
    print(f"resume採用 action_mass: {len(resume)} (境界step6000など除外: {resume_excl})")

    merged = orig + resume

    # --- 主指標: π(鳴き|可能∧赤保持) ---
    base = in_range(merged, BASE_LO, BASE_HI)
    win = in_range(merged, WIN_LO, WIN_HI)
    b_mean, b_n = wmean(base, "pi_call_given_possible_aka_held", "n_call_possible_aka_held")
    w_mean, w_n = wmean(win, "pi_call_given_possible_aka_held", "n_call_possible_aka_held")
    ratio = w_mean / b_mean if b_mean else float("nan")
    print("\n=== 主指標 π(鳴き|可能∧赤保持) ===")
    print(f"baseline (step {BASE_LO}-{BASE_HI}): {b_mean:.4f} (n={b_n:,})")
    print(f"窓平均 (step {WIN_LO}-{WIN_HI}):    {w_mean:.4f} (n={w_n:,})")
    print(f"倍率: {ratio:.3f}x  (閾値 2.0x)")

    # --- 四半期 ---
    print("\n=== 窓内四半期 (n加重) ===")
    q_means = []
    for i in range(4):
        qlo = WIN_LO + i * 2000
        qhi = qlo + 2000 if i < 3 else WIN_HI
        qm, qn = wmean(in_range(win, qlo, qhi),
                       "pi_call_given_possible_aka_held", "n_call_possible_aka_held")
        q_means.append(qm)
        print(f"Q{i+1} (step {qlo}-{qhi}): {qm:.4f} (n={qn:,}, 対baseline {qm/b_mean:.3f}x)")

    # --- トレンド: 200-stepバケット加重OLS ---
    buckets = {}
    for d in win:
        b = min((d["trainer_step"] - WIN_LO) // 200, N_BUCKETS - 1)
        buckets.setdefault(b, []).append(d)
    xs, ys, ws = [], [], []
    for b in sorted(buckets):
        m, n = wmean(buckets[b], "pi_call_given_possible_aka_held", "n_call_possible_aka_held")
        if n > 0 and not math.isnan(m):
            xs.append(WIN_LO + b * 200 + 100)  # バケット中心step
            ys.append(m)
            ws.append(n)
    slope, se = weighted_ols(xs, ys, ws)
    slope_1k, se_1k = slope * 1000, se * 1000
    print("\n=== トレンド (200-stepバケット x " + str(len(xs)) + ", 加重OLS) ===")
    print(f"傾き: {slope_1k:+.5f}/1000step, SE {se_1k:.5f}, slope/SE = {slope_1k/se_1k:+.2f}")

    # --- 三分岐判定材料 ---
    print("\n=== 三分岐判定材料 (裁定は監督側) ===")
    print(f"[1] 倍率 >= 2.0x: {ratio:.3f}x -> {'成立' if ratio >= 2.0 else '不成立'}")
    up = slope_1k / se_1k >= 3.0
    print(f"[2] 倍率 < 2.0x かつ 上昇トレンド無し(slope/SE < +3): "
          f"{'成立' if ratio < 2.0 and not up else '不成立'}")
    q4_ratio = q_means[3] / b_mean if b_mean else float("nan")
    print(f"[3] 倍率 [1.0,2.0) かつ slope/SE >= +3 かつ Q4単独 >= 2x: "
          f"倍率{1.0 <= ratio < 2.0}, slope/SE>=+3 {up}, Q4 {q4_ratio:.3f}x "
          f"-> {'成立' if (1.0 <= ratio < 2.0 and up and q4_ratio >= 2.0) else '不成立'}")

    # --- 副次: π(立直|可能) ---
    rb, rbn = wmean(base, "pi_riichi_given_possible", "n_riichi_possible")
    rw, rwn = wmean(win, "pi_riichi_given_possible", "n_riichi_possible")
    print("\n=== 副次 π(立直|可能) ===")
    print(f"baseline: {rb:.4f} (n={rbn:,}) / 窓平均: {rw:.4f} (n={rwn:,}) "
          f"/ {rw/rb:.2f}x")

    # --- 参考: π(鳴き|可能) 全体 ---
    cb, cbn = wmean(base, "pi_call_given_possible", "n_call_possible")
    cw, cwn = wmean(win, "pi_call_given_possible", "n_call_possible")
    print(f"\n=== 参考 π(鳴き|可能)全体 ===")
    print(f"baseline: {cb:.4f} (n={cbn:,}) / 窓平均: {cw:.4f} (n={cwn:,})")

    # --- 未集計イベントのスキーマダンプ ---
    print("\n=== 未集計イベントのサンプル (スキーマ確認用, 次ラウンドで扱う) ===")
    for src, samples in (("旧run", samples_o), ("resume", samples_r)):
        for ev, d in samples.items():
            print(f"--- {src} {ev} ---")
            print(json.dumps(d, ensure_ascii=False, indent=1)[:800])


if __name__ == "__main__":
    main()
