#!/usr/bin/env python3
"""Stage2 副次指標の集計 (result md 発見節用。判定条件ではない)。

使い方:
  python3 aggregate_stage2_secondary.py \
      /home/gamba/mahjong/runs/ppo/stage2_20260709_092541/logs/ppo_diag.jsonl \
      /home/gamba/mahjong/runs/ppo/stage2_20260709_194510_resume/logs/ppo_diag.jsonl

境界規則は主判定スクリプトと同一(旧run <=6000, resume >=6001)。
窓 = step 8000-16000、baseline = step 0-200 (参考値、n小)。
kyoku_reward_decomp / advantage_decomp: セル内 mean を n 加重で合成。
grp_calibration: 全レコードをそのまま時系列で印字。
"""
import json
import sys

ORIG_MAX = 6000
RESUME_MIN = 6001
BASE_LO, BASE_HI = 0, 200
WIN_LO, WIN_HI = 8000, 16000

KYOKU_GROUPS = [("by_riichi", "yes", "立直局"), ("by_riichi", "no", "非立直局"),
                ("by_call", "yes", "鳴き局"), ("by_call", "no", "非鳴き局")]
KYOKU_FIELDS = [("sotensu_mean", "素点"), ("grp_mean", "GRP"), ("chip_mean", "チップ")]
ADV_CLASSES = ["riichi_taken", "riichi_declined", "call_taken", "call_declined"]


def load(paths):
    events = {"kyoku_reward_decomp": [], "advantage_decomp": [], "grp_calibration": []}
    for path, lo, hi in paths:
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
                if ev not in events:
                    continue
                s = d.get("trainer_step")
                if s is None or not (lo <= s <= hi):
                    continue
                events[ev].append(d)
    for v in events.values():
        v.sort(key=lambda d: d["trainer_step"])
    return events


def in_win(recs, lo, hi):
    return [d for d in recs if lo <= d["trainer_step"] <= hi]


def agg_kyoku(recs, label):
    print(f"\n=== 局報酬分解 ({label}, n加重, /局) ===")
    print(f"{'条件':<8} {'n':>9} {'素点':>8} {'GRP':>8} {'チップ':>8}")
    for group, key, name in KYOKU_GROUPS:
        n_tot = 0
        sums = {f: 0.0 for f, _ in KYOKU_FIELDS}
        for d in recs:
            cell = (d.get(group) or {}).get(key)
            if not cell:
                continue
            n = cell.get("n") or 0
            if n <= 0:
                continue
            n_tot += n
            for f, _ in KYOKU_FIELDS:
                v = cell.get(f)
                if v is not None:
                    sums[f] += v * n
        if n_tot:
            vals = [sums[f] / n_tot for f, _ in KYOKU_FIELDS]
            print(f"{name:<8} {n_tot:>9,} {vals[0]:>+8.2f} {vals[1]:>+8.2f} {vals[2]:>+8.2f}")
        else:
            print(f"{name:<8} {'0':>9}       -        -        -")


def agg_adv(recs, label):
    print(f"\n=== advantage 分解 ({label}, n加重) ===")
    print(f"{'クラス':<16} {'raw':>8} {'norm':>8} {'n':>10}")
    for cls in ADV_CLASSES:
        n_tot = 0
        raw_sum = norm_sum = 0.0
        for d in recs:
            raw = (d.get("raw") or {}).get(cls)
            norm = (d.get("norm") or {}).get(cls)
            if not raw or not norm:
                continue
            n = raw.get("n") or 0
            if n <= 0:
                continue
            n_tot += n
            raw_sum += (raw.get("mean") or 0.0) * n
            norm_sum += (norm.get("mean") or 0.0) * n
        if n_tot:
            print(f"{cls:<16} {raw_sum/n_tot:>+8.3f} {norm_sum/n_tot:>+8.3f} {n_tot:>10,}")
        else:
            print(f"{cls:<16} {'-':>8} {'-':>8} {'0':>10}")


def main():
    orig_path, resume_path = sys.argv[1], sys.argv[2]
    ev = load([(orig_path, 0, ORIG_MAX), (resume_path, RESUME_MIN, 10**9)])
    n_k, n_a, n_g = (len(ev["kyoku_reward_decomp"]), len(ev["advantage_decomp"]),
                     len(ev["grp_calibration"]))
    print(f"採用レコード: kyoku_reward_decomp={n_k}, advantage_decomp={n_a}, "
          f"grp_calibration={n_g}")

    win_k = in_win(ev["kyoku_reward_decomp"], WIN_LO, WIN_HI)
    base_k = in_win(ev["kyoku_reward_decomp"], BASE_LO, BASE_HI)
    agg_kyoku(win_k, f"窓 step {WIN_LO}-{WIN_HI}")
    agg_kyoku(base_k, f"baseline step {BASE_LO}-{BASE_HI} 参考・n小")

    win_a = in_win(ev["advantage_decomp"], WIN_LO, WIN_HI)
    agg_adv(win_a, f"窓 step {WIN_LO}-{WIN_HI}")

    print("\n=== GRP calibration 全推移 ===")
    for d in ev["grp_calibration"]:
        print(f"step {d['trainer_step']:>6,}: mean_abs_rank_err = "
              f"{d['mean_abs_rank_err']:.4f} (n_hanchan={d.get('n_hanchan', '?'):,})")


if __name__ == "__main__":
    main()
