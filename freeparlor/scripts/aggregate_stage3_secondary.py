#!/usr/bin/env python3
"""Stage3 副次指標の集計 (result md 発見節用。判定条件ではない)。

使い方:
  python3 aggregate_stage3_secondary.py \
      /home/gamba/mahjong/runs/ppo/stage3_20260712_033403/logs/ppo_diag.jsonl

フェーズ分割 (stage3_design.md §2 の schedule 準拠、単一 run):
  bonus全開 [0,3999] / anneal [4000,7999] / 判定窓 [8000,16000]
kyoku_reward_decomp / advantage_decomp: セル内 mean を n 加重で合成
(手法は aggregate_stage2_secondary.py と同一)。kyoku_reward_decomp は
正典3ストリームのみを対象とする(ボーナスは第4ストリームとして分離ログ
されており、この decomp には混入しない — mortal/train_ppo.py 参照)。
事前登録済みの副次確認 (stage3_design.md §4): ボーナス期の鳴き局
正典3ストリーム収支が Stage1 の チップ -0.95/局 から改善方向か。
"""
import json
import sys

PHASES = [("bonus全開", 0, 3999), ("anneal", 4000, 7999), ("判定窓", 8000, 16000)]

KYOKU_GROUPS = [("by_riichi", "yes", "立直局"), ("by_riichi", "no", "非立直局"),
                ("by_call", "yes", "鳴き局"), ("by_call", "no", "非鳴き局")]
KYOKU_FIELDS = [("sotensu_mean", "素点"), ("grp_mean", "GRP"), ("chip_mean", "チップ")]
ADV_CLASSES = ["riichi_taken", "riichi_declined", "call_taken", "call_declined"]


def load(path):
    events = {"kyoku_reward_decomp": [], "advantage_decomp": [],
              "grp_calibration": [], "call_bonus": []}
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
            if ev in events and d.get("trainer_step") is not None:
                events[ev].append(d)
    for v in events.values():
        v.sort(key=lambda d: d["trainer_step"])
    return events


def in_win(recs, lo, hi):
    return [d for d in recs if lo <= d["trainer_step"] <= hi]


def agg_kyoku(recs, label):
    print(f"\n=== 局報酬分解 ({label}, 正典3ストリーム, n加重, /局) ===")
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
    ev = load(sys.argv[1])
    print(f"採用レコード: kyoku_reward_decomp={len(ev['kyoku_reward_decomp'])}, "
          f"advantage_decomp={len(ev['advantage_decomp'])}, "
          f"grp_calibration={len(ev['grp_calibration'])}, "
          f"call_bonus={len(ev['call_bonus'])}")

    for name, lo, hi in PHASES:
        agg_kyoku(in_win(ev["kyoku_reward_decomp"], lo, hi), f"{name} step {lo}-{hi}")

    for name, lo, hi in PHASES:
        agg_adv(in_win(ev["advantage_decomp"], lo, hi), f"{name} step {lo}-{hi}")

    print("\n=== call_bonus フェーズ別合計 (第4ストリーム、正典外) ===")
    for name, lo, hi in PHASES:
        recs = in_win(ev["call_bonus"], lo, hi)
        n_applied = sum(d.get("n_applied") or 0 for d in recs)
        total = sum(d.get("bonus_total") or 0.0 for d in recs)
        print(f"{name} [{lo},{hi}]: batches={len(recs)}, n_applied={n_applied:,}, "
              f"bonus_total={total:,.1f} (千点単位)")

    print("\n=== GRP calibration 全推移 ===")
    for d in ev["grp_calibration"]:
        print(f"step {d['trainer_step']:>6,}: mean_abs_rank_err = "
              f"{d['mean_abs_rank_err']:.4f} (n_hanchan={d.get('n_hanchan', '?'):,})")


if __name__ == "__main__":
    main()
