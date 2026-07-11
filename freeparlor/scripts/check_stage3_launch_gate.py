#!/usr/bin/env python3
"""Stage3 発進ゲート v2 (stage3_design.md §3, 2026-07-11 amendment) 集計。
事前登録済み・算出方法変更禁止（v1 の step 500 単一ゲートは較正ミスと裁定され、
機械ゲート(@200)/学習応答ゲート(@2000) の二段に分離された — 本書 §3 参照）。

使い方:
  python3 check_stage3_launch_gate.py <path/to/ppo_diag.jsonl> --gate mechanical
  python3 check_stage3_launch_gate.py <path/to/ppo_diag.jsonl> --gate learning_response

算出方法(実行前固定):
  機械ゲート (@step200, stage3_design.md §3):
    - call_bonus イベントで全バッチ b == 5.0 (schedule 整合)
    - trainer_step in [0, 200] で n_applied > 0 のバッチ割合 >= 50%
    未達 = 配管故障 -> 即 run 停止・実装調査

  学習応答ゲート (@step2000, stage3_design.md §3):
    - baseline = trainer_step in [0, 200] の pi_call_given_possible_aka_held の
      n_call_possible_aka_held 加重平均
    - gate値   = trainer_step in [1801, 2000] の同・加重平均
    - 判定: gate値 >= 2.0 * baseline で通過
    - 較正チェック: baseline が [0.20, 0.31] の外なら配管疑いで run 停止・報告
      (Stage1 実測 0.2462・v1 run 実測 0.2702 と整合するはず)
"""
import argparse
import json
import sys

MECH_LO, MECH_HI = 0, 200
MECH_B_EXPECTED = 5.0
MECH_N_APPLIED_FRAC_THRESHOLD = 0.5

BASE_LO, BASE_HI = 0, 200
GATE_LO, GATE_HI = 1801, 2000
THRESHOLD = 2.0
CALIB_LO, CALIB_HI = 0.20, 0.31


def load(path, event):
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
            if d.get('event') == event and d.get('trainer_step') is not None:
                recs.append(d)
    return recs


def in_range(recs, lo, hi):
    return [d for d in recs if lo <= d['trainer_step'] <= hi]


def wmean(recs, pi_key, n_key):
    num = den = 0.0
    for d in recs:
        n = d.get(n_key) or 0
        p = d.get(pi_key)
        if n > 0 and p is not None:
            num += p * n
            den += n
    return (num / den if den else float('nan')), int(den)


def run_mechanical(path):
    cb = load(path, 'call_bonus')
    print(f"call_bonus レコード総数: {len(cb)}")

    window = in_range(cb, MECH_LO, MECH_HI)
    print(f"\n=== Stage3 機械ゲート (@step{MECH_HI}) ===")
    print(f"窓 [{MECH_LO}, {MECH_HI}] のバッチ数: {len(window)}")

    if not window:
        print("窓内に call_bonus レコードが無い -> FAIL (配管故障)")
        sys.exit(1)

    b_values = {d['b'] for d in window}
    b_ok = b_values == {MECH_B_EXPECTED}
    print(f"b の値集合: {sorted(b_values)} (期待 {{{MECH_B_EXPECTED}}}) -> {'OK' if b_ok else 'NG'}")

    n_applied_pos = sum(1 for d in window if (d.get('n_applied') or 0) > 0)
    frac = n_applied_pos / len(window)
    frac_ok = frac >= MECH_N_APPLIED_FRAC_THRESHOLD
    print(f"n_applied>0 のバッチ割合: {frac:.3f} ({n_applied_pos}/{len(window)}) "
          f"(閾値 >= {MECH_N_APPLIED_FRAC_THRESHOLD}) -> {'OK' if frac_ok else 'NG'}")

    mech_pass = b_ok and frac_ok
    print(f"\n機械ゲート判定: {'通過' if mech_pass else '未達 (run停止・実装調査)'}")

    if not mech_pass:
        sys.exit(1)


def run_learning_response(path):
    am = load(path, 'action_mass')
    print(f"action_mass レコード総数: {len(am)}")

    base = in_range(am, BASE_LO, BASE_HI)
    gate = in_range(am, GATE_LO, GATE_HI)

    b_mean, b_n = wmean(base, 'pi_call_given_possible_aka_held', 'n_call_possible_aka_held')
    g_mean, g_n = wmean(gate, 'pi_call_given_possible_aka_held', 'n_call_possible_aka_held')
    ratio = g_mean / b_mean if b_mean and b_mean == b_mean else float('nan')

    print("\n=== Stage3 学習応答ゲート π(鳴き|可能∧赤保持) (@step2000) ===")
    print(f"baseline (step {BASE_LO}-{BASE_HI}): {b_mean:.4f} (n={b_n:,})")
    print(f"gate値   (step {GATE_LO}-{GATE_HI}): {g_mean:.4f} (n={g_n:,})")
    print(f"倍率: {ratio:.3f}x  (閾値 {THRESHOLD}x)")

    calib_ok = (b_mean == b_mean) and (CALIB_LO <= b_mean <= CALIB_HI)
    print(f"\n較正チェック: baseline in [{CALIB_LO}, {CALIB_HI}] -> "
          f"{'OK' if calib_ok else 'NG (配管疑い、run停止・報告)'}")

    gate_pass = (ratio == ratio) and ratio >= THRESHOLD
    print(f"学習応答ゲート判定: 倍率 >= {THRESHOLD}x -> {'通過' if gate_pass else '未達 (run停止・報告)'}")

    if not calib_ok or not gate_pass:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('diag_path')
    parser.add_argument('--gate', choices=['mechanical', 'learning_response'], required=True)
    args = parser.parse_args()

    if args.gate == 'mechanical':
        run_mechanical(args.diag_path)
    else:
        run_learning_response(args.diag_path)


if __name__ == '__main__':
    main()
