#!/usr/bin/env python3
"""基礎技能指標の cluster-robust 有意性パス (1v3 eval game_logs、read-only)。

fundamentals_degradation_diagnosis_20260725.md §3(a) の放銃/和了/着順差に
半荘クラスタの SE を付ける確証プローブ (§7 の CPU 実行可能部分)。
既存の game_logs (eval_grp_baseline_1v3.py 産) を再走査するのみで、
GPU・学習コード・eval 成果物に一切触れない。イベント解釈・最終スコア再構成は
analyze_freeparlor_pnl_1v3.py の実装をそのまま import する (ロジック複製禁止)。

指標定義 (libriichi Stat と同じ分母 = 局数):
  - agari 率  = challenger の hora (actor==seat) 件数 / 局数
  - houjuu 率 = 他家の hora (actor!=seat, target==seat) 件数 / 局数
  - avg_rank  = 半荘ごとの challenger 着順の平均
    (タイブレークは rankings.rs と同一: 降順 stable sort、同点は座席番号が若い方)

SE:
  - 率: 半荘クラスタの ratio-estimator 分散
      r = Σe_h / Σk_h,  Var(r) = Σ_h (e_h − r·k_h)² / (Σk_h)²
  - avg_rank: 半荘平均の通常 SE
  - 差 (ckpt − init): SE(差)² = SE₁² + SE₂² (独立2標本近似)。
    両脚は同一 seed 集合 [10000,10100) を使うため実際には配牌を共有しており
    正の相関がある → この SE は過大 (保守的)。committed な配備税チェック
    (ppo_p3_stage3_result.md) と同じ近似を踏襲する。

検算: 出力の pooled 率が各 stage result md の Stat 由来の数値
(例 stage1 init houjuu 12.10%, agari 20.67%) と一致することを目視確認する。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_freeparlor_pnl_1v3 import (  # noqa: E402
    load_events,
    reconstruct_final_scores,
    seat_from_filename,
)


def scan_hanchan(path: Path) -> dict:
    events = load_events(path)
    seat = seat_from_filename(path)
    names_ev = next(ev for ev in events if ev.get("type") == "start_game")
    if names_ev.get("names", [None] * 4)[seat] != "challenger":
        raise RuntimeError(f"seat {seat} is not challenger in {path}")
    n_kyoku = sum(1 for ev in events if ev.get("type") == "start_kyoku")
    n_agari = 0
    n_houjuu = 0
    for ev in events:
        if ev.get("type") == "hora":
            if ev["actor"] == seat:
                n_agari += 1
            elif ev.get("target") == seat:
                n_houjuu += 1
    final_scores = reconstruct_final_scores(events, path)
    order = sorted(range(4), key=lambda s: (-final_scores[s], s))
    rank = order.index(seat) + 1
    return {"n_kyoku": n_kyoku, "n_agari": n_agari, "n_houjuu": n_houjuu, "rank": rank}


def scan_dir(d: Path) -> list[dict]:
    paths = sorted(d.glob("*.json.gz"))
    if not paths:
        raise RuntimeError(f"no game logs in {d}")
    return [scan_hanchan(p) for p in paths]


def rate_and_se(rows: list[dict], num_key: str) -> tuple[float, float]:
    tot_k = sum(r["n_kyoku"] for r in rows)
    tot_e = sum(r[num_key] for r in rows)
    rate = tot_e / tot_k
    var = sum((r[num_key] - rate * r["n_kyoku"]) ** 2 for r in rows) / tot_k**2
    return rate, math.sqrt(var)


def rank_and_se(rows: list[dict]) -> tuple[float, float]:
    ranks = [r["rank"] for r in rows]
    n = len(ranks)
    mean = sum(ranks) / n
    var = sum((x - mean) ** 2 for x in ranks) / (n - 1) / n
    return mean, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-logs", required=True, help="init(ミラー較正)脚の game_logs dir")
    ap.add_argument("--ckpt-logs", required=True, help="checkpoint 脚の game_logs dir")
    ap.add_argument("--label", default="ckpt")
    args = ap.parse_args()

    legs = {}
    for name, d in [("init", args.init_logs), (args.label, args.ckpt_logs)]:
        rows = scan_dir(Path(d))
        legs[name] = rows
        print(f"[{name}] {len(rows)} hanchan / {sum(r['n_kyoku'] for r in rows)} kyoku from {d}")

    print(f"\n{'metric':<10} {'init':>18} {args.label:>18} {'diff':>9} {'SE(diff)':>9} {'z':>7}")
    for metric, fn, scale in [
        ("agari", lambda rows: rate_and_se(rows, "n_agari"), 100.0),
        ("houjuu", lambda rows: rate_and_se(rows, "n_houjuu"), 100.0),
        ("avg_rank", rank_and_se, 1.0),
    ]:
        v0, se0 = fn(legs["init"])
        v1, se1 = fn(legs[args.label])
        diff = v1 - v0
        se = math.sqrt(se0**2 + se1**2)
        unit = "%" if scale == 100.0 else " "
        print(
            f"{metric:<10} {v0*scale:>12.2f}{unit}±{se0*scale:.3f}"
            f" {v1*scale:>12.2f}{unit}±{se1*scale:.3f}"
            f" {diff*scale:>+9.3f} {se*scale:>9.3f} {diff/se:>+7.2f}"
        )


if __name__ == "__main__":
    main()
