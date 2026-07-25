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

拡張指標 (2026-07-25 追加、`--basic-only` で無効化可)。
**基本3指標の計算経路は無変更** — anchor 系列の判定 (`anchored_ppo_design.md` §6 の
放銃差 z<2) は基本表の houjuu 行を読むため、拡張は必ず加算のみで行う:
  - agari_pt   = 平均和了打点   (libriichi Stat と同一定義: deltas[seat] − 自分の立直棒1000)
  - houjuu_pt  = 平均放銃打点   (同: deltas[seat]。負値)
  - rank1..4   = 順位分布 (半荘ベース)
  - rank_pts   = 1半荘あたり平均順位点 (ウマオカ [+35,+5,−15,−25])
  - sotensu    = 1半荘あたり平均素点 (千点単位・30000返し基準)
  - chip/kyoku = 1局あたり平均チップ枚数

打点・順位分布は libriichi `Stat.from_log` をログ単位で呼んで取得する (定義の複製禁止)。
素点・順位点・チップは analyze_freeparlor_pnl_1v3.process_hanchan を再利用する (同上)。
Stat 由来の agari/houjuu 件数が純 Python 走査の件数と半荘ごとに一致することを
assert する (2実装の突き合わせ = パース乖離の検出)。
拡張指標には `PYTHONPATH=mortal` (libriichi.so) が要る。

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
    open_log,
    process_hanchan,
    reconstruct_final_scores,
    seat_from_filename,
)


def stat_from_log(path: Path, seat: int):
    """libriichi Stat を1半荘分だけ構築する (打点・順位分布の定義を複製しないため)。"""
    try:
        from libriichi.stat import Stat
    except ImportError as e:  # 大声で落とす (サイレントフォールバック禁止)
        raise RuntimeError(
            "拡張指標には libriichi が要る。PYTHONPATH=mortal を設定するか "
            "--basic-only を指定すること"
        ) from e
    with open_log(path) as f:
        text = f.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return Stat.from_log(text, seat)


def scan_extended(path: Path, row: dict) -> dict:
    """基本走査の行に拡張指標を加算する。row は破壊的に更新しない。"""
    seat = seat_from_filename(path)
    st = stat_from_log(path, seat)

    # 2実装の突き合わせ: Stat と純 Python 走査で件数が一致すること
    if st.agari != row["n_agari"] or st.houjuu != row["n_houjuu"] or st.round != row["n_kyoku"]:
        raise RuntimeError(
            f"Stat と純Python走査の件数が不一致 {path}: "
            f"agari {st.agari} vs {row['n_agari']}, houjuu {st.houjuu} vs {row['n_houjuu']}, "
            f"kyoku {st.round} vs {row['n_kyoku']}"
        )

    pnl = process_hanchan(path)
    if pnl.n_kyoku != row["n_kyoku"]:
        raise RuntimeError(
            f"局数が不一致 {path}: pnl {pnl.n_kyoku} vs basic {row['n_kyoku']}"
        )

    return {
        **row,
        "agari_point": float(st.agari_point_ko + st.agari_point_oya),
        "houjuu_point": float(st.houjuu_point_to_ko + st.houjuu_point_to_oya),
        "n_fuuro": int(st.fuuro),
        "n_riichi": int(st.riichi),
        "n_ryukyoku": int(st.ryukyoku),
        "rank_1": int(st.rank_1),
        "rank_2": int(st.rank_2),
        "rank_3": int(st.rank_3),
        "rank_4": int(st.rank_4),
        "sotensu": float(pnl.sotensu),
        "rank_pts": float(pnl.rank_pts),
        "chip_total": float(pnl.chip_total),
    }


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


def scan_dir(d: Path, extended: bool = False) -> list[dict]:
    paths = sorted(d.glob("*.json.gz"))
    if not paths:
        raise RuntimeError(f"no game logs in {d}")
    rows = [scan_hanchan(p) for p in paths]
    if extended:
        rows = [scan_extended(p, r) for p, r in zip(paths, rows)]
    return rows


def rate_and_se(rows: list[dict], num_key: str) -> tuple[float, float]:
    tot_k = sum(r["n_kyoku"] for r in rows)
    tot_e = sum(r[num_key] for r in rows)
    rate = tot_e / tot_k
    var = sum((r[num_key] - rate * r["n_kyoku"]) ** 2 for r in rows) / tot_k**2
    return rate, math.sqrt(var)


def ratio_and_se(rows: list[dict], num_key: str, den_key: str) -> tuple[float, float]:
    """任意の分母に対する比の cluster-robust 推定 (rate_and_se の一般化)。

    r = Σnum_h / Σden_h,  Var(r) = Σ_h (num_h − r·den_h)² / (Σden_h)²
    分母 0 (例: 和了 0 の脚) は推定不能につき大声で落とす。
    """
    tot_d = sum(r[den_key] for r in rows)
    if tot_d == 0:
        raise RuntimeError(f"分母 {den_key} の合計が 0 — 比が定義できない")
    tot_n = sum(r[num_key] for r in rows)
    ratio = tot_n / tot_d
    var = sum((r[num_key] - ratio * r[den_key]) ** 2 for r in rows) / tot_d**2
    return ratio, math.sqrt(var)


def mean_and_se(rows: list[dict], key: str) -> tuple[float, float]:
    """半荘単位の平均と SE (クラスタ = 半荘そのもの)。"""
    vals = [r[key] for r in rows]
    n = len(vals)
    if n < 2:
        raise RuntimeError("半荘が2未満で SE が計算できない")
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) / n
    return mean, math.sqrt(var)


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
    ap.add_argument(
        "--basic-only",
        action="store_true",
        help="拡張指標を出さない(libriichi 不要。判定用の基本3指標のみ)",
    )
    args = ap.parse_args()

    extended = not args.basic_only
    legs = {}
    for name, d in [("init", args.init_logs), (args.label, args.ckpt_logs)]:
        rows = scan_dir(Path(d), extended=extended)
        legs[name] = rows
        print(f"[{name}] {len(rows)} hanchan / {sum(r['n_kyoku'] for r in rows)} kyoku from {d}")

    def emit(title: str, specs: list) -> None:
        print(f"\n=== {title} ===")
        print(f"{'metric':<12} {'init':>18} {args.label:>18} {'diff':>9} {'SE(diff)':>9} {'z':>7}")
        for metric, fn, scale, unit in specs:
            v0, se0 = fn(legs["init"])
            v1, se1 = fn(legs[args.label])
            diff = v1 - v0
            se = math.sqrt(se0**2 + se1**2)
            print(
                f"{metric:<12} {v0*scale:>12.2f}{unit}±{se0*scale:.3f}"
                f" {v1*scale:>12.2f}{unit}±{se1*scale:.3f}"
                f" {diff*scale:>+9.3f} {se*scale:>9.3f} {diff/se:>+7.2f}"
            )

    # 基本3指標: 計算経路は 2026-07-25 の拡張前と同一 (判定はこの表を読む)
    emit(
        "基本指標(判定用・計算経路は不変)",
        [
            ("agari", lambda rows: rate_and_se(rows, "n_agari"), 100.0, "%"),
            ("houjuu", lambda rows: rate_and_se(rows, "n_houjuu"), 100.0, "%"),
            ("avg_rank", rank_and_se, 1.0, " "),
        ],
    )

    if extended:
        emit(
            "拡張指標(参考・判定外)",
            [
                ("fuuro", lambda r: rate_and_se(r, "n_fuuro"), 100.0, "%"),
                ("riichi", lambda r: rate_and_se(r, "n_riichi"), 100.0, "%"),
                ("ryukyoku", lambda r: rate_and_se(r, "n_ryukyoku"), 100.0, "%"),
                ("agari_pt", lambda r: ratio_and_se(r, "agari_point", "n_agari"), 1.0, " "),
                ("houjuu_pt", lambda r: ratio_and_se(r, "houjuu_point", "n_houjuu"), 1.0, " "),
                ("rank1", lambda r: mean_and_se(r, "rank_1"), 100.0, "%"),
                ("rank2", lambda r: mean_and_se(r, "rank_2"), 100.0, "%"),
                ("rank3", lambda r: mean_and_se(r, "rank_3"), 100.0, "%"),
                ("rank4", lambda r: mean_and_se(r, "rank_4"), 100.0, "%"),
                ("rank_pts", lambda r: mean_and_se(r, "rank_pts"), 1.0, " "),
                ("sotensu", lambda r: mean_and_se(r, "sotensu"), 1.0, " "),
                ("chip/kyoku", lambda r: ratio_and_se(r, "chip_total", "n_kyoku"), 1.0, " "),
            ],
        )
        print(
            "\n注: agari_pt/houjuu_pt は点(1000点=1000)、rank_pts/sotensu は千点単位/半荘、"
            "chip/kyoku は枚/局。houjuu_pt は負値が正常。"
        )


if __name__ == "__main__":
    main()
