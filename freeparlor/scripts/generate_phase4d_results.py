#!/usr/bin/env python3
"""Generate phase4d_sweep_results.md from configs, test_play, and aka-conditional analysis."""
import argparse
import subprocess
import sys
from pathlib import Path

import toml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mortal"))
sys.path.insert(0, str(ROOT / "freeparlor" / "scripts"))

from analyze_aka_conditional import ModelStats, analyze_dir, avg_aka_discard_turn, delta_pp, rate
from libriichi.stat import Stat

CFG_DIR = ROOT / "freeparlor" / "configs"
RUNS = Path("/home/gamba/mahjong/runs/phase4d")
EVAL = RUNS / "eval"
OUT_MD = ROOT / "freeparlor" / "docs" / "phase4d_sweep_results.md"

MODELS = [
    ("lo=0.0", "phase4d_lo00", CFG_DIR / "phase4d_lo00_beta0_3_192x40.toml"),
    ("lo=0.3", "phase4d_lo03", CFG_DIR / "phase4d_lo03_beta0_3_192x40.toml"),
    ("lo=0.6", "phase4d_lo06", CFG_DIR / "phase4d_lo06_beta0_3_192x40.toml"),
]

# Phase 4c documented human deltas (2009, 6897 files)
HUMAN = {
    "fuuro": +2.81,
    "riichi": +3.13,
    "ryukyoku": +4.08,
    "houjuu": None,
    "noten_end": None,
    "avg_aka_discard_turn": None,
}

# First md publish (pre Step-1 population check; same numbers, source unverified)
ARCHIVED_AKA_DELTA = {
    "lo=0.0": {"call": +0.52, "riichi": +4.74, "houjuu": +0.32, "ryukyoku": +3.66},
    "lo=0.3": {"call": -5.01, "riichi": +11.57, "houjuu": +0.80, "ryukyoku": +4.16},
    "lo=0.6": {"call": +2.10, "riichi": +4.99, "houjuu": -1.04, "ryukyoku": +4.02},
}
ARCHIVED_AKA_MONITOR = {
    "lo=0.0": {"noten_end_rate": 57.40, "avg_aka_discard_turn": 10.04},
    "lo=0.3": {"noten_end_rate": 52.40, "avg_aka_discard_turn": 10.59},
    "lo=0.6": {"noten_end_rate": 54.79, "avg_aka_discard_turn": 10.03},
}

# 1v3 summary fuuro rates used for Step-1 cross-check
EXPECTED_FUURO = {
    "phase4d_lo00": 11.95,
    "phase4d_lo03": 17.59,
    "phase4d_lo06": 20.29,
}


def load_env(cfg_path: Path) -> dict:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = toml.load(f)
    env = cfg.get("env", {})
    ds = cfg.get("dataset", {})
    ctrl = cfg.get("control", {})
    return {
        "alpha": env.get("alpha"),
        "gamma_pt": env.get("gamma_pt"),
        "beta": env.get("beta"),
        "lambda_opp": env.get("lambda_opp", 0.0),
        "noten_factor": env.get("noten_factor", 0.0),
        "file_index": ds.get("file_index"),
        "num_epochs": ds.get("num_epochs"),
        "seed_key": cfg.get("1v3", {}).get("seed_key"),
        "batch_size": ctrl.get("batch_size"),
        "conv_channels": cfg.get("resnet", {}).get("conv_channels"),
        "num_blocks": cfg.get("resnet", {}).get("num_blocks"),
    }


def fmt_pp(v: float | None) -> str:
    if v is None or v != v:
        return "—"
    return f"{v:+.2f}"


def fmt_rate(v: float | None) -> str:
    if v is None or v != v:
        return "—"
    return f"{v:.2f}%"


def stat_dict(log_dir: Path) -> dict:
    stat = Stat.from_dir(str(log_dir), "mortal")
    return {
        "avg_rank": stat.avg_rank,
        "agari_rate": stat.agari_rate * 100,
        "houjuu_rate": stat.houjuu_rate * 100,
        "fuuro_rate": stat.fuuro_rate * 100,
        "riichi_rate": stat.riichi_rate * 100,
        "ryukyoku_rate": stat.ryukyoku_rate * 100,
    }


def collect_train_test_play(name: str) -> dict:
    log_dir = RUNS / name / "test_play"
    if not log_dir.exists():
        return {}
    return stat_dict(log_dir)


def collect_1v3_play(name: str) -> dict:
    log_dir = EVAL / name / "1v3"
    if not log_dir.exists():
        return {}
    return stat_dict(log_dir)


def collect_aka(name: str) -> ModelStats | None:
    log_dir = EVAL / name / "1v3"
    if not log_dir.exists():
        return None
    return analyze_dir(log_dir)


def verify_aka_population() -> list[dict]:
    rows = []
    for label, run, _ in MODELS:
        log_dir = EVAL / run / "1v3"
        exp = EXPECTED_FUURO[run]
        stat = Stat.from_dir(str(log_dir), "mortal")
        s = analyze_dir(log_dir)
        stat_fuuro = stat.fuuro_rate * 100
        aka_fuuro = rate(s.total.call, s.total.rounds)
        rows.append(
            {
                "label": label,
                "run": run,
                "log_dir": str(log_dir),
                "stat_fuuro": stat_fuuro,
                "aka_fuuro": aka_fuuro,
                "rounds": s.total.rounds,
                "expected": exp,
                "match": abs(stat_fuuro - exp) < 0.05 and abs(aka_fuuro - exp) < 0.05,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-md", action="store_true")
    args = ap.parse_args()

    envs = {label: load_env(cfg) for label, _, cfg in MODELS}
    train_test_stats = {label: collect_train_test_play(run) for label, run, _ in MODELS}
    play_stats = {label: collect_1v3_play(run) for label, run, _ in MODELS}
    aka_stats = {label: collect_aka(run) for label, run, _ in MODELS}
    pop_check = verify_aka_population()
    if not all(r["match"] for r in pop_check):
        bad = [r["run"] for r in pop_check if not r["match"]]
        raise SystemExit(f"aka-conditional population mismatch: {bad}")

    rows_delta = [
        ("副露率 Δ", "call", "fuuro"),
        ("立直率 Δ", "riichi", "riichi"),
        ("放銃率 Δ", "houjuu", "houjuu"),
        ("流局率 Δ", "ryukyoku", "ryukyoku"),
    ]

    ai_delta: dict[str, dict[str, float]] = {}
    ai_monitor: dict[str, dict[str, float]] = {}
    for label, run, _ in MODELS:
        s = aka_stats[label]
        ai_delta[label] = {}
        ai_monitor[label] = {}
        if s is None:
            continue
        wa, wo = s.with_aka, s.without_aka
        for _, attr, _ in rows_delta:
            ai_delta[label][attr] = delta_pp(wa, wo, attr)
        ai_monitor[label]["noten_end_rate"] = rate(wa.noten_end, wa.rounds)
        ai_monitor[label]["avg_aka_discard_turn"] = avg_aka_discard_turn(wa)

    lines = [
        "# Phase 4d — lambda_opp スイープ結果",
        "",
        "## Config 確認（3 本共通 vs 差分）",
        "",
        "| 項目 | lo=0.0 | lo=0.3 | lo=0.6 |",
        "|---|---:|---:|---:|",
    ]
    keys = [
        ("beta", "beta"),
        ("noten_factor", "noten_factor"),
        ("alpha", "alpha"),
        ("gamma_pt", "gamma_pt"),
        ("lambda_opp", "lambda_opp"),
        ("file_index", "file_index"),
        ("num_epochs", "num_epochs"),
        ("seed_key (1v3)", "seed_key"),
        ("batch_size", "batch_size"),
        ("resnet", lambda e: f"{e['conv_channels']}×{e['num_blocks']}"),
    ]
    for label, key in keys:
        if callable(key):
            vals = [str(key(envs[l])) for l, _, _ in MODELS]
            flag = " ✓" if len(set(vals)) == 1 else ""
        else:
            vals = [str(envs[l][key]) for l, _, _ in MODELS]
            flag = " ✓" if len(set(vals)) == 1 or key == "lambda_opp" else ""
        lines.append(f"| {label}{flag} | {' | '.join(vals)} |")

    lines.extend(
        [
            "",
            "## aka-conditional 母集団照合（Step 1）",
            "",
            "集計スクリプト `analyze_aka_conditional.analyze_dir` の読み込み先 = 1v3 サマリと同一の `eval/<run>/1v3/`。",
            "全体副露率が 1v3 サマリと一致することを run ごとに確認。",
            "",
            "| run | ログ | rounds | Stat fuuro | analyze_dir fuuro | 1v3 期待 | 一致 |",
            "|---|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for r in pop_check:
        lines.append(
            f"| {r['label']} | `{r['log_dir']}` | {r['rounds']} | "
            f"{r['stat_fuuro']:.2f}% | {r['aka_fuuro']:.2f}% | {r['expected']:.2f}% | "
            f"{'✓' if r['match'] else '✗'} |"
        )

    lines.extend(
        [
            "",
            "## 赤条件別 Δ（赤あり − 赤なし, pp）",
            "",
            "集計元: `eval/phase4d_lo{00,03,06}/1v3/`（上表照合済み・mortal 席のみ）。",
            "",
        ]
    )
    header = "| 指標 | 人間(2009) | AI lo=0.0 | AI lo=0.3 | AI lo=0.6 |"
    lines.extend([header, "|---|---:|---:|---:|---:|"])
    for row_label, attr, human_key in rows_delta:
        row = f"| {row_label} | {fmt_pp(HUMAN[human_key])} |"
        for label, _, _ in MODELS:
            v = ai_delta.get(label, {}).get(attr)
            row += f" {fmt_pp(v)} |"
        lines.append(row)

    lines.extend(["", "## Monitor（1v3 自己対戦・aka-conditional）", ""])
    lines.append("集計元: 上記赤条件別 Δ と同一ログ。")
    lines.append("")
    lines.extend([header, "|---|---:|---:|---:|---:|"])
    monitor_rows = [
        ("赤保持ノーテン率", "noten_end_rate", True),
        ("赤平均切り順", "avg_aka_discard_turn", False),
    ]
    for row_label, key, is_pct in monitor_rows:
        row = f"| {row_label} | — |"
        for label, _, _ in MODELS:
            v = ai_monitor.get(label, {}).get(key)
            if is_pct:
                row += f" {fmt_rate(v)} |"
            else:
                row += f" {v:.2f} |" if v == v else " — |"
        lines.append(row)

    lines.extend([
        "",
        "## 評価卓構成（Phase4 `sweep_eval` と同方式）",
        "",
        "| 評価 | 1席 (challenger) | 3席 (champion) | ログ / 集計対象 |",
        "|---|---|---|---|",
        "| **1v3 自己対戦** | 学習済み `model.pth` (`name=mortal`) | 同一 `model.pth` を `eval/<run>/champion.pth` にコピー (`name=baseline`) | `eval/<run>/1v3/` · `Stat.from_dir(..., 'mortal')` · aka-conditional |",
        "| 学習時 test_play | 学習中 `mortal` | 固定 `grp_baseline.pth` (`[baseline.test]`) | `runs/phase4d/<run>/test_play/` · avg_rank≈1.0（参考外） |",
        "",
        "1v3 起動: `run_sweep_eval.py` → `champion.pth` に `model.pth` をコピー → `run_one_vs_three.py` → `py_vs_py(challenger, champion)`。",
        "",
        "## 1v3 自己対戦サマリ（打牌統計・Phase4 同条件）",
        "",
        "seed_key=42, games=400, mortal 席のみ集計。",
        "",
        "| run | avg_rank | 和了率 | 放銃率 | 副露率 | 立直率 | 流局率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, run, _ in MODELS:
        tp = play_stats[label]
        if not tp:
            lines.append(f"| {label} | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {tp['avg_rank']:.3f} | {tp['agari_rate']:.2f}% | "
            f"{tp['houjuu_rate']:.2f}% | {tp['fuuro_rate']:.2f}% | "
            f"{tp['riichi_rate']:.2f}% | {tp['ryukyoku_rate']:.2f}% |"
        )

    lines.extend([
        "",
        "## （参考外）学習時 test_play — `grp_baseline.pth` 固定3席",
        "",
        "学習ループ内 `TestPlayer.test_play`: 1席=学習中モデル、3席=`/home/gamba/mahjong/runs/grp_baseline.pth`。自己対戦ではない。",
        "",
        "| run | avg_rank | 和了率 | 放銃率 | 副露率 | 立直率 | 流局率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, run, _ in MODELS:
        tp = train_test_stats[label]
        if not tp:
            lines.append(f"| {label} | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {tp['avg_rank']:.3f} | {tp['agari_rate']:.2f}% | "
            f"{tp['houjuu_rate']:.2f}% | {tp['fuuro_rate']:.2f}% | "
            f"{tp['riichi_rate']:.2f}% | {tp['ryukyoku_rate']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## （参考外・壊れた母集団）赤条件別 Δ — 初回 md 掲載値",
            "",
            "母集団照合前の掲載値。数値は健全 1v3 再集計と一致（ソース未明示のため参考外として残す）。",
            "",
            header,
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row_label, attr, human_key in rows_delta:
        row = f"| {row_label} | {fmt_pp(HUMAN[human_key])} |"
        for label, _, _ in MODELS:
            v = ARCHIVED_AKA_DELTA[label][attr]
            row += f" {fmt_pp(v)} |"
        lines.append(row)

    lines.extend(["", "### （参考外）Monitor — 初回 md 掲載値", ""])
    lines.extend([header, "|---|---:|---:|---:|---:|"])
    for row_label, key, is_pct in monitor_rows:
        row = f"| {row_label} | — |"
        for label, _, _ in MODELS:
            v = ARCHIVED_AKA_MONITOR[label][key]
            if is_pct:
                row += f" {v:.2f}% |"
            else:
                row += f" {v:.2f} |"
        lines.append(row)

    text = "\n".join(lines) + "\n"
    print(text)
    if args.write_md:
        OUT_MD.write_text(text, encoding="utf-8")
        print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
