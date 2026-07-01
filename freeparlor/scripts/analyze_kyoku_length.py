#!/usr/bin/env python3
"""Measure trainee move count per kyoku (L) to determine chip_n_step for pure within-kyoku MC."""

import argparse
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mortal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloader import assign_r_chip_to_trainee_final_moves, load_kyoku_hora_r_chip
from libriichi.dataset import GameplayLoader

N_STEP_CANDIDATES = [3, 5, 7, 10, 15, 20, 30, 50]
MD_PATH = Path(__file__).resolve().parents[1] / "docs" / "kyoku_length_dist.md"


def stats_dict(values):
    if not values:
        return {
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": int(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def freq_table_top20(values):
    counts = Counter(values)
    return sorted(counts.items(), key=lambda x: -x[0])[:20]


def recommend_chip_n_step(max_l):
    if max_l <= 0:
        return 10
    return math.ceil(max_l / 10) * 10


def fmt_stats_row(label, s, n_kyoku):
    return (
        f"| {label} | {s['min']} | {s['mean']:.2f} | {s['median']:.2f} | "
        f"{s['p90']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} | {s['max']} | {n_kyoku} |"
    )


def fmt_freq_table(rows):
    lines = ["| L (moves/kyoku) | 局数 |", "|---|---|"]
    for l_val, cnt in rows:
        lines.append(f"| {l_val} | {cnt} |")
    return lines


def fmt_coverage_table(coverage_rows):
    lines = [
        "| n_step | MC局率 | bootstrap_move率(全局) | bootstrap_move率(chip局のみ) |",
        "|---|---|---|---|",
    ]
    for row in coverage_rows:
        lines.append(
            f"| {row['n_step']} | {row['mc_kyoku_pct']:.2%} | "
            f"{row['bootstrap_all_pct']:.2%} | {row['bootstrap_chip_pct']:.2%} |"
        )
    return lines


def check_kyoku_consecutive(at_kyoku):
    """連番前提: kyoku index が 0..max で欠番なし。"""
    assert at_kyoku.max() + 1 == len(np.unique(at_kyoku))


def process_logs_with_coverage(log_paths, version):
    loader = GameplayLoader(version=version, oracle=False, player_names=None)

    L_all = []
    L_chip = []
    skipped_games = 0
    total_games = 0
    mc_kyoku_counts = {n: 0 for n in N_STEP_CANDIDATES}
    bootstrap_all_counts = {n: 0 for n in N_STEP_CANDIDATES}
    bootstrap_chip_counts = {n: 0 for n in N_STEP_CANDIDATES}
    all_moves = 0
    chip_moves = 0

    for path in log_paths:
        games = loader.load_gz_log_files([str(path)])[0]
        for game in games:
            raw_at_kyoku = game.take_at_kyoku()
            if isinstance(raw_at_kyoku, (bytes, bytearray)):
                at_kyoku = np.frombuffer(raw_at_kyoku, dtype=np.uint8).astype(np.int64)
            else:
                at_kyoku = np.asarray(raw_at_kyoku, dtype=np.int64)
            player_id = game.take_player_id()
            game_size = len(at_kyoku)

            try:
                check_kyoku_consecutive(at_kyoku)
            except AssertionError:
                skipped_games += 1
                continue

            kyoku_ids, kyoku_lens = np.unique(at_kyoku, return_counts=True)
            L_all.extend(kyoku_lens.tolist())

            kyoku_hora_r_chip = load_kyoku_hora_r_chip(path, player_id)
            r_chip = assign_r_chip_to_trainee_final_moves(
                game_size, at_kyoku, kyoku_hora_r_chip,
            )

            last_idx_by_kyoku = {}
            for i in range(game_size):
                last_idx_by_kyoku[at_kyoku[i]] = i
            chip_kyoku_set = {
                k for k, idx in last_idx_by_kyoku.items() if r_chip[idx] != 0.0
            }

            for k, l_val in zip(kyoku_ids, kyoku_lens):
                if k in chip_kyoku_set:
                    L_chip.append(int(l_val))

            for n in N_STEP_CANDIDATES:
                mc_kyoku_counts[n] += sum(1 for l in kyoku_lens if l <= n)

            for i in range(game_size):
                k = at_kyoku[i]
                d = last_idx_by_kyoku[k] - i
                all_moves += 1
                is_chip = k in chip_kyoku_set
                if is_chip:
                    chip_moves += 1
                for n in N_STEP_CANDIDATES:
                    if d >= n:
                        bootstrap_all_counts[n] += 1
                        if is_chip:
                            bootstrap_chip_counts[n] += 1

            total_games += 1

    coverage_rows = []
    n_kyoku = len(L_all)
    for n in N_STEP_CANDIDATES:
        coverage_rows.append({
            "n_step": n,
            "mc_kyoku_pct": mc_kyoku_counts[n] / n_kyoku if n_kyoku else 0.0,
            "bootstrap_all_pct": bootstrap_all_counts[n] / all_moves if all_moves else 0.0,
            "bootstrap_chip_pct": (
                bootstrap_chip_counts[n] / chip_moves if chip_moves else 0.0
            ),
        })

    return {
        "L_all": L_all,
        "L_chip": L_chip,
        "total_games": total_games,
        "skipped_games": skipped_games,
        "all_moves": all_moves,
        "chip_moves": chip_moves,
        "coverage_rows": coverage_rows,
    }


def build_report(data, log_dir, n_files, cmd):
    L_all = data["L_all"]
    L_chip = data["L_chip"]
    stats_all = stats_dict(L_all)
    stats_chip = stats_dict(L_chip)
    freq_all = freq_table_top20(L_all)
    freq_chip = freq_table_top20(L_chip)
    max_l = stats_all["max"]
    recommend = recommend_chip_n_step(max_l)
    fifty_ok = max_l <= 50

    conclusion = (
        f"全局 MC に必要な最小 chip_n_step = max(L_all) = {max_l}。"
        f"MC アーム用に chip_n_step = {recommend} を推奨。"
        f"50 で足りる: {'Yes' if fifty_ok else 'No（max > 50）'}。"
    )

    lines = [
        "# 局長（trainee move 数/局）分布",
        "",
        f"**日付:** {date.today().isoformat()}",
        "",
        "## 実行条件",
        "",
        f"- コマンド: `{cmd}`",
        f"- `--log-dir`: `{log_dir}`",
        f"- 対象ファイル数: {n_files}",
        f"- 総 game 数: {data['total_games']}",
        f"- スキップ game 数（at_kyoku 連番前提違反）: {data['skipped_games']}",
        f"- 総局数: {len(L_all)}",
        f"- chip signal 局数: {len(L_chip)}",
        f"- 総 move 数: {data['all_moves']}",
        f"- chip 局 move 数: {data['chip_moves']}",
        "",
        "## Part 1: 局長分布（全局）",
        "",
        "| 指標 | min | mean | median | p90 | p95 | p99 | max | 局数 |",
        "|---|---|---|---|---|---|---|---|---|",
        fmt_stats_row("全局", stats_all, len(L_all)),
        "",
        "### 度数表（上位20、L降順）",
        "",
        *fmt_freq_table(freq_all),
        "",
        "## Part 2: チップ signal 局に限定した分布",
        "",
        "trainee がチップ和了（r_chip ≠ 0）した局のみ。",
        "",
        "| 指標 | min | mean | median | p90 | p95 | p99 | max | 局数 |",
        "|---|---|---|---|---|---|---|---|---|",
        fmt_stats_row("chip局", stats_chip, len(L_chip)),
        "",
        "### 度数表（上位20、L降順）",
        "",
        *fmt_freq_table(freq_chip),
        "",
        "## Part 3: n_step 候補ごとの coverage",
        "",
        "d = 局内最終 index − move index。MC局率 = L ≤ n_step の局の割合。",
        "bootstrap_move率 = d ≥ n_step の move の割合（純 bootstrap 学習割合）。",
        "",
        *fmt_coverage_table(data["coverage_rows"]),
        "",
        "## 結論",
        "",
        conclusion,
        "",
    ]
    return "\n".join(lines), stats_all, stats_chip, recommend, fifty_ok, conclusion


def print_summary(data, stats_all, recommend, fifty_ok, conclusion):
    print("=== analyze_kyoku_length ===")
    if data["skipped_games"]:
        print(f"skipped games (at_kyoku gap): {data['skipped_games']}")
    print(f"games: {data['total_games']}, kyoku: {len(data['L_all'])}, "
          f"chip_kyoku: {len(data['L_chip'])}")
    print(f"max L (全局): {stats_all['max']}")
    print(f"推奨 chip_n_step: {recommend}")
    print(f"50 で足りる: {'Yes' if fifty_ok else 'No'}")
    print()
    print("Part 3 coverage:")
    print("| n_step | MC局率 | bootstrap(全局) | bootstrap(chip) |")
    for row in data["coverage_rows"]:
        print(
            f"| {row['n_step']:>2} | {row['mc_kyoku_pct']:6.2%} | "
            f"{row['bootstrap_all_pct']:6.2%} | {row['bootstrap_chip_pct']:6.2%} |"
        )
    print()
    print(conclusion)


def main():
    parser = argparse.ArgumentParser(
        description="Measure kyoku length distribution for chip_n_step threshold",
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0, help="0=all")
    args = parser.parse_args()

    log_paths = sorted(args.log_dir.glob("*.json.gz"))
    if args.max_files > 0:
        log_paths = log_paths[: args.max_files]

    cmd = (
        f"python {Path(__file__).name} "
        f"--log-dir {args.log_dir} --version {args.version}"
    )
    if args.max_files > 0:
        cmd += f" --max-files {args.max_files}"

    if not log_paths:
        print("該当ファイルなし")
        md = (
            f"# 局長（trainee move 数/局）分布\n\n"
            f"**日付:** {date.today().isoformat()}\n\n"
            f"コマンド: `{cmd}`\n\n"
            f"`--log-dir`: `{args.log_dir}`\n\n"
            "該当ファイルなし。\n"
        )
        MD_PATH.write_text(md, encoding="utf-8")
        return

    data = process_logs_with_coverage(log_paths, args.version)
    md_text, stats_all, _, recommend, fifty_ok, conclusion = build_report(
        data, args.log_dir.resolve(), len(log_paths), cmd,
    )
    print_summary(data, stats_all, recommend, fifty_ok, conclusion)
    MD_PATH.write_text(md_text, encoding="utf-8")
    print(f"\nwritten: {MD_PATH}")


if __name__ == "__main__":
    main()
