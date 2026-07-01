#!/usr/bin/env python3
"""Count trainee aka-call-hoora (target positive) rarity from arena logs."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mortal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloader import (
    assign_r_chip_to_trainee_final_moves,
    get_hora_chip_delta,
    load_kyoku_hora_r_chip,
    open_log,
)
from libriichi.dataset import GameplayLoader

# libriichi/src/dataset/gameplay.rs (version 4 action labels)
CALL_ACTION_IDS = frozenset({38, 39, 40, 41, 42})
CALL_ACTION_LABELS = {
    38: "chi_low",
    39: "chi_mid",
    40: "chi_high",
    41: "pon",
    42: "kan_select (daiminkan / kakan / ankan)",
}
REACH_ACTION_ID = 37
REACH_ACTION_LABEL = "reach (riichi)"

MD_PATH = Path(__file__).resolve().parents[1] / "docs" / "aka_call_hora_count.md"
EXPECTED_CHIP_KYOKU = 5743  # cross-check vs kyoku_length_dist.md (same log-dir)


def collect_kyoku_event_data(path, player_id):
    """One pass: hora_by_kyoku (collect_hora_by_kyoku) + trainee n_aka (agari_detail)."""
    from libriichi.state import PlayerState

    with open_log(path) as f:
        events = [json.loads(line) for line in f if line.strip()]

    states = [PlayerState(i) for i in range(4)]
    kyoku_idx = -1
    hora_by_kyoku = defaultdict(list)
    n_aka_by_kyoku = {}

    for i, ev in enumerate(events):
        et = ev.get("type")
        if et == "start_kyoku":
            kyoku_idx += 1
        elif et == "hora" and kyoku_idx >= 0:
            actor = ev["actor"]
            cd = get_hora_chip_delta(ev)
            hora_by_kyoku[kyoku_idx].append(
                {
                    "event_idx": i,
                    "actor": actor,
                    "target": ev["target"],
                    "chip_delta": cd,
                    "is_ron": actor != ev["target"],
                }
            )
            if actor == player_id:
                is_ron = actor != ev["target"]
                ura = ev.get("ura_markers") or ev.get("ura_indicators") or []
                try:
                    detail = states[actor].agari_detail(is_ron, ura)
                    n_aka_by_kyoku[kyoku_idx] = int(detail.num_aka)
                except Exception:
                    n_aka_by_kyoku[kyoku_idx] = 0
        for s in states:
            s.update(json.dumps(ev, separators=(",", ":")))

    return hora_by_kyoku, n_aka_by_kyoku


def collect_kyoku_action_flags(at_kyoku, actions):
    """Per-kyoku trainee call / riichi from action IDs split by at_kyoku."""
    did_call = defaultdict(bool)
    did_riichi = defaultdict(bool)
    action_hist = Counter()
    for i in range(len(at_kyoku)):
        k = int(at_kyoku[i])
        a = int(actions[i])
        if a in CALL_ACTION_IDS:
            did_call[k] = True
            action_hist[a] += 1
        if a == REACH_ACTION_ID:
            did_riichi[k] = True
            action_hist[a] += 1
    return dict(did_call), dict(did_riichi), action_hist


def check_kyoku_consecutive(at_kyoku):
    assert at_kyoku.max() + 1 == len(np.unique(at_kyoku))


@dataclass
class KyokuCounts:
    total: int = 0
    trainee_hora: int = 0
    call_hora: int = 0
    aka_call_hora: int = 0
    riichi_aka_hora: int = 0
    chip_signal: int = 0


@dataclass
class Aggregate:
    global_counts: KyokuCounts = field(default_factory=KyokuCounts)
    per_file: list[tuple[str, KyokuCounts]] = field(default_factory=list)
    skipped_games: int = 0
    total_games: int = 0
    action_hist: Counter = field(default_factory=Counter)


def process_logs(log_paths, version):
    loader = GameplayLoader(version=version, oracle=False, player_names=None)
    agg = Aggregate()

    for file_idx, path in enumerate(log_paths, start=1):
        file_counts = KyokuCounts()
        games = loader.load_gz_log_files([str(path)])[0]
        hora_by_kyoku = None
        n_aka_by_kyoku = None

        for game in games:
            raw_at_kyoku = game.take_at_kyoku()
            if isinstance(raw_at_kyoku, (bytes, bytearray)):
                at_kyoku = np.frombuffer(raw_at_kyoku, dtype=np.uint8).astype(np.int64)
            else:
                at_kyoku = np.asarray(raw_at_kyoku, dtype=np.int64)
            actions = np.asarray(game.take_actions(), dtype=np.int64)
            player_id = game.take_player_id()
            game_size = len(at_kyoku)

            try:
                check_kyoku_consecutive(at_kyoku)
            except AssertionError:
                agg.skipped_games += 1
                continue

            n_kyoku = int(at_kyoku.max()) + 1
            file_counts.total += n_kyoku

            kyoku_hora_r_chip = load_kyoku_hora_r_chip(path, player_id)
            r_chip = assign_r_chip_to_trainee_final_moves(
                game_size, at_kyoku, kyoku_hora_r_chip,
            )
            last_idx_by_kyoku = {}
            for i in range(game_size):
                last_idx_by_kyoku[int(at_kyoku[i])] = i
            chip_kyoku = {
                k for k, idx in last_idx_by_kyoku.items() if r_chip[idx] != 0.0
            }
            file_counts.chip_signal += len(chip_kyoku)

            did_call, did_riichi, action_hist = collect_kyoku_action_flags(
                at_kyoku, actions,
            )
            agg.action_hist.update(action_hist)

            if hora_by_kyoku is None:
                hora_by_kyoku, n_aka_by_kyoku = collect_kyoku_event_data(path, player_id)

            for k in range(n_kyoku):
                horas = hora_by_kyoku.get(k, [])
                trainee_won = any(h["actor"] == player_id for h in horas)
                if not trainee_won:
                    continue

                file_counts.trainee_hora += 1
                has_call = did_call.get(k, False)
                n_aka = n_aka_by_kyoku.get(k, 0)
                has_aka = n_aka > 0
                has_riichi = did_riichi.get(k, False)

                if has_call:
                    file_counts.call_hora += 1
                if has_call and has_aka:
                    file_counts.aka_call_hora += 1
                if has_riichi and has_aka:
                    file_counts.riichi_aka_hora += 1

            agg.total_games += 1

        agg.per_file.append((path.name, file_counts))
        for field_name in KyokuCounts.__dataclass_fields__:
            setattr(
                agg.global_counts,
                field_name,
                getattr(agg.global_counts, field_name)
                + getattr(file_counts, field_name),
            )
        if file_idx % 100 == 0:
            print(f"  ... {file_idx}/{len(log_paths)} files", flush=True)

    return agg


def per_file_values(per_file, attr):
    return [getattr(c, attr) for _, c in per_file]


def file_stats(values):
    if not values:
        return {"min": 0, "median": 0.0, "max": 0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def pct(num, den):
    if den == 0:
        return 0.0
    return 100.0 * num / den


def fmt_pct(x):
    return f"{x:.4f}%"


def histogram_lines(counts, label):
    hist = Counter(counts)
    max_key = max(hist.keys()) if hist else 0
    lines = [f"### {label}", "", "| 件数/ファイル | ファイル数 |", "|---|---|"]
    for n in range(max_key + 1):
        lines.append(f"| {n} | {hist.get(n, 0)} |")
    zero_rate = pct(sum(1 for v in counts if v == 0), len(counts)) if counts else 0.0
    lines.append("")
    lines.append(f"0件ファイル率: {fmt_pct(zero_rate)}")
    return lines, zero_rate


def build_report(agg, log_dir, n_files, cmd):
    g = agg.global_counts
    chip = g.chip_signal
    total = g.total

    rows = []
    specs = [
        ("1. 全局数", "total", total),
        ("2. trainee和了", "trainee_hora", g.trainee_hora),
        ("3. trainee和了 ∧ 副露あり", "call_hora", g.call_hora),
        ("4. trainee和了 ∧ 副露 ∧ 赤 (target)", "aka_call_hora", g.aka_call_hora),
        ("5. trainee和了 ∧ 立直 ∧ 赤 (参考)", "riichi_aka_hora", g.riichi_aka_hora),
        ("6. chip signal (r_chip≠0)", "chip_signal", chip),
    ]
    for label, attr, count in specs:
        vals = per_file_values(agg.per_file, attr)
        fs = file_stats(vals)
        chip_pct = (
            100.0 if attr == "chip_signal"
            else (pct(count, chip) if attr != "total" else float("nan"))
        )
        rows.append({
            "label": label,
            "count": count,
            "all_pct": pct(count, total),
            "chip_pct": chip_pct,
            "mean_per_file": fs["mean"],
            "min": fs["min"],
            "median": fs["median"],
            "max": fs["max"],
        })

    target_vals = per_file_values(agg.per_file, "aka_call_hora")
    hist_lines, zero_rate = histogram_lines(target_vals, "正例(4) ファイル単位ヒストグラム")

    target = rows[3]
    chip_row = rows[5]
    chip_diff = abs(chip_row["count"] - EXPECTED_CHIP_KYOKU)
    chip_ok = chip_diff <= max(5, int(EXPECTED_CHIP_KYOKU * 0.01))

    if target["all_pct"] < 0.5 and target["mean_per_file"] < 2.0 and zero_rate > 50:
        rarity_verdict = "Yes"
        rarity_reason = (
            f"正例(4)は全局の{fmt_pct(target['all_pct'])}・1ファイル平均{target['mean_per_file']:.2f}件・"
            f"0件ファイル率{fmt_pct(zero_rate)}と極めて薄く、batch 内で平均化されやすい。"
        )
    elif target["all_pct"] < 1.0 and target["mean_per_file"] < 5.0:
        rarity_verdict = "Yes"
        rarity_reason = (
            f"正例(4)は全局比{fmt_pct(target['all_pct'])}・1ファイル平均{target['mean_per_file']:.2f}件と"
            " chip 局に比べて十分希少。"
        )
    else:
        rarity_verdict = "No"
        rarity_reason = (
            f"正例(4)は全局比{fmt_pct(target['all_pct'])}・1ファイル平均{target['mean_per_file']:.2f}件で、"
            "単純な希少性だけでは説明しにくい。"
        )

    action_labels = {**CALL_ACTION_LABELS, REACH_ACTION_ID: REACH_ACTION_LABEL}
    action_dump = ", ".join(
        f"{aid}={action_labels[aid]} ({agg.action_hist.get(aid, 0)} moves)"
        for aid in sorted(CALL_ACTION_IDS | {REACH_ACTION_ID})
    )

    lines = [
        "# 赤を活かした鳴き和了（target 正例）の希少性",
        "",
        f"**日付:** {date.today().isoformat()}",
        "",
        "## 実行条件",
        "",
        f"- コマンド: `{cmd}`",
        f"- `--log-dir`: `{log_dir}`",
        f"- 対象ファイル数: {n_files}",
        f"- 総 game 数: {agg.total_games}",
        f"- スキップ game 数（at_kyoku 連番前提違反）: {agg.skipped_games}",
        f"- 総局数: {total}",
        "",
        "## 集計表（局単位）",
        "",
        "| 区分 | 局数 | 全局比(%) | chip局比(%) | 1ファイル平均 | min | median | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        chip_s = "—" if r["chip_pct"] != r["chip_pct"] else f"{r['chip_pct']:.4f}"
        lines.append(
            f"| {r['label']} | {r['count']} | {r['all_pct']:.4f} | {chip_s} | "
            f"{r['mean_per_file']:.2f} | {r['min']} | {r['median']:.1f} | {r['max']} |"
        )

    lines.extend([
        "",
        *hist_lines,
        "",
        "## 判定根拠",
        "",
        "### 鳴きアクション ID（GameplayLoader / libriichi gameplay.rs v4）",
        "",
        f"- 副露判定: trainee action ∈ {sorted(CALL_ACTION_IDS)}",
        f"- 立直判定: trainee action == {REACH_ACTION_ID} ({REACH_ACTION_LABEL})",
        f"- 全ログ move 集計: {action_dump}",
        "",
        "※ action 42 は daiminkan に加え kakan/ankan も同 ID。副露は chi/pon/明カン中心だが、"
        "loader 上は 42 を副露相当として含む（暗槓/加槓のみで和了した局は稀な誤包含）。",
        "",
        "### 赤判定（n_aka > 0）",
        "",
        "- 経路: 生ログを `PlayerState` で再生 → trainee hora 時に `agari_detail(is_ron, ura)` → `detail.num_aka`",
        "- 根拠: `preprocess_chips.chip_base(detail)` が `detail.num_aka` を chip 枚数に含めるのと同経路",
        "- hora 集約: `collect_hora_by_kyoku` 相当を `collect_kyoku_event_data` に統合（`get_hora_chip_delta` 使用）",
        "- chip signal との対応: `load_kyoku_hora_r_chip` + `assign_r_chip_to_trainee_final_moves` で r_chip≠0",
        "",
        "### cross-check",
        "",
        f"- 期待 chip signal 局数（kyoku_length_dist.md）: {EXPECTED_CHIP_KYOKU}",
        f"- 今回 (6): {chip_row['count']}（差 {chip_diff}）→ {'OK' if chip_ok else '要確認'}",
        f"- 包含関係: (2)≥(3)≥(4): {g.trainee_hora}≥{g.call_hora}≥{g.aka_call_hora} → "
        f"{'OK' if g.trainee_hora >= g.call_hora >= g.aka_call_hora else 'NG'}",
        "",
        "## 結論",
        "",
        f"正例(4) = 全局の **{fmt_pct(target['all_pct'])}** ・"
        f"1ファイル平均 **{target['mean_per_file']:.2f}件** ・"
        f"0件ファイル率 **{fmt_pct(zero_rate)}**。",
        f"希少性が主因と言えるか: **{rarity_verdict}** — {rarity_reason}",
        "",
    ])
    return "\n".join(lines), rows, zero_rate, chip_ok, rarity_verdict


def print_summary(agg, rows, zero_rate, chip_ok):
    g = agg.global_counts
    target = rows[3]
    chip_row = rows[5]
    print("=== count_aka_call_hora ===")
    if agg.skipped_games:
        print(f"skipped games (at_kyoku gap): {agg.skipped_games}")
    print(f"games: {agg.total_games}, kyoku: {g.total}, chip_kyoku: {g.chip_signal}")
    print()
    print("| 区分 | 局数 | 全局比 | 1ファイル平均 |")
    for r in rows:
        print(
            f"| {r['label']} | {r['count']} | {r['all_pct']:.4f}% | "
            f"{r['mean_per_file']:.2f} |"
        )
    print()
    print(
        f"正例(4): 全局比={target['all_pct']:.4f}%, "
        f"1ファイル平均={target['mean_per_file']:.2f}, "
        f"0件ファイル率={zero_rate:.2f}%"
    )
    print(
        f"chip cross-check: {chip_row['count']} "
        f"(expected ~{EXPECTED_CHIP_KYOKU}) {'OK' if chip_ok else 'WARN'}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Count trainee aka-call-hoora target positive rarity",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/home/gamba/mahjong/runs/online_diag_b/train_play/client0"),
    )
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
            f"# 赤を活かした鳴き和了（target 正例）の希少性\n\n"
            f"**日付:** {date.today().isoformat()}\n\n"
            f"コマンド: `{cmd}`\n\n"
            f"`--log-dir`: `{args.log_dir}`\n\n"
            "該当ファイルなし。\n"
        )
        MD_PATH.write_text(md, encoding="utf-8")
        return

    agg = process_logs(log_paths, args.version)
    md_text, rows, zero_rate, chip_ok, _ = build_report(
        agg, args.log_dir.resolve(), len(log_paths), cmd,
    )
    print_summary(agg, rows, zero_rate, chip_ok)
    MD_PATH.write_text(md_text, encoding="utf-8")
    print(f"\nwritten: {MD_PATH}")


if __name__ == "__main__":
    main()
