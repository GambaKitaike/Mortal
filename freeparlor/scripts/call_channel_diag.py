#!/usr/bin/env python3
"""Diagnose call-channel fault: hora breakdown (Part A) and explore/skill split (Part B)."""

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

_MORTAL_ROOT = Path(__file__).resolve().parents[2] / "mortal"
if "MORTAL_CFG" not in os.environ:
    _default_cfg = _MORTAL_ROOT / "config.toml"
    if _default_cfg.exists():
        os.environ["MORTAL_CFG"] = str(_default_cfg)

sys.path.insert(0, str(_MORTAL_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from count_aka_call_hora import (
    CALL_ACTION_IDS,
    REACH_ACTION_ID,
    check_kyoku_consecutive,
    collect_kyoku_action_flags,
    collect_kyoku_event_data,
)
from libriichi.dataset import GameplayLoader

CHI_ACTION_IDS = frozenset({38, 39, 40})
PON_ACTION_ID = 41

HUMAN_RIICHI_PCT = 42.04
HUMAN_DAMA_PCT = 8.67
HUMAN_CALL_PCT = 49.29

MD_PATH = Path(__file__).resolve().parents[1] / "docs" / "call_channel_diag.md"

DEFAULT_LOG_DIRS = [
    "/home/gamba/mahjong/runs/online_diag_b/train_play/client0",
    "/home/gamba/mahjong/runs/online_cql_mqw03/train_play/client0",
    "/home/gamba/mahjong/runs/online_main/train_play/client0",
]
DEFAULT_LABELS = ["diag_b", "mqw03", "main"]
DEFAULT_TEACHER_GLOB = "/home/gamba/mahjong/data/tenhou/2009/*.mjson"


def pct(num, den):
    if den == 0:
        return 0.0
    return 100.0 * num / den


def fmt_pct(x):
    return f"{x:.2f}%"


def mask_arr(mask):
    return np.asarray(mask, dtype=bool)


def call_legal(mask):
    m = mask_arr(mask)
    return any(m[aid] for aid in CALL_ACTION_IDS if aid < len(m))


def chi_legal(mask):
    m = mask_arr(mask)
    return any(m[aid] for aid in CHI_ACTION_IDS if aid < len(m))


def pon_legal(mask):
    m = mask_arr(mask)
    return PON_ACTION_ID < len(m) and m[PON_ACTION_ID]


@dataclass
class PartAStats:
    total_kyoku: int = 0
    trainee_hora: int = 0
    call_hora: int = 0
    riichi_hora: int = 0
    dama_hora: int = 0
    call_hora_aka: int = 0
    skipped_games: int = 0
    total_games: int = 0
    n_files: int = 0


@dataclass
class PartB1Stats:
    call_legal_moves: int = 0
    call_moves: int = 0
    chi_legal_moves: int = 0
    chi_moves: int = 0
    pon_legal_moves: int = 0
    pon_moves: int = 0


@dataclass
class PartB2Stats:
    call_kyoku: int = 0
    menzen_kyoku: int = 0
    call_trainee_hora: int = 0
    menzen_trainee_hora: int = 0
    call_opponent_hora: int = 0
    call_ryukyoku: int = 0


@dataclass
class SourceResult:
    label: str
    log_dir: str
    part_a: PartAStats = field(default_factory=PartAStats)
    part_b1: PartB1Stats = field(default_factory=PartB1Stats)
    part_b2: PartB2Stats = field(default_factory=PartB2Stats)
    run_b: bool = False


def process_game(game, path, part_a, part_b1, part_b2, run_b):
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
        part_a.skipped_games += 1
        return False

    n_kyoku = int(at_kyoku.max()) + 1
    part_a.total_kyoku += n_kyoku

    did_call, did_riichi, _ = collect_kyoku_action_flags(at_kyoku, actions)
    hora_by_kyoku, n_aka_by_kyoku = collect_kyoku_event_data(path, player_id)

    if run_b:
        masks = game.take_masks()
        for i in range(game_size):
            m = masks[i]
            a = int(actions[i])
            if call_legal(m):
                part_b1.call_legal_moves += 1
                if a in CALL_ACTION_IDS:
                    part_b1.call_moves += 1
            if chi_legal(m):
                part_b1.chi_legal_moves += 1
                if a in CHI_ACTION_IDS:
                    part_b1.chi_moves += 1
            if pon_legal(m):
                part_b1.pon_legal_moves += 1
                if a == PON_ACTION_ID:
                    part_b1.pon_moves += 1

    for k in range(n_kyoku):
        horas = hora_by_kyoku.get(k, [])
        trainee_won = any(h["actor"] == player_id for h in horas)
        has_call = did_call.get(k, False)

        if trainee_won:
            part_a.trainee_hora += 1
            has_riichi = did_riichi.get(k, False)
            if has_call:
                part_a.call_hora += 1
                if n_aka_by_kyoku.get(k, 0) > 0:
                    part_a.call_hora_aka += 1
            elif has_riichi:
                part_a.riichi_hora += 1
            else:
                part_a.dama_hora += 1

        if run_b:
            if has_call:
                part_b2.call_kyoku += 1
                if trainee_won:
                    part_b2.call_trainee_hora += 1
                elif horas:
                    part_b2.call_opponent_hora += 1
                else:
                    part_b2.call_ryukyoku += 1
            else:
                part_b2.menzen_kyoku += 1
                if trainee_won:
                    part_b2.menzen_trainee_hora += 1

    part_a.total_games += 1
    return True


def process_log_paths(log_paths, version, run_b=False):
    loader = GameplayLoader(version=version, oracle=False, player_names=None)
    part_a = PartAStats()
    part_b1 = PartB1Stats()
    part_b2 = PartB2Stats()

    for file_idx, path in enumerate(log_paths, start=1):
        games = loader.load_gz_log_files([str(path)])[0]
        for game in games:
            process_game(game, path, part_a, part_b1, part_b2, run_b)
        if file_idx % 100 == 0:
            print(f"  ... {file_idx}/{len(log_paths)} files", flush=True)

    part_a.n_files = len(log_paths)
    assert part_a.call_hora + part_a.riichi_hora + part_a.dama_hora == part_a.trainee_hora
    if run_b:
        assert part_b2.call_kyoku + part_b2.menzen_kyoku == part_a.total_kyoku

    return part_a, part_b1, part_b2


def resolve_log_paths(log_dir, max_files):
    paths = sorted(Path(log_dir).glob("*.json.gz"))
    if max_files > 0:
        paths = paths[:max_files]
    return paths


def pick_b_label(labels, log_dirs):
    for preferred in ("mqw03", "best", "main"):
        if preferred in labels:
            return preferred
    if "diag_b" in labels:
        return "diag_b"
    return labels[0] if labels else None


def build_part_a_row(label, a: PartAStats):
    th = a.trainee_hora
    return {
        "label": label,
        "files": a.n_files,
        "kyoku": a.total_kyoku,
        "trainee_hora": th,
        "call": a.call_hora,
        "call_pct": pct(a.call_hora, th),
        "riichi": a.riichi_hora,
        "riichi_pct": pct(a.riichi_hora, th),
        "dama": a.dama_hora,
        "dama_pct": pct(a.dama_hora, th),
        "call_aka_pct": pct(a.call_hora_aka, a.call_hora),
        "games": a.total_games,
        "skipped": a.skipped_games,
    }


def build_conclusion(part_a_rows, b1: PartB1Stats, b2: PartB2Stats, teacher_row, b_label):
    lines = []
    ai_rows = [
        r for r in part_a_rows
        if r["label"] not in ("human", "teacher") and r["trainee_hora"] > 0
    ]
    if not ai_rows:
        lines.append("- 比較対象 ckpt なし（全 log-dir が空または和了 0）。")
        return lines

    call_pcts = [r["call_pct"] for r in ai_rows]
    human_like = [r for r in ai_rows if r["call_pct"] >= HUMAN_CALL_PCT * 0.7]
    broken = [r for r in ai_rows if r["call_pct"] < HUMAN_CALL_PCT * 0.25]

    if len(ai_rows) >= 2 and human_like and broken:
        good = ", ".join(r["label"] for r in human_like)
        bad = ", ".join(r["label"] for r in broken)
        lines.append(
            f"- **ckpt 間で副露和了率が二極化**（正常≈{max(r['call_pct'] for r in human_like):.2f}%: "
            f"{good} / 異常≈{min(r['call_pct'] for r in broken):.2f}%: {bad}）。"
            f" 人間={HUMAN_CALL_PCT:.2f}% → 故障は universal ではなく **特定 ckpt 回帰**。"
        )
    elif all(p < HUMAN_CALL_PCT * 0.5 for p in call_pcts):
        labels = ", ".join(r["label"] for r in ai_rows)
        lines.append(
            f"- **全 ckpt ({labels}) で副露和了率が人間参照({HUMAN_CALL_PCT:.2f}%)の半分未満** "
            f"→ 故障は universal（単一 ckpt 固有でない）。"
        )
    else:
        spread = max(call_pcts) - min(call_pcts)
        lines.append(
            f"- ckpt 間の副露和了率差 {spread:.2f}pp。"
            f"最低={min(call_pcts):.2f}% 最高={max(call_pcts):.2f}% "
            f"（人間={HUMAN_CALL_PCT:.2f}%）。"
        )

    p_call = pct(b1.call_moves, b1.call_legal_moves)
    p_chi = pct(b1.chi_moves, b1.chi_legal_moves)
    p_pon = pct(b1.pon_moves, b1.pon_legal_moves)
    p_hora_call = pct(b2.call_trainee_hora, b2.call_kyoku)
    p_hora_menzen = pct(b2.menzen_trainee_hora, b2.menzen_kyoku)

    if p_call < 5.0:
        lines.append(
            f"- **B1({b_label}) P(call|legal)={p_call:.2f}% は極端に低い** "
            f"→ explore 不足 → 探索注入(C) が有効。"
        )
    elif p_call < 15.0:
        lines.append(
            f"- B1({b_label}) P(call|legal)={p_call:.2f}% は低〜中程度 "
            f"(chi={p_chi:.2f}%, pon={p_pon:.2f}%)。"
        )
    else:
        lines.append(
            f"- B1({b_label}) P(call|legal)={p_call:.2f}% はそこそこ "
            f"(chi={p_chi:.2f}%, pon={p_pon:.2f}%)。"
        )

    if p_hora_call < p_hora_menzen * 0.7 and p_hora_menzen - p_hora_call > 2.0:
        ry_pct = pct(b2.call_ryukyoku, b2.call_kyoku)
        opp_pct = pct(b2.call_opponent_hora, b2.call_kyoku)
        if ry_pct > opp_pct:
            detail = f"流局偏重({ry_pct:.2f}%) → 形/役が作れていない疑い"
        else:
            detail = f"他家和了偏重({opp_pct:.2f}%) → 鳴いて守備崩壊疑い"
        lines.append(
            f"- **B2({b_label}) P(和了|副露局)={p_hora_call:.2f}% ≪ "
            f"P(和了|門前局)={p_hora_menzen:.2f}%** → 鳴いた手が死ぬ。"
            f" (C)単独では不足。{detail}。"
        )
    elif p_hora_call >= p_hora_menzen:
        lines.append(
            f"- B2({b_label}) P(和了|副露局)={p_hora_call:.2f}% ≥ "
            f"P(和了|門前局)={p_hora_menzen:.2f}% "
            f"→ 鳴けば上がれるが、B1 で鳴いていない可能性が高い。"
        )
    else:
        lines.append(
            f"- B2({b_label}) P(和了|副露局)={p_hora_call:.2f}% vs "
            f"門前={p_hora_menzen:.2f}% → 中間的乖離。"
        )

    if teacher_row is not None and teacher_row["trainee_hora"] > 0:
        t_call = teacher_row["call_pct"]
        if broken and human_like:
            best = max(human_like, key=lambda r: r["call_pct"])
            worst = min(broken, key=lambda r: r["call_pct"])
            lines.append(
                f"- 教師副露和了率={t_call:.2f}%（人間並み）。"
                f" 正常 ckpt **{best['label']}**={best['call_pct']:.2f}% は教師に近いが、"
                f" **{worst['label']}**={worst['call_pct']:.2f}% だけ崩れている "
                f"→ 教師/正常 ckpt には副露和了があるのに特定 ckpt で潰れた "
                f"→ **報酬/CQL 設計 or その ckpt 固有の学習経路**を疑う。"
            )
        elif t_call >= 35.0 and all(r["call_pct"] < t_call * 0.3 for r in ai_rows):
            ai_call = sum(call_pcts) / len(call_pcts)
            lines.append(
                f"- **教師副露和了率={t_call:.2f}% に対し全 AI ckpt 平均≈{ai_call:.2f}%** "
                f"→ 教師にはあるのに学習で潰れた → **報酬/CQL 設計起因**を疑う。"
            )
        elif t_call < 25.0:
            lines.append(
                f"- 教師副露和了率={t_call:.2f}% も人間比で低め → データ源自体の確認が必要。"
            )
        else:
            ai_call = sum(call_pcts) / len(call_pcts)
            lines.append(
                f"- 教師副露和了率={t_call:.2f}% vs AI 平均≈{ai_call:.2f}%。"
            )

    return lines


def build_report(cmd, sources, part_a_rows, b_source, teacher_note):
    lines = [
        "# 鳴き和了チャネル診断 (call_channel_diag)",
        "",
        f"**日付:** {date.today().isoformat()}",
        "",
        "## 実行条件",
        "",
        f"- コマンド: `{cmd}`",
        "",
    ]
    for src in sources:
        a = src.part_a
        lines.append(
            f"- **{src.label}**: `{src.log_dir}` — "
            f"ファイル {a.n_files}、game {a.total_games}、"
            f"スキップ {a.skipped_games}、総局 {a.total_kyoku}"
        )
    if teacher_note:
        lines.append(f"- **教師データ**: {teacher_note}")

    lines.extend([
        "",
        "## Part A: 和了時内訳（排他分割）",
        "",
        "副露和了 = trainee和了 ∧ 鳴き(38–42)あり / "
        "立直和了 = trainee和了 ∧ 門前 ∧ 立直(37) / "
        "ダマ和了 = trainee和了 ∧ 門前 ∧ 立直なし。",
        "",
        "| ソース | trainee和了 | 副露 | % | 立直 | % | ダマ | % | 副露和了赤率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for r in part_a_rows:
        lines.append(
            f"| {r['label']} | {r['trainee_hora']} | {r['call']} | {r['call_pct']:.2f}% | "
            f"{r['riichi']} | {r['riichi_pct']:.2f}% | {r['dama']} | {r['dama_pct']:.2f}% | "
            f"{r['call_aka_pct']:.2f}% |"
        )

    if b_source is not None:
        b1 = b_source.part_b1
        b2 = b_source.part_b2
        lines.extend([
            "",
            f"## Part B: explore / skill 切り分け（代表: {b_source.label}）",
            "",
            "### B1: 鳴ける時に鳴いているか",
            "",
            "| 指標 | 分子 | 分母 | P |",
            "|---|---:|---:|---:|",
            f"| P(call \\| call legal) | {b1.call_moves} | {b1.call_legal_moves} | "
            f"{pct(b1.call_moves, b1.call_legal_moves):.2f}% |",
            f"| P(chi \\| chi legal) | {b1.chi_moves} | {b1.chi_legal_moves} | "
            f"{pct(b1.chi_moves, b1.chi_legal_moves):.2f}% |",
            f"| P(pon \\| pon legal) | {b1.pon_moves} | {b1.pon_legal_moves} | "
            f"{pct(b1.pon_moves, b1.pon_legal_moves):.2f}% |",
            "",
            "### B2: 鳴いた局が和了に化けるか",
            "",
            "| 指標 | 値 |",
            "|---|---:|",
            f"| 副露局数 | {b2.call_kyoku} |",
            f"| 門前局数 | {b2.menzen_kyoku} |",
            f"| P(和了 \\| 副露局) | {pct(b2.call_trainee_hora, b2.call_kyoku):.2f}% |",
            f"| P(和了 \\| 門前局) | {pct(b2.menzen_trainee_hora, b2.menzen_kyoku):.2f}% |",
            "",
            "#### 副露局の結末内訳",
            "",
            "| 結末 | 局数 | 副露局比 |",
            "|---|---:|---:|",
            f"| trainee和了 | {b2.call_trainee_hora} | "
            f"{pct(b2.call_trainee_hora, b2.call_kyoku):.2f}% |",
            f"| 他家和了 | {b2.call_opponent_hora} | "
            f"{pct(b2.call_opponent_hora, b2.call_kyoku):.2f}% |",
            f"| 流局 | {b2.call_ryukyoku} | "
            f"{pct(b2.call_ryukyoku, b2.call_kyoku):.2f}% |",
        ])

    teacher_row = next((r for r in part_a_rows if r["label"] == "teacher"), None)
    conclusion = build_conclusion(
        part_a_rows,
        b_source.part_b1 if b_source else PartB1Stats(),
        b_source.part_b2 if b_source else PartB2Stats(),
        teacher_row,
        b_source.label if b_source else "?",
    )
    lines.extend([
        "",
        "## 結論",
        "",
        *conclusion,
        "",
        "### 判定根拠（action ID）",
        "",
        f"- 副露: trainee action ∈ {sorted(CALL_ACTION_IDS)}",
        f"- 立直: trainee action == {REACH_ACTION_ID}",
        "",
    ])
    return "\n".join(lines)


def print_summary(part_a_rows, b_source):
    print("=== call_channel_diag ===")
    print()
    print("Part A: 和了時内訳")
    print("| ソース | 副露% | 立直% | ダマ% | 副露赤率 | trainee和了 |")
    for r in part_a_rows:
        print(
            f"| {r['label']} | {r['call_pct']:.2f}% | {r['riichi_pct']:.2f}% | "
            f"{r['dama_pct']:.2f}% | {r['call_aka_pct']:.2f}% | {r['trainee_hora']} |"
        )
    print()
    if b_source is not None:
        b1 = b_source.part_b1
        b2 = b_source.part_b2
        print(f"Part B ({b_source.label}):")
        print(
            f"  B1 P(call|legal)={pct(b1.call_moves, b1.call_legal_moves):.2f}% "
            f"chi={pct(b1.chi_moves, b1.chi_legal_moves):.2f}% "
            f"pon={pct(b1.pon_moves, b1.pon_legal_moves):.2f}%"
        )
        print(
            f"  B2 P(hora|副露)={pct(b2.call_trainee_hora, b2.call_kyoku):.2f}% "
            f"P(hora|門前)={pct(b2.menzen_trainee_hora, b2.menzen_kyoku):.2f}%"
        )
        print(
            f"  副露局結末: trainee={b2.call_trainee_hora} "
            f"他家={b2.call_opponent_hora} 流局={b2.call_ryukyoku}"
        )


def main():
    parser = argparse.ArgumentParser(description="Call-channel hora breakdown and explore/skill diag")
    parser.add_argument("--log-dirs", nargs="+", type=Path, default=[Path(p) for p in DEFAULT_LOG_DIRS])
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0, help="0=all")
    parser.add_argument("--teacher-glob", default=DEFAULT_TEACHER_GLOB)
    parser.add_argument("--teacher-max-files", type=int, default=500)
    parser.add_argument(
        "--b-label",
        default="",
        help="Part B representative label (default: mqw03 > best > main > diag_b)",
    )
    args = parser.parse_args()

    log_dirs = [Path(d) for d in args.log_dirs]
    labels = list(args.labels)
    if len(labels) != len(log_dirs):
        labels = [d.parent.parent.name if d.parts else str(d) for d in log_dirs]

    cmd_parts = [
        f"python {Path(__file__).name}",
        f"--log-dirs {' '.join(str(d) for d in log_dirs)}",
        f"--labels {' '.join(labels)}",
        f"--version {args.version}",
    ]
    if args.max_files > 0:
        cmd_parts.append(f"--max-files {args.max_files}")
    if args.teacher_max_files != 500:
        cmd_parts.append(f"--teacher-max-files {args.teacher_max_files}")
    cmd = " ".join(cmd_parts)

    b_label = args.b_label or pick_b_label(labels, log_dirs)
    sources = []
    part_a_rows = []

    human_row = {
        "label": "human",
        "files": 0,
        "kyoku": 0,
        "trainee_hora": 0,
        "call": 0,
        "call_pct": HUMAN_CALL_PCT,
        "riichi": 0,
        "riichi_pct": HUMAN_RIICHI_PCT,
        "dama": 0,
        "dama_pct": HUMAN_DAMA_PCT,
        "call_aka_pct": float("nan"),
        "games": 0,
        "skipped": 0,
    }
    part_a_rows.append(human_row)

    for label, log_dir in zip(labels, log_dirs):
        paths = resolve_log_paths(log_dir, args.max_files)
        print(f"[{label}] {log_dir} — {len(paths)} files", flush=True)
        if not paths:
            print(f"  該当なし")
            sources.append(SourceResult(label=label, log_dir=str(log_dir.resolve())))
            part_a_rows.append(build_part_a_row(label, PartAStats()))
            continue

        run_b = label == b_label
        part_a, part_b1, part_b2 = process_log_paths(paths, args.version, run_b=run_b)
        src = SourceResult(
            label=label,
            log_dir=str(log_dir.resolve()),
            part_a=part_a,
            part_b1=part_b1,
            part_b2=part_b2,
            run_b=run_b,
        )
        sources.append(src)
        part_a_rows.append(build_part_a_row(label, part_a))

        if label == "diag_b" and args.max_files == 0:
            expected_call = 165
            expected_hora = 3352
            got_call = part_a.call_hora
            got_hora = part_a.trainee_hora
            ok = got_call == expected_call and got_hora == expected_hora
            print(
                f"  cross-check diag_b: call_hora={got_call}/{expected_call} "
                f"trainee_hora={got_hora}/{expected_hora} {'OK' if ok else 'WARN'}"
            )

    teacher_note = None
    teacher_paths = sorted(glob.glob(args.teacher_glob))
    if args.teacher_max_files > 0:
        teacher_paths = teacher_paths[: args.teacher_max_files]
    if teacher_paths:
        print(f"[teacher] {args.teacher_glob} — {len(teacher_paths)} files", flush=True)
        try:
            t_a, _, _ = process_log_paths(teacher_paths, args.version, run_b=False)
            teacher_note = (
                f"`{args.teacher_glob}` サンプル {len(teacher_paths)} ファイル、"
                f"game {t_a.total_games}、総局 {t_a.total_kyoku}"
            )
            part_a_rows.append(build_part_a_row("teacher", t_a))
        except Exception as exc:
            teacher_note = f"ロード失敗（スキップ）: {exc}"
            print(f"  teacher skip: {exc}", flush=True)
    else:
        teacher_note = f"`{args.teacher_glob}` に該当ファイルなし（スキップ）"

    b_source = next((s for s in sources if s.label == b_label and s.part_a.n_files > 0), None)
    if b_source is None:
        b_source = next((s for s in sources if s.part_a.n_files > 0), None)
        if b_source and not b_source.run_b:
            print(f"[B] re-run {b_source.label} with Part B", flush=True)
            paths = resolve_log_paths(Path(b_source.log_dir), args.max_files)
            _, b1, b2 = process_log_paths(paths, args.version, run_b=True)
            b_source.part_b1 = b1
            b_source.part_b2 = b2
            b_source.run_b = True

    print_summary(part_a_rows, b_source)
    md = build_report(cmd, sources, part_a_rows, b_source, teacher_note)
    MD_PATH.write_text(md, encoding="utf-8")
    print(f"\nwritten: {MD_PATH}")


if __name__ == "__main__":
    main()
