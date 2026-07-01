#!/usr/bin/env python3
"""Step-wise B1+B2 diagnosis for mqw03 call-hora collapse (6000→8000)."""

import argparse
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

from call_channel_diag import (
    PartAStats,
    PartB1Stats,
    PartB2Stats,
    build_part_a_row,
    pct,
    process_game,
    process_log_paths,
    resolve_log_paths,
)
from count_aka_call_hora import (
    CALL_ACTION_IDS,
    check_kyoku_consecutive,
    collect_kyoku_action_flags,
    collect_kyoku_event_data,
)
from libriichi.dataset import GameplayLoader

MD_PATH = Path(__file__).resolve().parents[1] / "docs" / "mqw03_collapse_diag.md"
BASE_RUN_DIR = Path("/home/gamba/mahjong/runs/online_cql_mqw03")
DEFAULT_STEPS = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]

# cross-check vs mqw03_call_channel.md (Part A call_hora / trainee_hora %)
REFERENCE_CALL_PCT = {
    2000: 36.60,
    4000: 36.39,
    6000: 38.24,
    8000: 20.58,
    10000: 23.16,
    12000: 22.23,
    14000: 21.97,
    16000: 28.21,
}


@dataclass
class AuxStats:
    call_kyoku_call_actions: int = 0
    call_kyoku_houjuu: int = 0
    call_kyoku_opponent_tsumo: int = 0


@dataclass
class StepResult:
    step: int
    log_dir: str
    skipped: bool = False
    part_a: PartAStats = field(default_factory=PartAStats)
    part_b1: PartB1Stats = field(default_factory=PartB1Stats)
    part_b2: PartB2Stats = field(default_factory=PartB2Stats)
    aux: AuxStats = field(default_factory=AuxStats)


def collect_aux_for_game(game, path, aux: AuxStats):
    raw_at_kyoku = game.take_at_kyoku()
    if isinstance(raw_at_kyoku, (bytes, bytearray)):
        at_kyoku = np.frombuffer(raw_at_kyoku, dtype=np.uint8).astype(np.int64)
    else:
        at_kyoku = np.asarray(raw_at_kyoku, dtype=np.int64)
    if at_kyoku.size == 0:
        return
    actions = np.asarray(game.take_actions(), dtype=np.int64)
    player_id = game.take_player_id()

    try:
        check_kyoku_consecutive(at_kyoku)
    except (AssertionError, ValueError):
        return

    n_kyoku = int(at_kyoku.max()) + 1
    did_call, _, _ = collect_kyoku_action_flags(at_kyoku, actions)
    hora_by_kyoku, _ = collect_kyoku_event_data(path, player_id)

    call_actions_by_kyoku = {}
    for i in range(len(at_kyoku)):
        k = int(at_kyoku[i])
        a = int(actions[i])
        if a in CALL_ACTION_IDS:
            call_actions_by_kyoku[k] = call_actions_by_kyoku.get(k, 0) + 1

    for k in range(n_kyoku):
        if not did_call.get(k, False):
            continue
        aux.call_kyoku_call_actions += call_actions_by_kyoku.get(k, 0)
        horas = hora_by_kyoku.get(k, [])
        trainee_won = any(h["actor"] == player_id for h in horas)
        if trainee_won or not horas:
            continue
        houjuu = any(h["is_ron"] and h["target"] == player_id for h in horas)
        if houjuu:
            aux.call_kyoku_houjuu += 1
        else:
            aux.call_kyoku_opponent_tsumo += 1


def process_log_paths_with_aux(log_paths, version):
    loader = GameplayLoader(version=version, oracle=False, player_names=None)
    part_a = PartAStats()
    part_b1 = PartB1Stats()
    part_b2 = PartB2Stats()
    aux = AuxStats()

    for file_idx, path in enumerate(log_paths, start=1):
        games = loader.load_gz_log_files([str(path)])[0]
        for game in games:
            process_game(game, path, part_a, part_b1, part_b2, run_b=True)
        # GameplayLoader consumes game buffers in process_game; reload for aux pass.
        aux_games = loader.load_gz_log_files([str(path)])[0]
        for game in aux_games:
            collect_aux_for_game(game, path, aux)
        if file_idx % 500 == 0:
            print(f"  ... {file_idx}/{len(log_paths)} files", flush=True)

    part_a.n_files = len(log_paths)
    assert part_a.call_hora + part_a.riichi_hora + part_a.dama_hora == part_a.trainee_hora
    assert part_b2.call_kyoku + part_b2.menzen_kyoku == part_a.total_kyoku
    assert (
        part_b2.call_trainee_hora + part_b2.call_opponent_hora + part_b2.call_ryukyoku
        == part_b2.call_kyoku
    )
    assert aux.call_kyoku_houjuu + aux.call_kyoku_opponent_tsumo == part_b2.call_opponent_hora

    return part_a, part_b1, part_b2, aux


def step_log_dir(base_dir: Path, step: int) -> Path:
    return base_dir / f"test_play_step{step}"


def build_step_metrics(result: StepResult) -> dict:
    a = result.part_a
    b1 = result.part_b1
    b2 = result.part_b2
    aux = result.aux
    row_a = build_part_a_row(f"step{result.step}", a)
    return {
        "step": result.step,
        "files": a.n_files,
        "kyoku": a.total_kyoku,
        "call_pct": row_a["call_pct"],
        "riichi_pct": row_a["riichi_pct"],
        "p_call_legal": pct(b1.call_moves, b1.call_legal_moves),
        "p_chi_legal": pct(b1.chi_moves, b1.chi_legal_moves),
        "p_pon_legal": pct(b1.pon_moves, b1.pon_legal_moves),
        "call_kyoku": b2.call_kyoku,
        "menzen_kyoku": b2.menzen_kyoku,
        "p_hora_call": pct(b2.call_trainee_hora, b2.call_kyoku),
        "p_hora_menzen": pct(b2.menzen_trainee_hora, b2.menzen_kyoku),
        "call_ryukyoku_pct": pct(b2.call_ryukyoku, b2.call_kyoku),
        "call_opponent_pct": pct(b2.call_opponent_hora, b2.call_kyoku),
        "call_houjuu_pct": pct(aux.call_kyoku_houjuu, b2.call_kyoku),
        "call_opponent_tsumo_pct": pct(aux.call_kyoku_opponent_tsumo, b2.call_kyoku),
        "avg_calls_per_call_kyoku": (
            aux.call_kyoku_call_actions / b2.call_kyoku if b2.call_kyoku else 0.0
        ),
        "trainee_hora": a.trainee_hora,
        "call_hora": a.call_hora,
        "skipped": result.skipped,
    }


def fmt_pp_delta(a, b):
    return f"{b - a:+.2f}pp"


def build_collapse_conclusion(m_by_step: dict[int, dict]) -> list[str]:
    lines = []
    if 6000 not in m_by_step or 8000 not in m_by_step:
        lines.append("- step6000 または step8000 が欠損のため崩落区間分析不可。")
        return lines

    s6 = m_by_step[6000]
    s8 = m_by_step[8000]
    d_call_pct = s8["call_pct"] - s6["call_pct"]
    d_p_call = s8["p_call_legal"] - s6["p_call_legal"]
    d_p_hora_call = s8["p_hora_call"] - s6["p_hora_call"]
    d_ryu = s8["call_ryukyoku_pct"] - s6["call_ryukyoku_pct"]
    d_opp = s8["call_opponent_pct"] - s6["call_opponent_pct"]
    d_houjuu = s8["call_houjuu_pct"] - s6["call_houjuu_pct"]

    lines.append(
        f"- **崩落区間 step6000→8000**: 副露和了率 {s6['call_pct']:.2f}% → "
        f"{s8['call_pct']:.2f}% ({fmt_pp_delta(s6['call_pct'], s8['call_pct'])})、"
        f"立直和了率 {s6['riichi_pct']:.2f}% → {s8['riichi_pct']:.2f}% "
        f"({fmt_pp_delta(s6['riichi_pct'], s8['riichi_pct'])})。"
    )
    lines.append(
        f"- B1: P(call|legal) {s6['p_call_legal']:.2f}% → {s8['p_call_legal']:.2f}% "
        f"({fmt_pp_delta(s6['p_call_legal'], s8['p_call_legal'])})、"
        f"P(chi|legal) {fmt_pp_delta(s6['p_chi_legal'], s8['p_chi_legal'])}、"
        f"P(pon|legal) {fmt_pp_delta(s6['p_pon_legal'], s8['p_pon_legal'])}。"
    )
    lines.append(
        f"- B2: P(和了|副露局) {s6['p_hora_call']:.2f}% → {s8['p_hora_call']:.2f}% "
        f"({fmt_pp_delta(s6['p_hora_call'], s8['p_hora_call'])})、"
        f"流局 {fmt_pp_delta(s6['call_ryukyoku_pct'], s8['call_ryukyoku_pct'])}、"
        f"他家和了 {fmt_pp_delta(s6['call_opponent_pct'], s8['call_opponent_pct'])} "
        f"(放銃 {fmt_pp_delta(s6['call_houjuu_pct'], s8['call_houjuu_pct'])}、"
        f"横移動 {fmt_pp_delta(s6['call_opponent_tsumo_pct'], s8['call_opponent_tsumo_pct'])})。"
    )
    lines.append(
        f"- 副露局平均鳴き回数: {s6['avg_calls_per_call_kyoku']:.3f} → "
        f"{s8['avg_calls_per_call_kyoku']:.3f} "
        f"({s8['avg_calls_per_call_kyoku'] - s6['avg_calls_per_call_kyoku']:+.3f})。"
    )

    contributions = [
        ("P(call|legal) 低下", abs(d_p_call), d_p_call),
        ("P(和了|副露局) 低下", abs(d_p_hora_call), d_p_hora_call),
        ("副露局流局率上昇", abs(d_ryu), d_ryu),
        ("副露局他家和了率上昇", abs(d_opp), d_opp),
        ("副露局放銃率上昇", abs(d_houjuu), d_houjuu),
    ]
    contributions.sort(key=lambda x: x[1], reverse=True)

    p_call_big_drop = d_p_call < -3.0
    p_hora_call_drop = d_p_hora_call < -2.0
    p_call_flat = abs(d_p_call) <= 3.0

    lines.append("")
    lines.append("### 決定木判定")

    if p_call_big_drop and not p_hora_call_drop:
        lines.append(
            "- **主因: 行動抑制** — P(call|legal) が 6000→8000 で大きく低下。"
            " CQL/報酬が鳴き行動を直接潰している。"
            " 対処: CQL 強度の step 依存 or 鳴き行動への保護。"
        )
    elif p_call_flat and p_hora_call_drop and d_ryu > d_opp:
        lines.append(
            "- **主因: conversion 劣化（手作り/テンパイ未到達）** — "
            "P(call|legal) は横ばいだが P(和了|副露局) が低下し流局が増加。"
            " 価値推定・手作り側の問題。"
        )
    elif p_call_flat and p_hora_call_drop and d_opp >= d_ryu:
        if d_houjuu > d_ryu * 0.5:
            lines.append(
                "- **主因: 守備崩壊** — P(和了|副露局) 低下に他家和了/放銃が増加。"
                " 鳴いて守備が崩れている。"
            )
        else:
            lines.append(
                "- **主因: conversion 劣化 + 守備** — 他家和了増が流局増と同程度以上。"
                " 手作りと守備の複合。"
            )
    elif p_call_big_drop and p_hora_call_drop:
        lines.append(
            "- **主因: 複合** — 行動抑制と conversion 劣化が同時に発生。"
        )
        ranked = ", ".join(f"{name}({abs(d):.2f}pp)" for name, _, d in contributions[:3] if d != 0)
        lines.append(f"  - 寄与順: {ranked}")
        if d_p_call < 0 and abs(d_p_call) >= max(abs(d_p_hora_call), abs(d_ryu), abs(d_opp)):
            lines.append("  - **行動抑制寄与が最大** → CQL/鳴き保護を優先。")
        elif d_ryu > d_opp:
            lines.append("  - **conversion(流局)寄与が大** → 価値推定/手作りを優先。")
        else:
            lines.append("  - **守備(他家和了)寄与が大** → 鳴き後の守備判断を優先。")
    else:
        lines.append(
            "- 6000→8000 の指標変化が決定木の典型パターンに完全一致しない。"
            " 上記差分を参照して判断。"
        )

    return lines


def build_report(cmd: str, base_dir: Path, results: list[StepResult], metrics: list[dict]) -> str:
    m_by_step = {m["step"]: m for m in metrics}
    skipped = [r for r in results if r.skipped]

    lines = [
        "# mqw03 副露和了崩落診断 (step-wise B1+B2)",
        "",
        f"**日付:** {date.today().isoformat()}",
        "",
        "## 実行条件",
        "",
        f"- コマンド: `{cmd}`",
        f"- ベース: `{base_dir}`",
        "",
        "### 対象 step",
        "",
        "| step | ディレクトリ | ファイル数 | 状態 |",
        "|---|---|---:|---|",
    ]

    for r in results:
        status = "skip（ディレクトリなし/空）" if r.skipped else "OK"
        lines.append(
            f"| {r.step} | `{r.log_dir}` | {r.part_a.n_files} | {status} |"
        )

    if skipped:
        lines.append("")
        lines.append(f"※ skip: {', '.join(f'step{s.step}' for s in skipped)}")

    lines.extend([
        "",
        "## 時系列サマリ（B1+B2 + Part A cross-check）",
        "",
        "副露和了% は Part A（trainee和了に対する副露和了比率）。"
        " mqw03_call_channel.md と整合確認用。",
        "",
        "| step | 副露和了% | ref% | Δref | P(call\\|legal) | P(chi\\|legal) | "
        "P(pon\\|legal) | P(和了\\|副露局) | P(和了\\|門前局) | 副露局流局% | "
        "他家和了% | 放銃% | 平均鳴き/副露局 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for m in metrics:
        ref = REFERENCE_CALL_PCT.get(m["step"])
        if ref is not None:
            d_ref = m["call_pct"] - ref
            ref_s = f"{ref:.2f}%"
            d_ref_s = f"{d_ref:+.2f}pp"
        else:
            ref_s = "—"
            d_ref_s = "—"
        lines.append(
            f"| step{m['step']} | {m['call_pct']:.2f}% | {ref_s} | {d_ref_s} | "
            f"{m['p_call_legal']:.2f}% | {m['p_chi_legal']:.2f}% | {m['p_pon_legal']:.2f}% | "
            f"{m['p_hora_call']:.2f}% | {m['p_hora_menzen']:.2f}% | "
            f"{m['call_ryukyoku_pct']:.2f}% | {m['call_opponent_pct']:.2f}% | "
            f"{m['call_houjuu_pct']:.2f}% | {m['avg_calls_per_call_kyoku']:.3f} |"
        )

    if 6000 in m_by_step and 8000 in m_by_step:
        s6, s8 = m_by_step[6000], m_by_step[8000]
        lines.extend([
            "",
            "## 崩落区間 step6000→8000（強調）",
            "",
            "| 指標 | step6000 | step8000 | Δ |",
            "|---|---:|---:|---:|",
            f"| 副露和了% | {s6['call_pct']:.2f}% | {s8['call_pct']:.2f}% | "
            f"{fmt_pp_delta(s6['call_pct'], s8['call_pct'])} |",
            f"| 立直和了% | {s6['riichi_pct']:.2f}% | {s8['riichi_pct']:.2f}% | "
            f"{fmt_pp_delta(s6['riichi_pct'], s8['riichi_pct'])} |",
            f"| P(call\\|legal) | {s6['p_call_legal']:.2f}% | {s8['p_call_legal']:.2f}% | "
            f"{fmt_pp_delta(s6['p_call_legal'], s8['p_call_legal'])} |",
            f"| P(chi\\|legal) | {s6['p_chi_legal']:.2f}% | {s8['p_chi_legal']:.2f}% | "
            f"{fmt_pp_delta(s6['p_chi_legal'], s8['p_chi_legal'])} |",
            f"| P(pon\\|legal) | {s6['p_pon_legal']:.2f}% | {s8['p_pon_legal']:.2f}% | "
            f"{fmt_pp_delta(s6['p_pon_legal'], s8['p_pon_legal'])} |",
            f"| P(和了\\|副露局) | {s6['p_hora_call']:.2f}% | {s8['p_hora_call']:.2f}% | "
            f"{fmt_pp_delta(s6['p_hora_call'], s8['p_hora_call'])} |",
            f"| P(和了\\|門前局) | {s6['p_hora_menzen']:.2f}% | {s8['p_hora_menzen']:.2f}% | "
            f"{fmt_pp_delta(s6['p_hora_menzen'], s8['p_hora_menzen'])} |",
            f"| 副露局流局% | {s6['call_ryukyoku_pct']:.2f}% | {s8['call_ryukyoku_pct']:.2f}% | "
            f"{fmt_pp_delta(s6['call_ryukyoku_pct'], s8['call_ryukyoku_pct'])} |",
            f"| 他家和了% | {s6['call_opponent_pct']:.2f}% | {s8['call_opponent_pct']:.2f}% | "
            f"{fmt_pp_delta(s6['call_opponent_pct'], s8['call_opponent_pct'])} |",
            f"| 放銃% | {s6['call_houjuu_pct']:.2f}% | {s8['call_houjuu_pct']:.2f}% | "
            f"{fmt_pp_delta(s6['call_houjuu_pct'], s8['call_houjuu_pct'])} |",
            f"| 平均鳴き/副露局 | {s6['avg_calls_per_call_kyoku']:.3f} | "
            f"{s8['avg_calls_per_call_kyoku']:.3f} | "
            f"{s8['avg_calls_per_call_kyoku'] - s6['avg_calls_per_call_kyoku']:+.3f} |",
        ])

    lines.extend([
        "",
        "## 結論",
        "",
        *build_collapse_conclusion(m_by_step),
        "",
        "### sanity",
        "",
        "- B2 母数: 各 step で 副露局+門前局 == trainee が打った局数（total_kyoku）。",
        "- 排他: 副露局結末 trainee和了+他家和了+流局 == 副露局数。",
        "- 放銃+横移動 == 他家和了（副露局）。",
        "",
        "### action ID",
        "",
        "- 副露: trainee action ∈ {38, 39, 40, 41, 42}",
        "- 立直: trainee action == 37",
        "",
    ])
    return "\n".join(lines)


def print_summary_table(metrics: list[dict]):
    print("=== mqw03_collapse_diag (step × B1+B2) ===")
    print()
    header = (
        "| step | 副露和了% | P(call|legal) | P(chi|legal) | P(pon|legal) | "
        "P(hora|副露) | P(hora|門前) | 流局% | 他家% | 放銃% | avg鳴き |"
    )
    print(header)
    for m in metrics:
        if m["skipped"]:
            print(f"| step{m['step']} | — (skip) |")
            continue
        print(
            f"| step{m['step']} | {m['call_pct']:.2f}% | {m['p_call_legal']:.2f}% | "
            f"{m['p_chi_legal']:.2f}% | {m['p_pon_legal']:.2f}% | "
            f"{m['p_hora_call']:.2f}% | {m['p_hora_menzen']:.2f}% | "
            f"{m['call_ryukyoku_pct']:.2f}% | {m['call_opponent_pct']:.2f}% | "
            f"{m['call_houjuu_pct']:.2f}% | {m['avg_calls_per_call_kyoku']:.3f} |"
        )


def main():
    parser = argparse.ArgumentParser(description="mqw03 step-wise call collapse B1+B2 diag")
    parser.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--base-dir", type=Path, default=BASE_RUN_DIR)
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0, help="0=all")
    args = parser.parse_args()

    base_dir = args.base_dir

    cmd_parts = [
        f"python {Path(__file__).name}",
        f"--base-dir {base_dir}",
        f"--version {args.version}",
        f"--steps {' '.join(str(s) for s in args.steps)}",
    ]
    if args.max_files > 0:
        cmd_parts.append(f"--max-files {args.max_files}")
    cmd = " ".join(cmd_parts)

    results: list[StepResult] = []
    metrics: list[dict] = []

    for step in sorted(args.steps):
        log_dir = base_dir / f"test_play_step{step}"
        print(f"[step{step}] {log_dir}", flush=True)
        paths = resolve_log_paths(log_dir, args.max_files)
        if not paths:
            print("  skip (empty/missing)", flush=True)
            res = StepResult(step=step, log_dir=str(log_dir.resolve()), skipped=True)
            results.append(res)
            metrics.append(build_step_metrics(res))
            continue

        print(f"  {len(paths)} files", flush=True)
        part_a, part_b1, part_b2, aux = process_log_paths_with_aux(paths, args.version)
        res = StepResult(
            step=step,
            log_dir=str(log_dir.resolve()),
            part_a=part_a,
            part_b1=part_b1,
            part_b2=part_b2,
            aux=aux,
        )
        results.append(res)
        m = build_step_metrics(res)
        metrics.append(m)

        ref = REFERENCE_CALL_PCT.get(step)
        if ref is not None:
            ok = abs(m["call_pct"] - ref) < 0.05
            print(
                f"  cross-check 副露和了%: {m['call_pct']:.2f}% vs ref {ref:.2f}% "
                f"({'OK' if ok else 'WARN'})",
                flush=True,
            )

    print_summary_table(metrics)
    md = build_report(cmd, base_dir, results, metrics)
    MD_PATH.write_text(md, encoding="utf-8")
    print(f"\nwritten: {MD_PATH}")


if __name__ == "__main__":
    main()
