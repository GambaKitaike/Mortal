#!/usr/bin/env python3
"""Collect metrics for online CQL min_q_weight sweep (numbers only)."""

import argparse
import gzip
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_scripts = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts))
sys.path.insert(0, str(_scripts.parents[1] / "mortal"))

from analyze_aka_conditional import analyze_dir, delta_pp, rate
from analyze_chip_realize import Aggregate, process_eval_file


def calc_beta_sel(step: int) -> float:
    warmup, ramp, mx = 2000, 2000, 0.3
    if step < warmup:
        return 0.0
    if step < warmup + ramp:
        return (step - warmup) / ramp * mx
    return mx


@dataclass
class StepMetrics:
    step: int
    min_q_weight: float
    fuuro_delta_pp: float | None = None
    fuuro_with_aka: float | None = None
    fuuro_without_aka: float | None = None
    n_aka_yes: int = 0
    n_aka_no: int = 0
    n_aka_held: int = 0
    call_win_rate: float | None = None
    chip_realize_rate: float | None = None
    riichi_rate: float | None = None
    houjuu_rate: float | None = None
    fuuro_rate: float | None = None
    avg_rank: float | None = None
    ryukyoku_rate: float | None = None
    beta_sel: float = 0.0


def parse_trainer_result(result_path: Path) -> dict:
    text = result_path.read_text()
    out = {}
    m = re.search(
        r"agari=([\d.]+)% houjuu=([\d.]+)% fuuro=([\d.]+)% "
        r"riichi=([\d.]+)% ryukyoku=([\d.]+)%",
        text,
    )
    if m:
        out["agari"] = float(m.group(1))
        out["houjuu"] = float(m.group(2))
        out["fuuro"] = float(m.group(3))
        out["riichi"] = float(m.group(4))
        out["ryukyoku"] = float(m.group(5))
    return out


def parse_avg_rank(trainer_log: Path, step: int) -> float | None:
    if not trainer_log.exists():
        return None
    lines = trainer_log.read_text().splitlines()
    for i, line in enumerate(lines):
        if "test_play behavior:" in line:
            ctx = "\n".join(lines[max(0, i - 10) : i + 1])
            m_step = re.findall(r"total steps:\s*([\d,]+)", ctx)
            if not m_step:
                continue
            s = int(m_step[-1].replace(",", ""))
            if s == step:
                m_rank = re.search(r"avg rank: ([\d.]+)", ctx)
                if m_rank:
                    return float(m_rank.group(1))
    return None


def analyze_aka_conditional(log_dir: Path) -> tuple[float, float, float, int, int]:
    stats = analyze_dir(log_dir)
    wa, wo = stats.with_aka, stats.without_aka
    w = rate(wa.call, wa.rounds)
    wo_r = rate(wo.call, wo.rounds)
    return w, wo_r, delta_pp(wa, wo, "call"), wa.rounds, wo.rounds


def analyze_chip(log_dir: Path) -> tuple[int, float, float]:
    agg = Aggregate()
    for path in sorted(log_dir.glob("*.json.gz")) + sorted(log_dir.glob("*.json")):
        per_kyoku, _ = process_eval_file(path)
        for km in per_kyoku:
            if km.aka_held_end <= 0:
                continue
            agg.aka_held_kyoku += 1
            if km.chip_delta > 0:
                agg.chip_realize += 1
            if km.won and km.did_call:
                agg.call_win += 1
    n = agg.aka_held_kyoku
    chip = 100.0 * agg.chip_realize / n if n else 0.0
    call_win = 100.0 * agg.call_win / n if n else 0.0
    return n, chip, call_win


def collect_run(run_dir: Path, min_q_weight: float, steps: list[int]) -> list[StepMetrics]:
    rows = []
    trainer_log = run_dir / "logs" / "trainer.log"
    for step in steps:
        log_dir = run_dir / f"test_play_step{step}"
        result = run_dir / f"result_step{step}.txt"
        if not log_dir.exists() or not result.exists():
            print(f"WARN missing {run_dir.name} step{step}", file=sys.stderr)
            continue
        beh = parse_trainer_result(result)
        w, wo, delta, n_yes, n_no = analyze_aka_conditional(log_dir)
        n_held, chip, call_win = analyze_chip(log_dir)
        rows.append(
            StepMetrics(
                step=step,
                min_q_weight=min_q_weight,
                fuuro_delta_pp=delta,
                fuuro_with_aka=w,
                fuuro_without_aka=wo,
                n_aka_yes=n_yes,
                n_aka_no=n_no,
                n_aka_held=n_held,
                call_win_rate=call_win,
                chip_realize_rate=chip,
                riichi_rate=beh.get("riichi"),
                houjuu_rate=beh.get("houjuu"),
                fuuro_rate=beh.get("fuuro"),
                avg_rank=parse_avg_rank(trainer_log, step),
                ryukyoku_rate=beh.get("ryukyoku"),
                beta_sel=calc_beta_sel(step),
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/home/gamba/mahjong/runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.runs_root)
    specs = [
        ("online_cql_mqw03", 0.3),
        ("online_cql_mqw05", 0.5),
        ("online_cql_mqw10", 1.0),
    ]
    all_rows: list[StepMetrics] = []
    for name, w in specs:
        all_rows.extend(collect_run(root / name, w, [2000, 4000, 6000]))

    lines = []
    lines.append("| min_q_weight | step | beta_sel | 副露Δ(pp) | 鳴き和了率 | chip実現率 | 立直率 | 放銃率 | 全体副露率 | avg_rank | 流局率 | n_aka_yes | n_aka_no | n_aka_held |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(all_rows, key=lambda x: (x.min_q_weight, x.step)):
        lines.append(
            f"| {r.min_q_weight} | {r.step} | {r.beta_sel:.3f} | "
            f"{r.fuuro_delta_pp:+.2f} | {r.call_win_rate:.2f}% | {r.chip_realize_rate:.2f}% | "
            f"{r.riichi_rate:.2f}% | {r.houjuu_rate:.2f}% | {r.fuuro_rate:.2f}% | "
            f"{r.avg_rank:.4f} | {r.ryukyoku_rate:.2f}% | {r.n_aka_yes} | {r.n_aka_no} | {r.n_aka_held} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
