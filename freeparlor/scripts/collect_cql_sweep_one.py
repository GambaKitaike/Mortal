#!/usr/bin/env python3
"""Collect metrics for one sweep checkpoint (fast, for parallel use)."""
import json
import re
import sys
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


def parse_result(path: Path) -> dict:
    text = path.read_text()
    m = re.search(
        r"agari=([\d.]+)% houjuu=([\d.]+)% fuuro=([\d.]+)% "
        r"riichi=([\d.]+)% ryukyoku=([\d.]+)%",
        text,
    )
    if not m:
        return {}
    return {
        "agari": float(m.group(1)),
        "houjuu": float(m.group(2)),
        "fuuro": float(m.group(3)),
        "riichi": float(m.group(4)),
        "ryukyoku": float(m.group(5)),
    }


def parse_avg_rank(trainer_log: Path, step: int) -> float | None:
    if not trainer_log.exists():
        return None
    lines = trainer_log.read_text().splitlines()
    for i, line in enumerate(lines):
        if "test_play behavior:" not in line:
            continue
        ctx = "\n".join(lines[max(0, i - 10) : i + 1])
        steps = re.findall(r"total steps:\s*([\d,]+)", ctx)
        if not steps:
            continue
        if int(steps[-1].replace(",", "")) == step:
            m = re.search(r"avg rank: ([\d.]+)", ctx)
            if m:
                return float(m.group(1))
    return None


def analyze_chip(log_dir: Path) -> tuple[int, float, float]:
    agg = Aggregate()
    for path in sorted(log_dir.glob("*.json.gz")):
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


def main():
    run_dir = Path(sys.argv[1])
    step = int(sys.argv[2])
    min_q_weight = float(sys.argv[3])
    log_dir = run_dir / f"test_play_step{step}"
    beh = parse_result(run_dir / f"result_step{step}.txt")
    stats = analyze_dir(log_dir)
    wa, wo = stats.with_aka, stats.without_aka
    n_held, chip, call_win = analyze_chip(log_dir)
    out = {
        "min_q_weight": min_q_weight,
        "step": step,
        "beta_sel": calc_beta_sel(step),
        "fuuro_delta_pp": delta_pp(wa, wo, "call"),
        "call_win_rate": call_win,
        "chip_realize_rate": chip,
        "riichi_rate": beh.get("riichi"),
        "houjuu_rate": beh.get("houjuu"),
        "fuuro_rate": beh.get("fuuro"),
        "avg_rank": parse_avg_rank(run_dir / "logs" / "trainer.log", step),
        "ryukyoku_rate": beh.get("ryukyoku"),
        "n_aka_yes": wa.rounds,
        "n_aka_no": wo.rounds,
        "n_aka_held": n_held,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
