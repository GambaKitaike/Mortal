#!/usr/bin/env python3
"""Phase 4d: chip realization from aka-held kyoku (mortal seat, end-snapshot aka_held)."""

import argparse
import gzip
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_scripts = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts))
sys.path.insert(0, str(_scripts.parents[1] / "mortal"))
from libriichi.state import PlayerState

from preprocess_chips import count_hand_aka, hora_chip_deltas, open_log

CALL_TYPES = frozenset({"chi", "pon", "daiminkan"})


@dataclass
class KyokuMetrics:
    aka_held_end: int = 0
    chip_delta: int = 0
    won: bool = False
    win_num_aka: int = 0
    did_riichi: bool = False
    did_call: bool = False


@dataclass
class Aggregate:
    aka_held_kyoku: int = 0
    chip_realize: int = 0
    aka_chip_realize: int = 0
    win: int = 0
    riichi_win: int = 0
    call_win: int = 0
    chip_sum: int = 0


def process_eval_file(path: Path) -> tuple[list[KyokuMetrics], int]:
    with open_log(path) as f:
        events = [json.loads(line) for line in f if line.strip()]

    mortal_id = 0
    for ev in events:
        if ev.get("type") == "start_game":
            names = ev.get("names") or []
            if "mortal" not in names:
                raise ValueError(f"'mortal' not in start_game names: {names} @ {path}")
            mortal_id = names.index("mortal")
            break

    states = [PlayerState(i) for i in range(4)]
    kyoku_idx = -1
    per_kyoku: dict[int, list[KyokuMetrics]] = {}
    cur_riichi = False
    cur_call = False
    win_num_aka: dict[int, int] = {}

    for ev in events:
        et = ev.get("type")
        if et == "start_kyoku":
            kyoku_idx += 1
            cur_riichi = False
            cur_call = False

        if kyoku_idx >= 0:
            if et == "reach" and ev.get("actor") == mortal_id:
                cur_riichi = True
            if et in CALL_TYPES and ev.get("actor") == mortal_id:
                cur_call = True

        if kyoku_idx >= 0 and et == "hora":
            actor = ev["actor"]
            is_ron = actor != ev["target"]
            ura = ev.get("ura_markers") or ev.get("ura_indicators") or []
            try:
                detail = states[actor].agari_detail(is_ron, ura)
            except Exception:
                pass
            else:
                if actor == mortal_id:
                    win_num_aka[kyoku_idx] = detail.num_aka
                chip_d = hora_chip_deltas(ev, detail)
                m = per_kyoku.setdefault(kyoku_idx, KyokuMetrics())
                m.chip_delta += chip_d[mortal_id]

        if kyoku_idx >= 0 and et in ("hora", "ryukyoku"):
            m = per_kyoku.setdefault(kyoku_idx, KyokuMetrics())
            m.aka_held_end = count_hand_aka(states[mortal_id])
            m.did_riichi = cur_riichi
            m.did_call = cur_call
            if kyoku_idx in win_num_aka:
                m.won = True
                m.win_num_aka = win_num_aka[kyoku_idx]

        for s in states:
            s.update(json.dumps(ev, separators=(",", ":")))

    return list(per_kyoku.values()), mortal_id


def add_kyoku(agg: Aggregate, m: KyokuMetrics) -> None:
    if m.aka_held_end <= 0:
        return
    agg.aka_held_kyoku += 1
    agg.chip_sum += m.chip_delta
    if m.chip_delta > 0:
        agg.chip_realize += 1
    if m.won and m.win_num_aka > 0:
        agg.aka_chip_realize += 1
    if m.won:
        agg.win += 1
        if m.did_riichi:
            agg.riichi_win += 1
        if m.did_call:
            agg.call_win += 1


def analyze_eval_dir(log_dir: Path) -> Aggregate:
    files = sorted(log_dir.glob("*.json.gz")) + sorted(log_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no logs in {log_dir}")
    agg = Aggregate()
    for path in files:
        metrics, _ = process_eval_file(path)
        for m in metrics:
            add_kyoku(agg, m)
    return agg


def analyze_human_mjson(data_dir: Path, limit: int | None = None) -> Aggregate:
    files = sorted(data_dir.glob("*.mjson"))
    if limit is not None:
        files = files[:limit]
    agg = Aggregate()
    for path in files:
        metrics = process_human_file(path)
        for m in metrics:
            add_kyoku(agg, m)
    return agg


def process_human_file(path: Path) -> list[KyokuMetrics]:
    """All 4 seats; one KyokuMetrics row per (kyoku, player) with aka_held_end."""
    with open_log(path) as f:
        events = [json.loads(line) for line in f if line.strip()]

    states = [PlayerState(i) for i in range(4)]
    kyoku_idx = -1
    per_player: dict[tuple[int, int], KyokuMetrics] = {}
    riichi: dict[int, list[bool]] = {}
    call: dict[int, list[bool]] = {}
    win_num_aka: dict[tuple[int, int], int] = {}
    chip_delta: dict[tuple[int, int], int] = {}

    for ev in events:
        et = ev.get("type")
        if et == "start_kyoku":
            kyoku_idx += 1
            riichi[kyoku_idx] = [False] * 4
            call[kyoku_idx] = [False] * 4

        if kyoku_idx >= 0:
            actor = ev.get("actor")
            if et == "reach" and actor is not None:
                riichi[kyoku_idx][actor] = True
            if et in CALL_TYPES and actor is not None:
                call[kyoku_idx][actor] = True

        if kyoku_idx >= 0 and et == "hora":
            actor = ev["actor"]
            is_ron = actor != ev["target"]
            ura = ev.get("ura_markers") or ev.get("ura_indicators") or []
            try:
                detail = states[actor].agari_detail(is_ron, ura)
            except Exception:
                pass
            else:
                win_num_aka[(kyoku_idx, actor)] = detail.num_aka
                chip_d = hora_chip_deltas(ev, detail)
                for p in range(4):
                    key = (kyoku_idx, p)
                    chip_delta[key] = chip_delta.get(key, 0) + chip_d[p]

        if kyoku_idx >= 0 and et in ("hora", "ryukyoku"):
            for p in range(4):
                key = (kyoku_idx, p)
                m = KyokuMetrics()
                m.aka_held_end = count_hand_aka(states[p])
                m.chip_delta = chip_delta.get(key, 0)
                m.did_riichi = riichi[kyoku_idx][p]
                m.did_call = call[kyoku_idx][p]
                if key in win_num_aka:
                    m.won = True
                    m.win_num_aka = win_num_aka[key]
                per_player[key] = m

        for s in states:
            s.update(json.dumps(ev, separators=(",", ":")))

    return list(per_player.values())


def analyze_human_npz(chips_dir: Path) -> Aggregate:
    """chip_realize / avg chips / win rate only (no num_aka or riichi/call path)."""
    agg = Aggregate()
    for path in sorted(chips_dir.glob("*.npz")):
        data = np.load(path)
        aka = data["aka_held"]
        chips = data["chips"]
        won = data["won"]
        n = aka.shape[0]
        for k in range(n):
            for p in range(4):
                if aka[k, p] <= 0:
                    continue
                agg.aka_held_kyoku += 1
                cd = int(chips[k, p])
                agg.chip_sum += cd
                if cd > 0:
                    agg.chip_realize += 1
                if won[k, p]:
                    agg.win += 1
    return agg


def pct(num: int, den: int) -> str:
    if den == 0:
        return "—"
    return f"{100.0 * num / den:.2f}%"


def avg_chips(agg: Aggregate) -> str:
    if agg.aka_held_kyoku == 0:
        return "—"
    return f"{agg.chip_sum / agg.aka_held_kyoku:.3f}"


def row_cells(agg: Aggregate) -> dict[str, str]:
    den = agg.aka_held_kyoku
    return {
        "aka_held": str(den),
        "chip_realize": pct(agg.chip_realize, den),
        "aka_chip": pct(agg.aka_chip_realize, den),
        "win": pct(agg.win, den),
        "riichi_win": pct(agg.riichi_win, den),
        "call_win": pct(agg.call_win, den),
        "avg_chips": avg_chips(agg),
    }


def print_report(results: dict[str, Aggregate], human: Aggregate | None, human_npz: Aggregate | None) -> None:
    cols = ["人間(2009)"] if human or human_npz else []
    cols.extend(results.keys())
    header = "| 指標 | " + " | ".join(cols) + " |"
    ncols = len(cols)

    def col_aggs() -> list[Aggregate]:
        out: list[Aggregate] = []
        if human:
            out.append(human)
        elif human_npz:
            out.append(human_npz)
        out.extend(results.values())
        return out

    aggs = col_aggs()
    rows = [row_cells(a) for a in aggs]

    def line(label: str, key: str) -> str:
        vals = [r[key] for r in rows]
        return f"| {label} | " + " | ".join(vals) + " |"

    print("# Phase 4d — 赤保持→チップ実現")
    print()
    print("集計: mortal 席（AI）/ 全席（人間2009）。赤保持局 = 局終了スナップショット `aka_held>0`（preprocess 同定義）。")
    print()
    print("## 母集団")
    print()
    print(header)
    print("|---|" + "|".join(["---:"] * ncols) + "|")
    print(line("赤保持局数", "aka_held"))
    print()
    print("## chip_realize_rate / aka_chip_realize_rate")
    print()
    print(header)
    print("|---|" + "|".join(["---:"] * ncols) + "|")
    print(line("chip_realize_rate", "chip_realize"))
    print(line("aka_chip_realize_rate", "aka_chip"))
    print()
    print("## 赤保持局 — 和了経路内訳")
    print()
    print(header)
    print("|---|" + "|".join(["---:"] * ncols) + "|")
    print(line("和了率", "win"))
    print(line("立直和了率", "riichi_win"))
    print(line("鳴き和了率", "call_win"))
    print()
    print("## 赤保持局あたり平均チップ枚数（net）")
    print()
    print(header)
    print("|---|" + "|".join(["---:"] * ncols) + "|")
    print(line("平均チップ枚数", "avg_chips"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-mjson", type=Path, default=Path("/home/gamba/mahjong/data/tenhou/2009"))
    ap.add_argument("--human-chips", type=Path, default=Path("/home/gamba/mahjong/data/tenhou/chips"))
    ap.add_argument("--human-limit", type=int, default=None)
    ap.add_argument("--skip-human-mjson", action="store_true")
    ap.add_argument(
        "--eval",
        nargs="*",
        default=[
            "lo=0.0:/home/gamba/mahjong/runs/phase4d/eval/phase4d_lo00/1v3",
            "lo=0.3:/home/gamba/mahjong/runs/phase4d/eval/phase4d_lo03/1v3",
            "lo=0.6:/home/gamba/mahjong/runs/phase4d/eval/phase4d_lo06/1v3",
        ],
    )
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    results: dict[str, Aggregate] = {}
    for item in args.eval:
        name, _, log_dir = item.partition(":")
        agg = analyze_eval_dir(Path(log_dir))
        results[name] = agg
        print(
            f"{name}: aka_held={agg.aka_held_kyoku} "
            f"chip_realize={pct(agg.chip_realize, agg.aka_held_kyoku)} "
            f"aka_chip={pct(agg.aka_chip_realize, agg.aka_held_kyoku)}",
            file=sys.stderr,
        )

    human: Aggregate | None = None
    human_npz: Aggregate | None = None
    if not args.skip_human_mjson and args.human_mjson.is_dir():
        print("Processing human mjson...", file=sys.stderr)
        human = analyze_human_mjson(args.human_mjson, args.human_limit)
        print(f"human mjson: aka_held={human.aka_held_kyoku}", file=sys.stderr)
    elif args.human_chips.is_dir():
        human_npz = analyze_human_npz(args.human_chips)
        print(f"human npz (partial): aka_held={human_npz.aka_held_kyoku}", file=sys.stderr)

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(results, human, human_npz)
    report = buf.getvalue()
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
