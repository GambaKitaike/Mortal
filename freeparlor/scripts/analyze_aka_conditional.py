#!/usr/bin/env python3
"""Phase 4: aka-conditional playstyle analysis from 1v3 eval logs."""

import argparse
import gzip
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mortal"))
from libriichi.state import PlayerState

AKA_TILES = frozenset({"5mr", "5pr", "5sr"})
CALL_TYPES = frozenset({"chi", "pon", "daiminkan"})
FUURO_TYPES = frozenset({"chi", "pon", "daiminkan", "ankan", "kakan"})


def is_aka(pai: str) -> bool:
    return pai in AKA_TILES


def open_log(path: Path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


@dataclass
class KyokuRecord:
    has_aka: bool = False
    max_aka: int = 0
    did_call: bool = False
    did_win: bool = False
    win_aka: int = 0
    did_riichi: bool = False
    is_ryukyoku: bool = False
    did_houjuu: bool = False
    is_noten_end: bool = False
    aka_discard_turn_sum: int = 0
    aka_discard_count: int = 0


@dataclass
class Bucket:
    rounds: int = 0
    call: int = 0
    win: int = 0
    riichi: int = 0
    ryukyoku: int = 0
    houjuu: int = 0
    noten_end: int = 0
    win_aka_sum: int = 0
    aka_discard_turn_sum: int = 0
    aka_discard_count: int = 0


@dataclass
class ModelStats:
    total: Bucket = field(default_factory=Bucket)
    with_aka: Bucket = field(default_factory=Bucket)
    without_aka: Bucket = field(default_factory=Bucket)
    agari_aka_mismatch: int = 0


def count_aka_now(ps: PlayerState, fuuro_aka: int) -> int:
    return int(sum(ps.akas_in_hand)) + fuuro_aka


def aka_in_fuuro_event(ev: dict) -> int:
    tiles = [ev["pai"]] if "pai" in ev else []
    tiles.extend(ev.get("consumed") or [])
    return sum(is_aka(t) for t in tiles)


def finalize_kyoku(cur: KyokuRecord, states: list[PlayerState], mortal_id: int) -> None:
    if not cur.has_aka:
        return
    if cur.did_win:
        return
    if states[mortal_id].shanten != 0:
        cur.is_noten_end = True


def process_file(path: Path) -> tuple[list[KyokuRecord], int, int]:
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
    records: list[KyokuRecord] = []
    cur: KyokuRecord | None = None
    fuuro_aka = 0
    agari_mismatch = 0
    mortal_discard_turn = 0

    for ev in events:
        et = ev.get("type")

        if et == "start_kyoku":
            if cur is not None:
                finalize_kyoku(cur, states, mortal_id)
                records.append(cur)
            cur = KyokuRecord()
            fuuro_aka = 0
            mortal_discard_turn = 0

        if cur is not None and et == "hora":
            actor = ev["actor"]
            is_ron = actor != ev["target"]
            if actor != mortal_id and ev.get("target") == mortal_id:
                cur.did_houjuu = True
            ura = ev.get("ura_indicators") or []
            tracked = count_aka_now(states[actor], fuuro_aka)
            if is_ron and is_aka(ev.get("pai", "")):
                tracked += 1
            try:
                detail = states[actor].agari_detail(is_ron, ura)
            except RuntimeError:
                if actor == mortal_id:
                    cur.did_win = True
                    cur.win_aka = tracked
            else:
                if actor == mortal_id and detail.num_aka != tracked:
                    agari_mismatch += 1
                if actor == mortal_id:
                    cur.did_win = True
                    cur.win_aka = detail.num_aka

        if cur is not None:
            if et in CALL_TYPES and ev.get("actor") == mortal_id:
                cur.did_call = True
            if et == "reach" and ev.get("actor") == mortal_id:
                cur.did_riichi = True
            if et == "ryukyoku":
                cur.is_ryukyoku = True
            if et in FUURO_TYPES and ev.get("actor") == mortal_id:
                fuuro_aka += aka_in_fuuro_event(ev)
            if et == "dahai" and ev.get("actor") == mortal_id:
                mortal_discard_turn += 1
                if is_aka(ev.get("pai", "")):
                    cur.aka_discard_turn_sum += mortal_discard_turn
                    cur.aka_discard_count += 1

        for s in states:
            s.update(json.dumps(ev, separators=(",", ":")))

        if cur is not None:
            aka_now = count_aka_now(states[mortal_id], fuuro_aka)
            if aka_now > 0:
                cur.has_aka = True
            cur.max_aka = max(cur.max_aka, aka_now)

    if cur is not None:
        finalize_kyoku(cur, states, mortal_id)
        records.append(cur)

    return records, agari_mismatch, mortal_id


def add_record(bucket: Bucket, rec: KyokuRecord) -> None:
    bucket.rounds += 1
    bucket.call += int(rec.did_call)
    bucket.win += int(rec.did_win)
    bucket.riichi += int(rec.did_riichi)
    bucket.ryukyoku += int(rec.is_ryukyoku)
    bucket.houjuu += int(rec.did_houjuu)
    bucket.noten_end += int(rec.is_noten_end)
    bucket.win_aka_sum += rec.win_aka
    bucket.aka_discard_turn_sum += rec.aka_discard_turn_sum
    bucket.aka_discard_count += rec.aka_discard_count


def pct(num: int, den: int) -> str:
    if den == 0:
        return "—"
    return f"{100.0 * num / den:.2f}%"


def rate(num: int, den: int) -> float:
    if den == 0:
        return float("nan")
    return 100.0 * num / den


def delta_pp(wa: Bucket, wo: Bucket, attr: str) -> float:
    a = getattr(wa, attr)
    b = getattr(wo, attr)
    return rate(a, wa.rounds) - rate(b, wo.rounds)


def avg_aka_discard_turn(b: Bucket) -> float:
    if b.aka_discard_count == 0:
        return float("nan")
    return b.aka_discard_turn_sum / b.aka_discard_count


def analyze_dir(log_dir: Path) -> ModelStats:
    files = sorted(log_dir.glob("*.json.gz")) + sorted(log_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no logs in {log_dir}")

    stats = ModelStats()
    for path in files:
        recs, mism, _ = process_file(path)
        stats.agari_aka_mismatch += mism
        for rec in recs:
            add_record(stats.total, rec)
            if rec.has_aka:
                add_record(stats.with_aka, rec)
            else:
                add_record(stats.without_aka, rec)

    return stats


def row(label: str, b: Bucket) -> str:
    avg_turn = avg_aka_discard_turn(b)
    avg_turn_s = f"{avg_turn:.2f}" if avg_turn == avg_turn else "—"
    return (
        f"| {label} | {b.rounds} | {pct(b.call, b.rounds)} | {pct(b.win, b.rounds)} | "
        f"{pct(b.riichi, b.rounds)} | {pct(b.ryukyoku, b.rounds)} | "
        f"{pct(b.houjuu, b.rounds)} | {pct(b.noten_end, b.rounds)} | {avg_turn_s} |"
    )


def print_delta_table(name: str, s: ModelStats) -> None:
    wa, wo = s.with_aka, s.without_aka
    print(f"## {name}")
    print(
        f"  fuuro_delta={delta_pp(wa, wo, 'call'):+.2f}pp "
        f"riichi_delta={delta_pp(wa, wo, 'riichi'):+.2f}pp "
        f"houjuu_delta={delta_pp(wa, wo, 'houjuu'):+.2f}pp "
        f"ryukyoku_delta={delta_pp(wa, wo, 'ryukyoku'):+.2f}pp "
        f"noten_end_rate={rate(wa.noten_end, wa.rounds):.2f}% "
        f"avg_aka_discard_turn={avg_aka_discard_turn(wa):.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="*",
        help="name:log_dir pairs, e.g. lo00:/path/to/1v3",
    )
    args = ap.parse_args()

    if args.models:
        pairs: list[t[str, Path]] = []
        for item in args.models:
            name, _, log_dir = item.partition(":")
            pairs.append((name, Path(log_dir)))
    else:
        base = Path("/home/gamba/mahjong/runs/phase4/sweep_eval")
        models = ["beta0", "beta0_1", "beta0_3", "beta0_5", "beta1"]
        pairs = [(name, base / name / "1v3") for name in models]

    print("=== Phase 4 aka-conditional analysis ===")
    all_stats: dict[str, ModelStats] = {}
    for name, log_dir in pairs:
        stats = analyze_dir(log_dir)
        all_stats[name] = stats
        n = stats.total.rounds
        aka_n = stats.with_aka.rounds
        print(
            f"{name}: rounds={n}, aka_rounds={aka_n} ({pct(aka_n, n)}), "
            f"agari_aka_mismatch={stats.agari_aka_mismatch}"
        )
        print_delta_table(name, stats)

    print("\n--- TABLE DATA ---")
    print(
        "| 条件 | 局数 | 副露率 | 和了率 | 立直率 | 流局率 | 放銃率 | 赤保持ノーテン率 | 赤平均切り順 |"
    )
    for name, s in all_stats.items():
        print(f"## {name}")
        print(row("全体", s.total))
        print(row("赤あり", s.with_aka))
        print(row("赤なし", s.without_aka))
        wa, wo = s.with_aka, s.without_aka
        print(
            "| 差分(赤あり−赤なし) | — | "
            f"{delta_pp(wa, wo, 'call'):+.2f}pp | "
            f"{rate(wa.win, wa.rounds) - rate(wo.win, wo.rounds):+.2f}pp | "
            f"{delta_pp(wa, wo, 'riichi'):+.2f}pp | "
            f"{delta_pp(wa, wo, 'ryukyoku'):+.2f}pp | "
            f"{delta_pp(wa, wo, 'houjuu'):+.2f}pp | "
            f"{rate(wa.noten_end, wa.rounds) - rate(wo.noten_end, wo.rounds):+.2f}pp | "
            f"{avg_aka_discard_turn(wa):.2f} |"
        )
        if s.with_aka.win > 0:
            print(
                f"| 赤あり・和了時平均赤 | {s.with_aka.win_aka_sum / s.with_aka.win:.2f} | "
                f"(和了{s.with_aka.win}局) |"
            )


if __name__ == "__main__":
    main()
