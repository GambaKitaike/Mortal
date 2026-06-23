#!/usr/bin/env python3
"""Phase 4c: aka-conditional playstyle analysis from HUMAN tenhou-2009 logs.

人間データ版。analyze_aka_conditional.py を流用し、入力を人間牌譜2009、
対象を全4プレイヤーに変更。集計単位は「局 × プレイヤー」(1局=4サンプル)。
定義(赤判定 5m/5p/5sr、副露 chi/pon/daiminkan、has_aka、立直、流局)は
AI分析(analyze_aka_conditional.py)と厳密に一致させる。
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict
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


@dataclass
class Bucket:
    rounds: int = 0
    call: int = 0
    win: int = 0
    riichi: int = 0
    ryukyoku: int = 0
    win_aka_sum: int = 0
    max_aka_sum: int = 0  # 赤あり局の平均赤枚数算出用


@dataclass
class ModelStats:
    total: Bucket = field(default_factory=Bucket)
    with_aka: Bucket = field(default_factory=Bucket)
    without_aka: Bucket = field(default_factory=Bucket)
    agari_aka_mismatch: int = 0
    files: int = 0


def count_aka_now(ps: PlayerState, fuuro_aka: int) -> int:
    return int(sum(ps.akas_in_hand)) + fuuro_aka


def aka_in_fuuro_event(ev: dict) -> int:
    tiles = [ev["pai"]] if "pai" in ev else []
    tiles.extend(ev.get("consumed") or [])
    return sum(is_aka(t) for t in tiles)


def process_file(path: Path) -> tuple[list[KyokuRecord], int]:
    """1ファイルを再生し、全4プレイヤー×全局の KyokuRecord を返す。"""
    with open_log(path) as f:
        events = [json.loads(line) for line in f if line.strip()]

    states = [PlayerState(i) for i in range(4)]
    records: list[KyokuRecord] = []
    # 各局: プレイヤーごとに 1 レコード (index = actor)
    cur: list[KyokuRecord] | None = None
    fuuro_aka = [0, 0, 0, 0]
    agari_mismatch = 0

    for ev in events:
        et = ev.get("type")

        if et == "start_kyoku":
            if cur is not None:
                records.extend(cur)
            cur = [KyokuRecord() for _ in range(4)]
            fuuro_aka = [0, 0, 0, 0]

        if cur is not None and et == "hora":
            actor = ev["actor"]
            is_ron = actor != ev["target"]
            ura = ev.get("ura_markers") or []
            tracked = count_aka_now(states[actor], fuuro_aka[actor])
            if is_ron and is_aka(ev.get("pai", "")):
                tracked += 1
            try:
                detail = states[actor].agari_detail(is_ron, ura)
            except RuntimeError:
                cur[actor].did_win = True
                cur[actor].win_aka = tracked
            else:
                if detail.num_aka != tracked:
                    agari_mismatch += 1
                cur[actor].did_win = True
                cur[actor].win_aka = detail.num_aka

        if cur is not None:
            if et in CALL_TYPES:
                actor = ev.get("actor")
                if actor is not None:
                    cur[actor].did_call = True
            if et == "reach":
                actor = ev.get("actor")
                if actor is not None:
                    cur[actor].did_riichi = True
            if et == "ryukyoku":
                for r in cur:
                    r.is_ryukyoku = True
            if et in FUURO_TYPES:
                actor = ev.get("actor")
                if actor is not None:
                    fuuro_aka[actor] += aka_in_fuuro_event(ev)

        for s in states:
            s.update(json.dumps(ev, separators=(",", ":")))

        if cur is not None:
            for p in range(4):
                aka_now = count_aka_now(states[p], fuuro_aka[p])
                if aka_now > 0:
                    cur[p].has_aka = True
                cur[p].max_aka = max(cur[p].max_aka, aka_now)

    if cur is not None:
        records.extend(cur)

    return records, agari_mismatch


def add_record(bucket: Bucket, rec: KyokuRecord) -> None:
    bucket.rounds += 1
    bucket.call += int(rec.did_call)
    bucket.win += int(rec.did_win)
    bucket.riichi += int(rec.did_riichi)
    bucket.ryukyoku += int(rec.is_ryukyoku)
    bucket.win_aka_sum += rec.win_aka
    bucket.max_aka_sum += rec.max_aka


def pct(num: int, den: int) -> str:
    if den == 0:
        return "—"
    return f"{100.0 * num / den:.2f}%"


def analyze_files(files: list[Path]) -> ModelStats:
    stats = ModelStats()
    for i, path in enumerate(files):
        try:
            recs, mism = process_file(path)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] skip {path.name}: {e}", file=sys.stderr)
            continue
        stats.files += 1
        stats.agari_aka_mismatch += mism
        for rec in recs:
            add_record(stats.total, rec)
            if rec.has_aka:
                add_record(stats.with_aka, rec)
            else:
                add_record(stats.without_aka, rec)
        if (i + 1) % 500 == 0:
            print(
                f"  ...{i + 1} files, samples={stats.total.rounds}",
                file=sys.stderr,
            )
    return stats


def row(label: str, b: Bucket) -> str:
    return (
        f"| {label} | {b.rounds} | {pct(b.call, b.rounds)} | {pct(b.win, b.rounds)} | "
        f"{pct(b.riichi, b.rounds)} | {pct(b.ryukyoku, b.rounds)} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir", default="/home/gamba/mahjong/data/tenhou/2009"
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="先頭Nファイルのみ (0=全件)"
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.mjson"))
    if not files:
        files = sorted(data_dir.glob("*.json.gz")) + sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no logs in {data_dir}")
    if args.limit > 0:
        files = files[: args.limit]

    print(f"=== Phase 4c human aka-conditional analysis ===")
    print(f"data_dir={data_dir}  files={len(files)}  (集計単位: 局×プレイヤー)")

    stats = analyze_files(files)

    n = stats.total.rounds
    aka_n = stats.with_aka.rounds
    print(
        f"\nprocessed_files={stats.files}  samples(局×人)={n}  "
        f"aka_samples={aka_n} ({pct(aka_n, n)})  "
        f"agari_aka_mismatch={stats.agari_aka_mismatch}"
    )

    print("\n--- TABLE DATA ---")
    print("| 条件 | 局×人 | 副露率 | 和了率 | 立直率 | 流局率 |")
    print("|---|---:|---:|---:|---:|---:|")
    print(row("全体", stats.total))
    print(row("赤あり", stats.with_aka))
    print(row("赤なし", stats.without_aka))

    def delta(a: Bucket, b: Bucket, f) -> str:
        if a.rounds == 0 or b.rounds == 0:
            return "—"
        return f"{100.0 * (f(a) / a.rounds - f(b) / b.rounds):+.2f}pp"

    wa, wo = stats.with_aka, stats.without_aka
    print(
        "| 差分(赤あり−赤なし) | — | "
        f"{delta(wa, wo, lambda b: b.call)} | "
        f"{delta(wa, wo, lambda b: b.win)} | "
        f"{delta(wa, wo, lambda b: b.riichi)} | "
        f"{delta(wa, wo, lambda b: b.ryukyoku)} |"
    )

    if wa.rounds > 0:
        print(f"\n赤あり局の平均赤枚数(max_aka): {wa.max_aka_sum / wa.rounds:.3f}")
    if wa.win > 0:
        print(
            f"赤あり・和了時平均赤枚数: {wa.win_aka_sum / wa.win:.3f} "
            f"(和了{wa.win}サンプル)"
        )


if __name__ == "__main__":
    main()
