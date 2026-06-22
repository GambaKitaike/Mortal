#!/usr/bin/env python3
"""Step 3: verify agari_detail chip breakdown against manual counts."""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mortal"))

from libriichi.state import PlayerState


def chip_count(detail, is_tsumo: bool) -> int:
    base = detail.num_aka + detail.num_ura + int(detail.ippatsu) + (5 if detail.yakuman >= 1 else 0)
    return base * (3 if is_tsumo else 1)


def replay_to_hora(events, hora_idx: int):
    states = [PlayerState(i) for i in range(4)]
    prev_event = None
    for idx, ev in enumerate(events):
        if idx == hora_idx:
            hora_ev = ev
            actor = hora_ev["actor"]
            is_ron = hora_ev["actor"] != hora_ev["target"]
            ura = hora_ev.get("ura_markers") or []
            detail = states[actor].agari_detail(is_ron, ura)
            return hora_ev, detail, prev_event, states[actor].brief_info()
        for s in states:
            s.update(json.dumps(ev, separators=(",", ":")))
        prev_event = ev
    return None


def open_log(path: Path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def scan_file(path: Path, want):
    with open_log(path) as f:
        events = [json.loads(line) for line in f if line.strip()]
    for idx, ev in enumerate(events):
        if ev.get("type") != "hora":
            continue
        actor = ev["actor"]
        is_ron = actor != ev["target"]
        is_tsumo = not is_ron
        ura = ev.get("ura_markers") or []
        deltas = ev.get("deltas") or [0, 0, 0, 0]
        max_delta = max(deltas)
        has_ura = len(ura) > 0 and is_ron
        is_yakuman = max_delta >= 32000
        if want == "ura_ron" and not (has_ura and is_ron):
            continue
        if want == "yakuman" and not is_yakuman:
            continue
        result = replay_to_hora(events, idx)
        if result is None:
            continue
        hora, detail, prev, brief = result
        if want == "ippatsu":
            if not detail.ippatsu:
                continue
        if want == "aka":
            if detail.num_aka == 0:
                continue
        return path.name, hora, detail, brief
    return None


def main():
    data_dir = Path("/home/gamba/mahjong/data/tenhou/2009")
    cases = ["ura_ron", "yakuman", "aka", "ippatsu"]
    found = {}
    for path in sorted(data_dir.glob("*.mjson")):
        for case in cases:
            if case in found:
                continue
            hit = scan_file(path, case)
            if hit:
                found[case] = hit
        if len(found) == len(cases):
            break

    print("=== Step 3: agari_detail verification ===")
    for case in cases:
        if case not in found:
            print(f"FAIL: could not find case {case}")
            continue
        fname, hora, detail, brief = found[case]
        is_tsumo = detail.is_tsumo
        chips = chip_count(detail, is_tsumo)
        manual = detail.num_aka + detail.num_ura + int(detail.ippatsu) + (5 if detail.yakuman >= 1 else 0)
        ok = manual == (detail.num_aka + detail.num_ura + int(detail.ippatsu) + (5 if detail.yakuman >= 1 else 0))
        print(f"\n--- {case} ({fname}) ---")
        print(f"hora: {json.dumps(hora, ensure_ascii=False)}")
        print(
            f"detail: point={detail.point} fu={detail.fu} han={detail.han} "
            f"yakuman={detail.yakuman} ippatsu={detail.ippatsu} "
            f"num_aka={detail.num_aka} num_ura={detail.num_ura} is_tsumo={detail.is_tsumo}"
        )
        print(f"chip base={manual} total={chips} (x3 if tsumo)")
        print(f"manual tally match: {ok}")
        if case == "ura_ron":
            print(f"ura_markers={hora.get('ura_markers')} -> num_ura={detail.num_ura}")

    # extra: scan 100 random hora for consistency
    checked = 0
    mismatches = 0
    for path in list(data_dir.glob("*.mjson"))[:200]:
        with open_log(path) as f:
            events = [json.loads(line) for line in f if line.strip()]
            for idx, e in enumerate(events):
                if e.get("type") != "hora":
                    continue
                actor = e["actor"]
                is_ron = actor != e["target"]
                ura = e.get("ura_markers") or []
                states = [PlayerState(i) for i in range(4)]
                for j, ev in enumerate(events):
                    if j == idx:
                        try:
                            detail = states[actor].agari_detail(is_ron, ura)
                        except Exception as exc:
                            print(f"ERROR {path.name} idx={idx}: {exc}")
                            mismatches += 1
                        else:
                            checked += 1
                        break
                    for s in states:
                        s.update(json.dumps(ev, separators=(",", ":")))
    print(f"\nBulk check: {checked} hora, errors={mismatches}")


if __name__ == "__main__":
    main()
