#!/usr/bin/env python3
"""Step 4: bake chip counts per kyoku/player from mjson logs."""

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mortal"))
from libriichi.state import PlayerState


def open_log(path: Path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def chip_base(detail) -> int:
    return detail.num_aka + detail.num_ura + int(detail.ippatsu) + (5 if detail.yakuman >= 1 else 0)


def hora_chip_deltas(ev, detail) -> list[int]:
    actor = ev["actor"]
    target = ev["target"]
    base = chip_base(detail)
    deltas = [0, 0, 0, 0]
    if detail.is_tsumo:
        for i in range(4):
            if i == actor:
                deltas[i] = base * 3
            else:
                deltas[i] = -base
    else:
        deltas[actor] = base
        deltas[target] = -base
    return deltas


def process_file(path: Path) -> tuple[str, np.ndarray, int, dict, dict]:
    with open_log(path) as f:
        events = [json.loads(line) for line in f if line.strip()]

    states = [PlayerState(i) for i in range(4)]
    kyoku_idx = -1
    per_kyoku = defaultdict(lambda: [0, 0, 0, 0])
    hora_count = 0
    base_hist = defaultdict(int)
    stats = {"aka": 0, "ura": 0, "ippatsu": 0, "yakuman": 0, "ok": 0}

    for ev in events:
        if ev.get("type") == "start_kyoku":
            kyoku_idx += 1
        elif ev.get("type") == "hora" and kyoku_idx >= 0:
            actor = ev["actor"]
            is_ron = actor != ev["target"]
            ura = ev.get("ura_markers") or []
            try:
                detail = states[actor].agari_detail(is_ron, ura)
            except Exception:
                pass
            else:
                base = chip_base(detail)
                base_hist[base] += 1
                chip_d = hora_chip_deltas(ev, detail)
                for p in range(4):
                    per_kyoku[kyoku_idx][p] += chip_d[p]
                hora_count += 1
                stats["ok"] += 1
                stats["aka"] += int(detail.num_aka > 0)
                stats["ura"] += int(detail.num_ura > 0)
                stats["ippatsu"] += int(detail.ippatsu)
                stats["yakuman"] += int(detail.yakuman >= 1)

        for s in states:
            s.update(json.dumps(ev, separators=(",", ":")))

    if not per_kyoku:
        arr = np.zeros((0, 4), dtype=np.int16)
    else:
        n = max(per_kyoku) + 1
        arr = np.zeros((n, 4), dtype=np.int16)
        for k, v in per_kyoku.items():
            arr[k] = v

    return path.name, arr, hora_count, dict(base_hist), stats


def main():
    data_dir = Path("/home/gamba/mahjong/data/tenhou/2009")
    out_dir = Path("/home/gamba/mahjong/data/tenhou/chips")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.mjson"))
    total_hora = 0
    global_base = defaultdict(int)
    aka_hora = 0
    ura_hora = 0
    ippatsu_hora = 0
    yakuman_hora = 0
    detail_ok = 0

    for i, path in enumerate(files):
        name, arr, hora_count, base_hist, stats = process_file(path)
        np.savez_compressed(out_dir / f"{name}.npz", chips=arr)
        total_hora += hora_count
        detail_ok += stats["ok"]
        aka_hora += stats["aka"]
        ura_hora += stats["ura"]
        ippatsu_hora += stats["ippatsu"]
        yakuman_hora += stats["yakuman"]
        for k, v in base_hist.items():
            global_base[k] += v
        if (i + 1) % 500 == 0:
            print(f"processed {i+1}/{len(files)} files, hora so far={total_hora}")

    print("=== Step 4: chip preprocessing ===")
    print(f"files: {len(files)}")
    print(f"total hora processed: {total_hora}")
    print(f"output dir: {out_dir}")
    print(f"chip base distribution (top): {sorted(global_base.items(), reverse=True)[:10]}")
    if detail_ok:
        print(
            f"rates: aka={aka_hora/detail_ok:.3f} ura={ura_hora/detail_ok:.3f} "
            f"ippatsu={ippatsu_hora/detail_ok:.3f} yakuman={yakuman_hora/detail_ok:.3f}"
        )


if __name__ == "__main__":
    main()
