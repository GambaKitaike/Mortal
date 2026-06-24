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


def count_hand_aka(state: PlayerState) -> int:
    return int(sum(state.akas_in_hand))


def is_tenpai_at_end(state: PlayerState) -> bool:
    return state.shanten == 0


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


def snapshot_kyoku_end(states, kyoku_idx, aka_held, tenpai_end, won, dealt_in, ev=None):
    for p in range(4):
        aka_held[kyoku_idx][p] = count_hand_aka(states[p])
        tenpai_end[kyoku_idx][p] = int(is_tenpai_at_end(states[p]))
    if ev is not None and ev.get("type") == "hora":
        actor = ev["actor"]
        target = ev["target"]
        is_ron = actor != target
        won[kyoku_idx][actor] = 1
        if is_ron:
            dealt_in[kyoku_idx][target] = 1


def process_file(path: Path) -> tuple[str, np.ndarray, dict, int, dict, dict, dict]:
    with open_log(path) as f:
        events = [json.loads(line) for line in f if line.strip()]

    states = [PlayerState(i) for i in range(4)]
    kyoku_idx = -1
    per_kyoku = defaultdict(lambda: [0, 0, 0, 0])
    aka_held = defaultdict(lambda: [0, 0, 0, 0])
    tenpai_end = defaultdict(lambda: [0, 0, 0, 0])
    won = defaultdict(lambda: [0, 0, 0, 0])
    dealt_in = defaultdict(lambda: [0, 0, 0, 0])
    hora_count = 0
    base_hist = defaultdict(int)
    stats = {"aka": 0, "ura": 0, "ippatsu": 0, "yakuman": 0, "ok": 0}
    sanity = {"aka_held_pos": 0, "fire": 0, "kyoku_player": 0}

    for ev in events:
        if ev.get("type") == "start_kyoku":
            kyoku_idx += 1
        elif kyoku_idx >= 0 and ev.get("type") in ("hora", "ryukyoku"):
            snapshot_kyoku_end(
                states, kyoku_idx, aka_held, tenpai_end, won, dealt_in,
                ev if ev.get("type") == "hora" else None,
            )
            for p in range(4):
                sanity["kyoku_player"] += 1
                if aka_held[kyoku_idx][p] > 0:
                    sanity["aka_held_pos"] += 1
                if (
                    won[kyoku_idx][p] == 0
                    and dealt_in[kyoku_idx][p] == 0
                    and aka_held[kyoku_idx][p] > 0
                    and tenpai_end[kyoku_idx][p] == 1
                ):
                    sanity["fire"] += 1

        if ev.get("type") == "hora" and kyoku_idx >= 0:
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

    kyoku_keys = set(per_kyoku) | set(aka_held) | set(tenpai_end) | set(won) | set(dealt_in)
    if not kyoku_keys:
        arr = np.zeros((0, 4), dtype=np.int16)
        aka_arr = np.zeros((0, 4), dtype=np.int16)
        tenpai_arr = np.zeros((0, 4), dtype=np.int8)
        won_arr = np.zeros((0, 4), dtype=np.int8)
        dealt_in_arr = np.zeros((0, 4), dtype=np.int8)
    else:
        n = max(kyoku_keys) + 1
        arr = np.zeros((n, 4), dtype=np.int16)
        aka_arr = np.zeros((n, 4), dtype=np.int16)
        tenpai_arr = np.zeros((n, 4), dtype=np.int8)
        won_arr = np.zeros((n, 4), dtype=np.int8)
        dealt_in_arr = np.zeros((n, 4), dtype=np.int8)
        for k, v in per_kyoku.items():
            arr[k] = v
        for k, v in aka_held.items():
            aka_arr[k] = v
        for k, v in tenpai_end.items():
            tenpai_arr[k] = v
        for k, v in won.items():
            won_arr[k] = v
        for k, v in dealt_in.items():
            dealt_in_arr[k] = v

    arrays = {
        "aka_held": aka_arr,
        "tenpai_end": tenpai_arr,
        "won": won_arr,
        "dealt_in": dealt_in_arr,
    }
    return path.name, arr, arrays, hora_count, dict(base_hist), stats, sanity


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
    total_aka_held_pos = 0
    total_fire = 0
    total_kyoku_player = 0

    for i, path in enumerate(files):
        name, arr, arrays, hora_count, base_hist, stats, sanity = process_file(path)
        np.savez_compressed(
            out_dir / f"{name}.npz",
            chips=arr,
            **arrays,
        )
        total_hora += hora_count
        detail_ok += stats["ok"]
        aka_hora += stats["aka"]
        ura_hora += stats["ura"]
        ippatsu_hora += stats["ippatsu"]
        yakuman_hora += stats["yakuman"]
        total_aka_held_pos += sanity["aka_held_pos"]
        total_fire += sanity["fire"]
        total_kyoku_player += sanity["kyoku_player"]
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
    if total_kyoku_player:
        print(
            f"sanity: aka_held>0 rate={total_aka_held_pos/total_kyoku_player:.4f} "
            f"fire rate={total_fire/total_kyoku_player:.4f} "
            f"(fire = tenpai + aka held + not won + not dealt_in)"
        )


if __name__ == "__main__":
    main()
