#!/usr/bin/env python3
"""Cross-check arena meta.chip_delta vs preprocess_chips.process_file()."""

import argparse
import gzip
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mortal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import MortalEngine
from libriichi.arena import OneVsThree
from libriichi.state import PlayerState
from model import Brain, DQN
from preprocess_chips import chip_base, hora_chip_deltas, process_file


def get_chip_delta(ev: dict) -> list[int] | None:
    meta = ev.get("meta") or {}
    if "chip_delta" in meta:
        return meta["chip_delta"]
    return ev.get("chip_delta")


def expected_chip_delta_at_hora(events: list[dict], hora_idx: int) -> list[int]:
    """Replay to hora_idx skipping prior hora updates (arena does not broadcast hora)."""
    ev = events[hora_idx]
    states = [PlayerState(i) for i in range(4)]
    for e in events[:hora_idx]:
        if e.get("type") == "hora":
            continue
        for s in states:
            s.update(json.dumps(e, separators=(",", ":")))
    actor = ev["actor"]
    is_ron = actor != ev["target"]
    ura = ev.get("ura_markers") or []
    detail = states[actor].agari_detail(is_ron, ura)
    return hora_chip_deltas(ev, detail)


def preprocess_per_kyoku_arena_semantics(path: Path) -> dict[int, list[int]]:
    """Like process_file() but skip hora broadcast (matches arena state at each hora)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    per_kyoku: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    kyoku_idx = -1
    for i, ev in enumerate(events):
        if ev.get("type") == "start_kyoku":
            kyoku_idx += 1
        elif ev.get("type") == "hora" and kyoku_idx >= 0:
            try:
                chip_d = expected_chip_delta_at_hora(events, i)
            except Exception:
                continue
            for p in range(4):
                per_kyoku[kyoku_idx][p] += chip_d[p]
    return per_kyoku


def load_engine(
    state_file: Path,
    device: torch.device,
    *,
    name: str,
    boltzmann_epsilon: float,
    boltzmann_temp: float,
    enable_rule_based_agari_guard: bool,
) -> MortalEngine:
    state = torch.load(state_file, weights_only=True, map_location="cpu")
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    conv_channels = cfg["resnet"]["conv_channels"]
    num_blocks = cfg["resnet"]["num_blocks"]
    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    return MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        device=device,
        enable_amp=device.type == "cuda",
        enable_quick_eval=True,
        enable_rule_based_agari_guard=enable_rule_based_agari_guard,
        name=name,
        boltzmann_epsilon=boltzmann_epsilon,
        boltzmann_temp=boltzmann_temp,
    )


class AgariAcceptEngine:
    """Always accept agari; otherwise tsumogiri discard (mjai-log)."""

    engine_type = "mjai-log"
    name = "agari_accept"

    def __init__(self):
        self.player_ids = None

    def set_player_ids(self, player_ids):
        self.player_ids = list(player_ids)

    def react_batch(self, game_states):
        out = []
        for gs in game_states:
            s = gs.state
            pid = self.player_ids[gs.game_index]
            cans = s.last_cans
            if cans.can_tsumo_agari:
                out.append(
                    json.dumps(
                        {"type": "hora", "actor": pid, "target": pid},
                        separators=(",", ":"),
                    )
                )
            elif cans.can_ron_agari:
                out.append(
                    json.dumps(
                        {
                            "type": "hora",
                            "actor": pid,
                            "target": cans.target_actor,
                        },
                        separators=(",", ":"),
                    )
                )
            elif cans.can_discard:
                out.append(
                    json.dumps(
                        {
                            "type": "dahai",
                            "actor": pid,
                            "pai": s.last_self_tsumo(),
                            "tsumogiri": True,
                        },
                        separators=(",", ":"),
                    )
                )
            else:
                out.append('{"type":"none"}')
        return out

    def start_game(self, game_idx: int):
        pass

    def end_kyoku(self, game_idx: int):
        pass

    def end_game(self, game_idx: int, scores):
        pass


def verify_log(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    kyoku_idx = -1
    arena_per_kyoku = defaultdict(lambda: [0, 0, 0, 0])
    hora_rows = []
    event_mismatch = 0
    meta_missing = 0
    hora_indices = []

    for i, ev in enumerate(events):
        if ev.get("type") == "start_kyoku":
            kyoku_idx += 1
        elif ev.get("type") == "hora" and kyoku_idx >= 0:
            hora_indices.append(i)
            cd = get_chip_delta(ev)
            if cd is None:
                meta_missing += 1
            else:
                try:
                    expected = expected_chip_delta_at_hora(events, i)
                except Exception as exc:
                    event_mismatch += 1
                    if event_mismatch <= 3:
                        print(f"EVENT FAIL {path.name} idx={i}: {exc}")
                else:
                    if cd != expected:
                        event_mismatch += 1
                for p in range(4):
                    arena_per_kyoku[kyoku_idx][p] += cd[p] if cd else 0
                if cd and max(abs(x) for x in cd) > 0:
                    hora_rows.append(
                        {
                            "file": path.name,
                            "kyoku": kyoku_idx,
                            "actor": ev["actor"],
                            "chip_delta": cd,
                        }
                    )

    # preprocess process_file() — sequential replay (hora updates applied)
    _, prep_arr, _, _, _, _, _ = process_file(path)
    prep_legacy = {
        k: prep_arr[k].tolist() for k in range(len(prep_arr))
    }

    # preprocess-equivalent with arena hora semantics
    prep_arena = preprocess_per_kyoku_arena_semantics(path)

    kyoku_mismatch_legacy = 0
    kyoku_mismatch_arena = 0
    mismatch_examples = []
    multi_ron_kyoku = defaultdict(int)
    for idx in hora_indices:
        # count hora per kyoku
        pass
    kyoku_hora_count = defaultdict(int)
    k = -1
    for ev in events:
        if ev.get("type") == "start_kyoku":
            k += 1
        elif ev.get("type") == "hora":
            kyoku_hora_count[k] += 1

    all_kyoku = set(arena_per_kyoku) | set(prep_legacy) | set(prep_arena)
    for k in sorted(all_kyoku):
        a = arena_per_kyoku.get(k, [0, 0, 0, 0])
        p_legacy = prep_legacy.get(k, [0, 0, 0, 0])
        p_arena = prep_arena.get(k, [0, 0, 0, 0])
        if a != p_legacy:
            kyoku_mismatch_legacy += 1
        if a != p_arena:
            kyoku_mismatch_arena += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append((path.name, k, a, p_arena))
        if kyoku_hora_count.get(k, 0) > 1 and a != p_legacy:
            multi_ron_kyoku[k] += 1

    return {
        "path": path,
        "hora_nonzero": len(hora_rows),
        "hora_rows": hora_rows,
        "meta_missing": meta_missing,
        "kyoku_mismatch_legacy": kyoku_mismatch_legacy,
        "kyoku_mismatch_arena": kyoku_mismatch_arena,
        "event_mismatch": event_mismatch,
        "mismatch_examples": mismatch_examples,
        "multi_ron_kyoku": len(multi_ron_kyoku),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("/home/gamba/mahjong/runs/mortal.pth"),
    )
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--min-nonzero-hora", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("mortal", "mortal_vs_agari", "agari"),
        default="mortal",
        help="mortal: model self-play; mortal_vs_agari: model vs agari bot; agari: 4x agari bot",
    )
    parser.add_argument("--boltzmann-epsilon", type=float, default=0.3)
    parser.add_argument("--boltzmann-temp", type=float, default=0.5)
    parser.add_argument("--agari-guard", action="store_true")
    parser.add_argument(
        "--verify-only",
        type=Path,
        default=None,
        help="Skip generation; verify existing log directory",
    )
    args = parser.parse_args()

    if args.verify_only:
        tmpdir = args.verify_only
        logs = sorted(tmpdir.glob("*.json.gz"))
        print(f"verify-only log_dir={tmpdir} ({len(logs)} files)")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"device={device} mode={args.mode} seeds=[{args.seed_start}, {args.seed_start + args.seed_count})")

        tmpdir = Path(tempfile.mkdtemp(prefix="verify_arena_chip_delta_"))
        print(f"log_dir={tmpdir}")

        if args.mode == "agari":
            challenger = AgariAcceptEngine()
            champion = AgariAcceptEngine()
        else:
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            challenger = load_engine(
                args.state_file,
                device,
                name="challenger",
                boltzmann_epsilon=args.boltzmann_epsilon,
                boltzmann_temp=args.boltzmann_temp,
                enable_rule_based_agari_guard=args.agari_guard,
            )
            if args.mode == "mortal_vs_agari":
                champion = AgariAcceptEngine()
            else:
                champion = load_engine(
                    args.state_file,
                    device,
                    name="champion",
                    boltzmann_epsilon=args.boltzmann_epsilon,
                    boltzmann_temp=args.boltzmann_temp,
                    enable_rule_based_agari_guard=args.agari_guard,
                )

        env = OneVsThree(disable_progress_bar=False, log_dir=str(tmpdir))
        env.py_vs_py(
            challenger,
            champion,
            (args.seed_start, 0),
            args.seed_count,
        )

        logs = sorted(tmpdir.glob("*.json.gz"))
        print(f"generated {len(logs)} log files")

    total_nonzero = 0
    total_hora = 0
    meta_missing = 0
    kyoku_mismatch_legacy = 0
    kyoku_mismatch_arena = 0
    event_mismatch = 0
    multi_ron_files = 0
    sample_rows = []

    for path in logs:
        r = verify_log(path)
        total_nonzero += r["hora_nonzero"]
        meta_missing += r["meta_missing"]
        kyoku_mismatch_legacy += r["kyoku_mismatch_legacy"]
        kyoku_mismatch_arena += r["kyoku_mismatch_arena"]
        event_mismatch += r["event_mismatch"]
        if r["multi_ron_kyoku"]:
            multi_ron_files += 1
        for row in r["hora_rows"]:
            sample_rows.append(row)
        if r["mismatch_examples"]:
            print("KYOKU MISMATCH (arena semantics):", r["mismatch_examples"])

    # count all hora
    for path in logs:
        with gzip.open(path, "rt") as f:
            for line in f:
                if json.loads(line).get("type") == "hora":
                    total_hora += 1

    print("\n=== verify_arena_chip_delta ===")
    print(f"total hora events: {total_hora}")
    print(f"non-zero chip_delta hora: {total_nonzero}")
    print(f"meta missing: {meta_missing}")
    print(f"per-event mismatches (arena-semantics replay): {event_mismatch}")
    print(f"per-kyoku mismatches vs process_file(): {kyoku_mismatch_legacy}")
    print(f"  (multi-ron affected files: {multi_ron_files})")
    print(f"per-kyoku mismatches vs preprocess+skip-hora replay: {kyoku_mismatch_arena}")
    print("\nnon-zero hora samples (first 20):")
    for row in sample_rows[:20]:
        print(f"  {row}")

    ok = (
        total_nonzero >= args.min_nonzero_hora
        and meta_missing == 0
        and event_mismatch == 0
        and kyoku_mismatch_arena == 0
    )
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    if total_nonzero < args.min_nonzero_hora:
        print(f"  need >={args.min_nonzero_hora} non-zero hora, got {total_nonzero}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
