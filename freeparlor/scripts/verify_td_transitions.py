#!/usr/bin/env python3
"""Layer-2 gate: TD transitions (next_obs, next_mask, done_chip, r_chip) in dataloader."""

import argparse
import gzip
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mortal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloader import (
    assign_r_chip_to_trainee_final_moves,
    build_td_transitions,
    get_hora_chip_delta,
    load_kyoku_hora_r_chip,
    open_log,
)
from engine import MortalEngine
from libriichi.arena import OneVsThree
from libriichi.dataset import GameplayLoader
from model import Brain, DQN
from preprocess_chips import hora_chip_deltas


def inject_chip_delta_metadata(events):
    """Replay tenhou/offline logs: attach meta.chip_delta to hora (for offline smoke)."""
    from libriichi.state import PlayerState

    states = [PlayerState(i) for i in range(4)]
    out = []
    for ev in events:
        ev = dict(ev)
        if ev.get("type") == "hora":
            actor = ev["actor"]
            is_ron = actor != ev["target"]
            ura = ev.get("ura_markers") or []
            try:
                detail = states[actor].agari_detail(is_ron, ura)
                cd = hora_chip_deltas(ev, detail)
                ev["meta"] = {"chip_delta": cd}
            except Exception:
                pass
        for s in states:
            s.update(json.dumps(ev, separators=(",", ":")))
        out.append(ev)
    return out


def load_engine(state_file, device):
    state = torch.load(state_file, weights_only=True, map_location="cpu")
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    mortal = Brain(
        version=version,
        conv_channels=cfg["resnet"]["conv_channels"],
        num_blocks=cfg["resnet"]["num_blocks"],
    ).eval()
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
        enable_rule_based_agari_guard=False,
        name="champion",
        boltzmann_epsilon=0.5,
        boltzmann_temp=1.0,
    )


def collect_hora_by_kyoku(path):
    with open_log(path) as f:
        events = [json.loads(line) for line in f if line.strip()]
    kyoku_idx = -1
    per_kyoku = defaultdict(list)
    for i, ev in enumerate(events):
        if ev.get("type") == "start_kyoku":
            kyoku_idx += 1
        elif ev.get("type") == "hora" and kyoku_idx >= 0:
            cd = get_hora_chip_delta(ev)
            per_kyoku[kyoku_idx].append(
                {
                    "event_idx": i,
                    "actor": ev["actor"],
                    "target": ev["target"],
                    "chip_delta": cd,
                    "is_ron": ev["actor"] != ev["target"],
                }
            )
    return events, per_kyoku


def build_core_entries(obs, actions, masks, at_kyoku, dones, apply_gamma, kyoku_rewards, player_ranks):
    game_size = len(obs)
    steps_to_done = np.zeros(game_size, dtype=np.int64)
    for i in reversed(range(game_size)):
        if not dones[i]:
            steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])
    entries = []
    for i in range(game_size):
        entries.append(
            (
                obs[i],
                actions[i],
                masks[i],
                steps_to_done[i],
                kyoku_rewards[at_kyoku[i]],
                player_ranks[at_kyoku[i] + 1],
            )
        )
    return entries


def process_game(file_path, game, kyoku_rewards, player_ranks):
    obs = game.take_obs()
    actions = game.take_actions()
    masks = game.take_masks()
    at_kyoku = game.take_at_kyoku()
    dones = game.take_dones()
    apply_gamma = game.take_apply_gamma()
    player_id = game.take_player_id()
    game_size = len(obs)

    kyoku_hora_r_chip = load_kyoku_hora_r_chip(file_path, player_id)
    r_chip = assign_r_chip_to_trainee_final_moves(game_size, at_kyoku, kyoku_hora_r_chip)
    next_obs, next_masks, done_chip = build_td_transitions(obs, masks, at_kyoku, dones)
    core = build_core_entries(
        obs, actions, masks, at_kyoku, dones, apply_gamma, kyoku_rewards, player_ranks,
    )
    return {
        "player_id": player_id,
        "obs": obs,
        "actions": actions,
        "at_kyoku": at_kyoku,
        "dones": dones,
        "core": core,
        "r_chip": r_chip,
        "next_obs": next_obs,
        "next_masks": next_masks,
        "done_chip": done_chip,
        "kyoku_hora_r_chip": kyoku_hora_r_chip,
    }


def gate1_match_hora(data, hora_by_kyoku):
    mismatches = []
    at_kyoku = data["at_kyoku"]
    r_chip = data["r_chip"]
    pid = data["player_id"]
    last_idx = {}
    for i, k in enumerate(at_kyoku):
        last_idx[k] = i
    for kyoku, idx in last_idx.items():
        horas = hora_by_kyoku.get(kyoku, [])
        expected = sum(
            (h["chip_delta"] or [0, 0, 0, 0])[pid] for h in horas
        )
        got = float(r_chip[idx])
        if got != expected:
            mismatches.append((kyoku, idx, expected, got, horas))
    return mismatches


def gate2_ron_trace(path, data, hora_by_kyoku):
    pid = data["player_id"]
    for kyoku, horas in hora_by_kyoku.items():
        ron_h = [h for h in horas if h["is_ron"] and h["chip_delta"] and h["chip_delta"][pid] != 0]
        if not ron_h:
            continue
        last_idx = max(i for i, k in enumerate(data["at_kyoku"]) if k == kyoku)
        return {
            "file": Path(path).name,
            "kyoku": kyoku,
            "player_id": pid,
            "trainee_last_move_idx": last_idx,
            "trainee_last_action": int(data["actions"][last_idx]),
            "r_chip_at_last_move": float(data["r_chip"][last_idx]),
            "expected_sum": sum(h["chip_delta"][pid] for h in horas),
            "hora_events": ron_h,
            "nonzero_r_chip_indices": [
                (i, float(data["r_chip"][i]))
                for i in range(len(data["r_chip"]))
                if data["r_chip"][i] != 0
            ],
        }
    return None


def gate3_multi_ron_trace(path, data, hora_by_kyoku):
    pid = data["player_id"]
    best = None
    for kyoku, horas in hora_by_kyoku.items():
        if len(horas) < 2:
            continue
        last_idx = max(i for i, k in enumerate(data["at_kyoku"]) if k == kyoku)
        expected = sum(
            (h["chip_delta"] or [0, 0, 0, 0])[pid] for h in horas
        )
        trace = {
            "file": Path(path).name,
            "kyoku": kyoku,
            "player_id": pid,
            "hora_count": len(horas),
            "trainee_last_move_idx": last_idx,
            "trainee_last_action": int(data["actions"][last_idx]),
            "r_chip_at_last_move": float(data["r_chip"][last_idx]),
            "expected_sum": expected,
            "hora_events": horas,
            "nonzero_r_chip_indices": [
                (i, float(data["r_chip"][i]))
                for i in range(len(data["r_chip"]))
                if data["r_chip"][i] != 0
            ],
        }
        if best is None or abs(expected) > abs(best["expected_sum"]):
            best = trace
    return best


def gate4_done_and_next(data):
    errors = []
    for i, done in enumerate(data["done_chip"]):
        if done:
            if not np.all(data["next_obs"][i] == 0):
                errors.append((i, "next_obs not zero when done_chip=1"))
            if np.any(data["next_masks"][i]):
                errors.append((i, "next_mask not all-False when done_chip=1"))
        if i + 1 < len(data["at_kyoku"]):
            crosses = data["at_kyoku"][i + 1] != data["at_kyoku"][i]
            if crosses and done != 1:
                errors.append((i, "at_kyoku cross but done_chip!=1"))
    return errors


def gate5_core_regression(data):
    errors = []
    for i, (obs, act, mask, std, rew, rank) in enumerate(data["core"]):
        if not np.array_equal(obs, data["obs"][i]):
            errors.append((i, "obs mismatch"))
        if act != data["actions"][i]:
            errors.append((i, "action mismatch"))
    return errors


def verify_logs(log_paths, version, player_names=None):
    loader = GameplayLoader(version=version, oracle=False, player_names=player_names)
    results = {
        "gate1_mismatches": [],
        "gate2_trace": None,
        "gate3_trace": None,
        "gate4_errors": [],
        "gate5_errors": [],
        "games_checked": 0,
        "kyoku_with_hora_checked": 0,
    }

    for path in log_paths:
        _, hora_by_kyoku = collect_hora_by_kyoku(path)
        games = loader.load_gz_log_files([str(path)])[0]
        for game in games:
            grp = game.take_grp()
            player_id = game.take_player_id()
            grp_feature = grp.take_feature()
            final_scores = grp.take_final_scores()
            scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
            rank_by_player_seq = (-scores_seq).argsort(-1, kind="stable").argsort(-1, kind="stable")
            player_ranks = rank_by_player_seq[:, player_id]
            kyoku_rewards = np.zeros(len(grp_feature))

            data = process_game(path, game, kyoku_rewards, player_ranks)
            results["games_checked"] += 1
            results["gate1_mismatches"].extend(gate1_match_hora(data, hora_by_kyoku))
            results["gate4_errors"].extend(gate4_done_and_next(data))
            results["gate5_errors"].extend(gate5_core_regression(data))
            results["kyoku_with_hora_checked"] += sum(
                1 for k in set(data["at_kyoku"]) if hora_by_kyoku.get(k)
            )
            if results["gate2_trace"] is None:
                results["gate2_trace"] = gate2_ron_trace(path, data, hora_by_kyoku)
            g3 = gate3_multi_ron_trace(path, data, hora_by_kyoku)
            if g3 is not None:
                prev = results["gate3_trace"]
                if prev is None or abs(g3["expected_sum"]) > abs(prev["expected_sum"]):
                    results["gate3_trace"] = g3

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, help="Directory of arena .json.gz logs")
    parser.add_argument("--state-file", type=Path, help="Generate logs via OneVsThree")
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--offline-sample", type=Path, help="Optional tenhou mjson for offline smoke")
    parser.add_argument("--max-files", type=int, default=0, help="Limit log files (0=all)")
    args = parser.parse_args()

    log_paths = []
    if args.log_dir:
        log_paths = sorted(args.log_dir.glob("*.json.gz"))
        if args.max_files > 0:
            log_paths = log_paths[: args.max_files]
    elif args.state_file:
        device = torch.device(args.device)
        engine = load_engine(args.state_file, device)
        tmpdir = Path(tempfile.mkdtemp(prefix="verify_td_transitions_"))
        print(f"generating logs in {tmpdir}")
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            for split in "abcd":
                out = tmpdir / f"{seed}_0_{split}.json.gz"
                OneVsThree.py_vs_py(
                    challenger=engine,
                    champion=engine,
                    seed=seed * 4 + ord(split) - ord("a"),
                    log_path=str(out),
                )
        log_paths = sorted(tmpdir.glob("*.json.gz"))
    else:
        default = Path("/tmp/verify_arena_chip_delta_aibvnlx4")
        if default.exists():
            log_paths = sorted(default.glob("*.json.gz"))
        else:
            parser.error("provide --log-dir or --state-file")

    print(f"verifying {len(log_paths)} log files...")
    results = verify_logs(log_paths, args.version, player_names=None)

    print("\n=== verify_td_transitions ===")
    print(f"games checked: {results['games_checked']}")
    print(f"kyoku with hora: {results['kyoku_with_hora_checked']}")

    ok = True

    g1 = len(results["gate1_mismatches"])
    print(f"\nGate 1 (r_chip vs hora sum on trainee final move): {g1} mismatches")
    if g1:
        ok = False
        for m in results["gate1_mismatches"][:5]:
            print(" ", m)

    g2 = results["gate2_trace"]
    print("\nGate 2 (ron attribution trace):")
    if g2:
        print(json.dumps(g2, indent=2, ensure_ascii=False))
    else:
        print("  no ron trace found (WARN)")
        ok = False

    g3 = results["gate3_trace"]
    print("\nGate 3 (multi-ron trace):")
    if g3:
        print(json.dumps(g3, indent=2, ensure_ascii=False))
    else:
        print("  no multi-ron trace found (WARN)")
        ok = False

    g4 = len(results["gate4_errors"])
    print(f"\nGate 4 (done_chip / next zeroization): {g4} errors")
    if g4:
        ok = False
        print(" ", results["gate4_errors"][:5])

    g5 = len(results["gate5_errors"])
    print(f"\nGate 5 (core 6-field regression): {g5} errors")
    if g5:
        ok = False

    if args.offline_sample and args.offline_sample.exists():
        print(f"\nOffline smoke: {args.offline_sample.name}")
        with open_log(args.offline_sample) as f:
            events = [json.loads(line) for line in f if line.strip()]
        injected_path = Path(tempfile.mktemp(suffix=".json.gz"))
        with gzip.open(injected_path, "wt") as f:
            for ev in inject_chip_delta_metadata(events):
                f.write(json.dumps(ev, separators=(",", ":")) + "\n")
        off = verify_logs([injected_path], args.version, player_names=None)
        print(f"  games={off['games_checked']} gate1_mismatches={len(off['gate1_mismatches'])}")
        injected_path.unlink(missing_ok=True)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
