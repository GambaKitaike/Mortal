#!/usr/bin/env python3
"""Layer-3 gate: Q_chip head, target Polyak, n-step TD, beta_sel warmup."""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mortal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloader import collate_moves, compute_nstep_chip, unpack_entry
from model import Brain, ChipDQNTarget, DQN, q_total


def gate2_polyak(version=4):
    dqn = DQN(version=version)
    tgt = ChipDQNTarget(dqn)
    tgt.copy_from(dqn)
    before = copy.deepcopy(tgt.state_dict_for_save())
    dqn.chip_net.weight.data.add_(1.0)
    tgt.polyak_update(dqn, tau=0.005)
    after = tgt.state_dict_for_save()
    key = "chip_net"
    w0 = before[key]["weight"]
    w1 = after[key]["weight"]
    expected = 0.005 * dqn.chip_net.weight.detach().cpu() + 0.995 * w0
    ok = torch.allclose(w1, expected, atol=1e-5)
    return ok, float((w1 - w0).abs().max())


def gate3_nstep_trace():
    # synthetic game: 5 moves same kyoku, r only on last
    game = []
    for i in range(5):
        obs = np.zeros((8, 34), dtype=np.float32)
        mask = np.ones(46, dtype=bool)
        done = 1 if i == 4 else 0
        r = 2.0 if i == 4 else 0.0
        entry = [obs, i, mask, 0, 0.0, 0, obs, mask, done, r]
        game.append(entry)

    traces = []
    for i in range(5):
        R, boot_obs, boot_mask, boot_done = compute_nstep_chip(game, i, n_step=3, gamma=1.0)
        traces.append((i, R, boot_done))

    # i=0: three non-terminal steps -> bootstrap; i=2 hits terminal move 4 with r=2
    ok = (
        traces[0][1] == 0.0 and traces[0][2] is False
        and traces[2][1] == 2.0 and traces[2][2] is True
        and traces[4][1] == 2.0 and traces[4][2] is True
    )
    return ok, traces


def gate1_regression_smoke(cfg_path, steps=20):
    """Run train loop with chip_weight=0; verify dqn path only."""
    import os

    os.environ["MORTAL_CFG"] = str(cfg_path)
    # reload config
    import importlib
    import config as cfg_mod

    importlib.reload(cfg_mod)
    cfg_mod.config["env"]["chip_weight"] = 0.0
    cfg_mod.config["env"]["beta_sel_max"] = 0.0
    cfg_mod.config["control"]["save_every"] = steps
    cfg_mod.config["control"]["test_every"] = 10**9
    cfg_mod.config["dataset"]["num_epochs"] = 5
    cfg_mod.config["dataset"]["games_per_batch"] = 2
    cfg_mod.config["control"]["batch_size"] = 32

    from train import train

    try:
        train()
        return True, "completed"
    except Exception as ex:
        return False, str(ex)


def gate4_chip_smoke(cfg_path, steps=100):
    import os
    from pathlib import Path

    os.environ["MORTAL_CFG"] = str(cfg_path)
    import importlib
    import config as cfg_mod

    importlib.reload(cfg_mod)
    cfg_mod.config["env"]["chip_weight"] = 1.0
    cfg_mod.config["control"]["save_every"] = 25
    cfg_mod.config["control"]["test_every"] = 10**9
    cfg_mod.config["dataset"]["num_epochs"] = 20
    cfg_mod.config["dataset"]["games_per_batch"] = 2
    cfg_mod.config["control"]["batch_size"] = 32
    state_path = Path(cfg_mod.config["control"]["state_file"])
    state_path.parent.mkdir(parents=True, exist_ok=True)

    from train import train

    try:
        train()
        if not state_path.exists():
            return False, f"no state at {state_path}"
        state = torch.load(state_path, weights_only=True)
        if state["steps"] < 25:
            return False, f"only {state['steps']} steps"
        if "chip_target" not in state:
            return False, "chip_target missing from state"
        return True, f"steps={state['steps']}"
    except Exception as ex:
        return False, str(ex)


def gate5_warmstart(state_file, version=4):
    state = torch.load(state_file, weights_only=True, map_location="cpu")
    dqn = DQN(version=version)
    missing = dqn.load_state_dict(state["current_dqn"], strict=False)
    has_chip_missing = any("chip" in k for k in missing.missing_keys)
    tgt = ChipDQNTarget(dqn)
    tgt.copy_from(dqn)
    phi = torch.randn(2, 1024)
    mask = torch.ones(2, 46, dtype=torch.bool)
    q = dqn.forward_chip(phi, mask)
    q_tgt = tgt(phi, mask)
    ok = has_chip_missing and torch.isfinite(q).all() and torch.isfinite(q_tgt).all()
    return ok, missing.missing_keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-file", type=Path, help="4d lo=0.3 checkpoint for gate 5")
    p.add_argument("--config", type=Path, help="config for gate 1/4 smoke")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    results = {}

    ok, delta = gate2_polyak()
    results["gate2_polyak"] = ok
    print(f"Gate 2 Polyak: {'PASS' if ok else 'FAIL'} (max delta={delta:.6f})")

    ok, traces = gate3_nstep_trace()
    results["gate3_nstep"] = ok
    print(f"Gate 3 n-step truncate: {'PASS' if ok else 'FAIL'}")
    for t in traces:
        print(f"  i={t[0]} R={t[1]} boot_done={t[2]}")

    # forward smoke
    dqn = DQN(version=4)
    phi = torch.randn(4, 1024)
    mask = torch.ones(4, 46, dtype=torch.bool)
    q_main, q_chip = dqn(phi, mask, return_q_chip=True)
    q_tot = q_total(q_main, q_chip, 0.3)
    fwd_ok = q_main.shape == q_chip.shape == q_tot.shape
    results["forward"] = fwd_ok
    print(f"Forward Q_main/Q_chip/Q_total: {'PASS' if fwd_ok else 'FAIL'}")

    if args.state_file and args.state_file.exists():
        ok, missing = gate5_warmstart(args.state_file)
        results["gate5_warmstart"] = ok
        print(f"Gate 5 warm-start: {'PASS' if ok else 'FAIL'} missing={missing}")
    else:
        print("Gate 5 warm-start: SKIP (no --state-file)")

    if not args.skip_train and args.config and args.config.exists():
        ok, msg = gate1_regression_smoke(args.config)
        results["gate1_regression"] = ok
        print(f"Gate 1 regression (chip_weight=0): {'PASS' if ok else 'FAIL'} {msg}")

        ok, msg = gate4_chip_smoke(args.config)
        results["gate4_chip_smoke"] = ok
        print(f"Gate 4 chip smoke: {'PASS' if ok else 'FAIL'} {msg}")
    else:
        print("Gate 1/4 train smoke: SKIP")

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("All executed gates PASS")


if __name__ == "__main__":
    main()
