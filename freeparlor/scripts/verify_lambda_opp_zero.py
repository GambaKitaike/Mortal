#!/usr/bin/env python3
"""Verify calc_delta_blend with lambda_opp=0 matches the pre-opp formula."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mortal"))
from config import config
from libriichi.dataset import GameplayLoader
from reward_calculator import RewardCalculator
from model import GRP
import torch

def old_calc(reward_calc, player_id, grp_feature, rank_by_player, final_scores,
             chip_deltas, beta, chip_value):
    sotensu = reward_calc.calc_delta_points(player_id, grp_feature, final_scores) / 1000.0
    juni = reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
    reward = reward_calc.alpha * sotensu + reward_calc.gamma_pt * juni
    if chip_deltas is not None:
        reward = reward + beta * chip_deltas * chip_value
    return reward


import glob

def main():
    chip_dir = Path(config['env'].get('chip_dir', '/home/gamba/mahjong/data/tenhou/chips'))
    globs = config['dataset']['globs']
    loader = GameplayLoader(version=config['control']['version'])
    files = sorted(glob.glob(globs[0]) or glob.glob(globs[0].replace('/**/*.mjson', '/*.mjson')))[:1]
    if not files:
        print("no files found")
        return 1

    grp = GRP(**config['grp']['network'])
    grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location='cpu')
    grp.load_state_dict(grp_state['model'])
    reward_calc = RewardCalculator(
        grp, config['env']['pts'],
        alpha=config['env'].get('alpha', 1.0),
        gamma_pt=config['env'].get('gamma_pt', 1.0),
    )
    beta = config['env'].get('beta', 0.0)
    chip_value = config['env'].get('chip_value', 5.0)

    data = loader.load_gz_log_files(files)
    file_path = files[0]
    for game in data[0]:
        player_id = game.take_player_id()
        grp_obj = game.take_grp()
        grp_feature = grp_obj.take_feature()
        rank_by_player = grp_obj.take_rank_by_player()
        final_scores = grp_obj.take_final_scores()
        n = len(grp_feature)
        chip_path = chip_dir / f"{Path(file_path).name}.npz"
        if chip_path.exists():
            chips = np.load(chip_path)
            raw = chips['chips']
            if raw.shape[0] < n:
                padded = np.zeros((n, 4), dtype=np.float64)
                padded[:raw.shape[0]] = raw
                raw = padded
            chip_deltas = raw[:n, player_id].astype(np.float64)
            probe = {}
            for k in ('aka_held', 'tenpai_end', 'won', 'dealt_in'):
                if k in chips:
                    arr = chips[k]
                    if arr.shape[0] < n:
                        pad = np.zeros((n, 4), dtype=arr.dtype)
                        pad[:arr.shape[0]] = arr
                        arr = pad
                    probe[k] = arr[:n, player_id]
                else:
                    probe[k] = np.zeros(n)
        else:
            chip_deltas = np.zeros(n)
            probe = {k: np.zeros(n) for k in ('aka_held', 'tenpai_end', 'won', 'dealt_in')}

        ref = old_calc(
            reward_calc, player_id, grp_feature, rank_by_player, final_scores,
            chip_deltas, beta, chip_value,
        )
        new = reward_calc.calc_delta_blend(
            player_id, grp_feature, rank_by_player, final_scores,
            alpha=reward_calc.alpha, gamma_pt=reward_calc.gamma_pt,
            chip_deltas=chip_deltas, beta=beta, chip_value=chip_value,
            aka_held=probe['aka_held'], tenpai_end=probe['tenpai_end'],
            won=probe['won'], dealt_in=probe['dealt_in'],
            lambda_opp=0.0, noten_factor=0.0,
        )
        if not np.allclose(ref, new):
            print("MISMATCH", file_path, np.max(np.abs(ref - new)))
            return 1
        print(f"OK: {Path(file_path).name} player={player_id} n_kyoku={n} max_diff=0")
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
