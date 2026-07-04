#!/usr/bin/env python3
"""Measure GPU VRAM for N resident Brain+ActorCritic pairs (pool cache sizing)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mortal'))

from libriichi.consts import ACTION_SPACE, obs_shape
from model import ActorCritic, Brain, load_ppo_from_mortal_checkpoint


def gpu_mem_mib() -> tuple[int, int]:
    out = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used,memory.total',
        '--format=csv,noheader,nounits',
    ], text=True).strip()
    used, total = out.split(',')
    return int(used.strip()), int(total.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth')
    parser.add_argument('--pairs', type=int, default=7)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    version = state['config']['control']['version']
    resnet = state['config']['resnet']
    c, w = obs_shape(version)

    baseline_used, total = gpu_mem_mib()
    print(f'baseline GPU mem: {baseline_used} / {total} MiB')

    models = []
    for i in range(args.pairs):
        brain = Brain(version=version, **resnet).eval().to(device)
        ac = ActorCritic(version=version, tau=1.0).eval().to(device)
        load_ppo_from_mortal_checkpoint(ac, args.checkpoint, map_location=device)
        brain.load_state_dict(state['mortal'])
        x = torch.zeros(1, c, w, device=device)
        mask = torch.ones(1, ACTION_SPACE, dtype=torch.bool, device=device)
        with torch.inference_mode():
            phi = brain(x)
            ac(phi, mask)
        models.append((brain, ac))
        used, _ = gpu_mem_mib()
        print(f'after {i + 1} pairs: {used} MiB (+{used - baseline_used})')

    used_n, _ = gpu_mem_mib()
    print(
        f'RESULT resident_{args.pairs}_pairs: {used_n} MiB used, '
        f'delta={used_n - baseline_used} MiB over baseline, total={total} MiB'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
