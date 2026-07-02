"""Build PPO trajectories from mjai logs and drain unpacked .traj payloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model import Brain, ActorCritic
from ppo import action_log_probs
from ppo_transport import TrajectoryBatch, numpy_trajectory_to_batch, unpack_trajectory


def assign_rewards_and_dones(at_kyoku, kyoku_rewards, game_size):
    """Assign kyoku-end reward on the terminal step only (§2.2: episode = kyoku)."""
    rewards = np.zeros(game_size, dtype=np.float32)
    dones = np.zeros(game_size, dtype=bool)
    last_by_kyoku = {}
    for i in range(game_size):
        last_by_kyoku[at_kyoku[i]] = i
    for kyoku, idx in last_by_kyoku.items():
        rewards[idx] = float(kyoku_rewards[kyoku])
        dones[idx] = True
    return rewards, dones


def load_trajectory_file(path: Path | str, *, map_location='cpu') -> TrajectoryBatch:
    with open(path, 'rb') as f:
        return unpack_trajectory(f.read(), map_location=map_location)


def collate_trajectory_batches(batches: list[TrajectoryBatch]) -> TrajectoryBatch:
    return TrajectoryBatch(
        obs=torch.cat([b.obs for b in batches], dim=0),
        action=torch.cat([b.action for b in batches], dim=0),
        logp_old=torch.cat([b.logp_old for b in batches], dim=0),
        mask=torch.cat([b.mask for b in batches], dim=0),
        reward=torch.cat([b.reward for b in batches], dim=0),
        done=torch.cat([b.done for b in batches], dim=0),
        param_version=batches[-1].param_version,
    )


def recompute_logp_old(
    batch: TrajectoryBatch,
    brain: Brain,
    actor_critic: ActorCritic,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        obs = batch.obs.to(device=device, dtype=torch.float32)
        masks = batch.mask.to(device=device)
        actions = batch.action.to(device=device)
        phi = brain(obs)
        logits, _ = actor_critic(phi, masks)
        return action_log_probs(logits, masks, actions).cpu()
