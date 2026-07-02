"""Client→trainer trajectory payload: (obs, action, logp_old, mask, reward, done)."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import torch
from torch import Tensor

TRAJECTORY_FIELDS = ('obs', 'action', 'logp_old', 'mask', 'reward', 'done')


@dataclass
class TrajectoryBatch:
    obs: Tensor
    action: Tensor
    logp_old: Tensor
    mask: Tensor
    reward: Tensor
    done: Tensor
    param_version: int = -1

    def __post_init__(self):
        n = self.obs.shape[0]
        for name in TRAJECTORY_FIELDS[1:]:
            t = getattr(self, name)
            assert t.shape[0] == n, f'{name} batch dim mismatch'

    def to_dict(self) -> dict[str, Any]:
        return {
            'format': 'ppo_trajectory_v1',
            'param_version': self.param_version,
            **{k: getattr(self, k) for k in TRAJECTORY_FIELDS},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrajectoryBatch:
        assert d.get('format') == 'ppo_trajectory_v1'
        return cls(
            obs=d['obs'],
            action=d['action'],
            logp_old=d['logp_old'],
            mask=d['mask'],
            reward=d['reward'],
            done=d['done'],
            param_version=d.get('param_version', -1),
        )


def pack_trajectory(batch: TrajectoryBatch) -> bytes:
    buf = BytesIO()
    torch.save(batch.to_dict(), buf)
    return buf.getvalue()


def unpack_trajectory(data: bytes, *, map_location='cpu') -> TrajectoryBatch:
    return TrajectoryBatch.from_dict(
        torch.load(BytesIO(data), weights_only=False, map_location=map_location),
    )


def numpy_trajectory_to_batch(steps: list[dict[str, Any]], *, param_version: int = -1) -> TrajectoryBatch:
    return TrajectoryBatch(
        obs=torch.as_tensor(np.stack([s['obs'] for s in steps]), dtype=torch.float32),
        action=torch.as_tensor([s['action'] for s in steps], dtype=torch.int64),
        logp_old=torch.as_tensor([s['logp_old'] for s in steps], dtype=torch.float32),
        mask=torch.as_tensor(np.stack([s['mask'] for s in steps]), dtype=torch.bool),
        reward=torch.as_tensor([s['reward'] for s in steps], dtype=torch.float32),
        done=torch.as_tensor([s['done'] for s in steps], dtype=torch.bool),
        param_version=param_version,
    )
