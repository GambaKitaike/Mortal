"""Client→trainer trajectory payload: (obs, action, logp_old, mask, reward, done)."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import torch
from torch import Tensor

TRAJECTORY_FIELDS = ('obs', 'action', 'logp_old', 'mask', 'reward', 'done')
REWARD_COMPONENT_FIELDS = ('reward_sotensu', 'reward_grp', 'reward_chip')
OPTIONAL_FIELDS = ('at_kyoku',)


@dataclass
class TrajectoryBatch:
    obs: Tensor
    action: Tensor
    logp_old: Tensor
    mask: Tensor
    reward: Tensor
    done: Tensor
    param_version: int = -1
    reward_sotensu: Tensor | None = None
    reward_grp: Tensor | None = None
    reward_chip: Tensor | None = None
    grp_pred_rank: Tensor | None = None
    grp_actual_rank: Tensor | None = None
    at_kyoku: Tensor | None = None

    def __post_init__(self):
        n = self.obs.shape[0]
        for name in TRAJECTORY_FIELDS[1:]:
            t = getattr(self, name)
            assert t.shape[0] == n, f'{name} batch dim mismatch'
        for name in REWARD_COMPONENT_FIELDS:
            t = getattr(self, name)
            if t is not None:
                assert t.shape[0] == n, f'{name} batch dim mismatch'
        for name in ('grp_pred_rank', 'grp_actual_rank', 'at_kyoku'):
            t = getattr(self, name)
            if t is not None:
                assert t.shape[0] == n, f'{name} batch dim mismatch'

    def to_dict(self) -> dict[str, Any]:
        payload = {
            'format': 'ppo_trajectory_v2',
            'param_version': self.param_version,
            **{k: getattr(self, k) for k in TRAJECTORY_FIELDS},
        }
        for name in REWARD_COMPONENT_FIELDS:
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        for name in ('grp_pred_rank', 'grp_actual_rank', 'at_kyoku'):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrajectoryBatch:
        fmt = d.get('format')
        assert fmt in ('ppo_trajectory_v1', 'ppo_trajectory_v2')
        kwargs = {
            'obs': d['obs'],
            'action': d['action'],
            'logp_old': d['logp_old'],
            'mask': d['mask'],
            'reward': d['reward'],
            'done': d['done'],
            'param_version': d.get('param_version', -1),
        }
        for name in REWARD_COMPONENT_FIELDS:
            kwargs[name] = d.get(name)
        for name in ('grp_pred_rank', 'grp_actual_rank', 'at_kyoku'):
            kwargs[name] = d.get(name)
        return cls(**kwargs)


def pack_trajectory(batch: TrajectoryBatch) -> bytes:
    buf = BytesIO()
    torch.save(batch.to_dict(), buf)
    return buf.getvalue()


def unpack_trajectory(data: bytes, *, map_location='cpu') -> TrajectoryBatch:
    return TrajectoryBatch.from_dict(
        torch.load(BytesIO(data), weights_only=False, map_location=map_location),
    )


def numpy_trajectory_to_batch(steps: list[dict[str, Any]], *, param_version: int = -1) -> TrajectoryBatch:
    kwargs = {
        'obs': torch.as_tensor(np.stack([s['obs'] for s in steps]), dtype=torch.float32),
        'action': torch.as_tensor([s['action'] for s in steps], dtype=torch.int64),
        'logp_old': torch.as_tensor([s['logp_old'] for s in steps], dtype=torch.float32),
        'mask': torch.as_tensor(np.stack([s['mask'] for s in steps]), dtype=torch.bool),
        'reward': torch.as_tensor([s['reward'] for s in steps], dtype=torch.float32),
        'done': torch.as_tensor([s['done'] for s in steps], dtype=torch.bool),
        'param_version': param_version,
    }
    if steps and 'reward_sotensu' in steps[0]:
        kwargs['reward_sotensu'] = torch.as_tensor(
            [s['reward_sotensu'] for s in steps], dtype=torch.float32,
        )
        kwargs['reward_grp'] = torch.as_tensor(
            [s['reward_grp'] for s in steps], dtype=torch.float32,
        )
        kwargs['reward_chip'] = torch.as_tensor(
            [s['reward_chip'] for s in steps], dtype=torch.float32,
        )
    if steps and 'grp_pred_rank' in steps[0]:
        kwargs['grp_pred_rank'] = torch.as_tensor(
            [s['grp_pred_rank'] for s in steps], dtype=torch.float32,
        )
        kwargs['grp_actual_rank'] = torch.as_tensor(
            [s['grp_actual_rank'] for s in steps], dtype=torch.float32,
        )
    if steps and 'at_kyoku' in steps[0]:
        kwargs['at_kyoku'] = torch.as_tensor(
            [s['at_kyoku'] for s in steps], dtype=torch.int64,
        )
    return TrajectoryBatch(**kwargs)
