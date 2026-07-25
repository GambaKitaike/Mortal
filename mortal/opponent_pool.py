"""Opponent checkpoint pool for PPO self-play (design §4)."""

from __future__ import annotations

import random
import re
from pathlib import Path

import torch

_STEP_RE = re.compile(r'step_(\d+)\.pth$')


class OpponentPool:
    """Sample opponent checkpoints: latest_prob latest, else uniform over past K."""

    def __init__(
        self,
        ckpt_dir: str | Path,
        *,
        past_k: int = 5,
        latest_prob: float = 0.5,
        fallback_checkpoint: str | Path | None = None,
        anchor_prob: float = 0.0,
        anchor_checkpoint: str | Path | None = None,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.past_k = past_k
        self.latest_prob = latest_prob
        self.fallback_checkpoint = Path(fallback_checkpoint) if fallback_checkpoint else None
        self.anchor_prob = float(anchor_prob)
        self.anchor_checkpoint = Path(anchor_checkpoint) if anchor_checkpoint else None
        self.last_draw_kind: str | None = None
        if self.anchor_prob > 0.0:
            if self.anchor_checkpoint is None or not self.anchor_checkpoint.is_file():
                raise ValueError(
                    f'anchor_prob={self.anchor_prob} > 0 but anchor_checkpoint is missing or not a file: '
                    f'{anchor_checkpoint!r}'
                )

    def list_checkpoints(self) -> list[Path]:
        if not self.ckpt_dir.is_dir():
            return []
        cks = [p for p in self.ckpt_dir.glob('step_*.pth') if _STEP_RE.search(p.name)]
        return sorted(cks, key=lambda p: int(_STEP_RE.search(p.name).group(1)))

    def sample(self) -> tuple[Path | None, str]:
        if self.anchor_prob > 0.0 and random.random() < self.anchor_prob:
            self.last_draw_kind = 'anchor'
            return self.anchor_checkpoint, 'anchor'
        cks = self.list_checkpoints()
        if not cks:
            self.last_draw_kind = 'fallback'
            return self.fallback_checkpoint, 'fallback'
        latest = cks[-1]
        if len(cks) == 1 or random.random() < self.latest_prob:
            self.last_draw_kind = 'latest'
            return latest, 'latest'
        past = cks[-(self.past_k + 1):-1]
        if not past:
            self.last_draw_kind = 'latest'
            return latest, 'latest'
        self.last_draw_kind = 'past'
        return random.choice(past), 'past'

    def load_ppo(self, checkpoint: Path | None, brain, actor_critic, *, map_location='cpu') -> bool:
        if checkpoint is None or not Path(checkpoint).is_file():
            return False
        state = torch.load(checkpoint, weights_only=True, map_location=map_location)
        brain.load_state_dict(state['mortal'])
        if 'actor_critic' in state:
            actor_critic.load_state_dict(state['actor_critic'])
            return True
        from model import load_ppo_from_mortal_checkpoint
        load_ppo_from_mortal_checkpoint(actor_critic, str(checkpoint), map_location=map_location)
        return True
