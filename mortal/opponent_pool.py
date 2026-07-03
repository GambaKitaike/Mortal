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
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.past_k = past_k
        self.latest_prob = latest_prob
        self.fallback_checkpoint = Path(fallback_checkpoint) if fallback_checkpoint else None

    def list_checkpoints(self) -> list[Path]:
        if not self.ckpt_dir.is_dir():
            return []
        cks = [p for p in self.ckpt_dir.glob('step_*.pth') if _STEP_RE.search(p.name)]
        return sorted(cks, key=lambda p: int(_STEP_RE.search(p.name).group(1)))

    def sample(self) -> Path | None:
        cks = self.list_checkpoints()
        if not cks:
            return self.fallback_checkpoint
        latest = cks[-1]
        if len(cks) == 1 or random.random() < self.latest_prob:
            return latest
        past = cks[-(self.past_k + 1):-1]
        if not past:
            return latest
        return random.choice(past)

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
