"""PPO opponent engine with per-game checkpoint sampling from OpponentPool."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from opponent_pool import OpponentPool
from ppo import action_log_probs


class PPOOpponentPoolEngine:
    engine_type = 'mortal'

    def __init__(
        self,
        brain,
        actor_critic,
        pool: OpponentPool,
        *,
        is_oracle=False,
        version,
        device=None,
        enable_amp=False,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=False,
        name='opp_pool',
        eval_mode=False,
    ):
        self.device = device or torch.device('cpu')
        self.brain = brain.to(self.device).eval()
        self.actor_critic = actor_critic.to(self.device).eval()
        self.pool = pool
        self.is_oracle = is_oracle
        self.version = version
        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name
        self.eval_mode = eval_mode
        self._game_ckpt: dict[str, Path | None] = {}
        self._loaded_ckpt: Path | None = object()  # sentinel

    def _ckpt_for_game(self, game_key: str) -> Path | None:
        if game_key not in self._game_ckpt:
            self._game_ckpt[game_key] = self.pool.sample()
        return self._game_ckpt[game_key]

    def _ensure_weights(self, ckpt: Path | None):
        if ckpt == self._loaded_ckpt:
            return
        self.pool.load_ppo(ckpt, self.brain, self.actor_critic, map_location=self.device)
        self._loaded_ckpt = ckpt

    def react_batch(self, obs, masks, invisible_obs, step_meta=None):
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch(obs, masks, invisible_obs, step_meta)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}') from ex

    def _react_batch(self, obs, masks, invisible_obs, step_meta=None):
        del invisible_obs
        batch_size = len(obs)
        if step_meta is not None and len(step_meta) == batch_size:
            groups: dict[Path | None, list[int]] = {}
            for i, (game_key, _seq, _record) in enumerate(step_meta):
                ckpt = self._ckpt_for_game(game_key or f'__anon_{i}')
                groups.setdefault(ckpt, []).append(i)
            actions = [0] * batch_size
            is_greedy = [False] * batch_size
            q_parts: list[torch.Tensor] = []
            for ckpt, idxs in groups.items():
                self._ensure_weights(ckpt)
                sub_obs = torch.as_tensor(np.stack([obs[i] for i in idxs], axis=0), device=self.device)
                sub_masks = torch.as_tensor(np.stack([masks[i] for i in idxs], axis=0), device=self.device)
                sub_actions, sub_greedy, sub_logits = self._forward(sub_obs, sub_masks)
                for j, i in enumerate(idxs):
                    actions[i] = int(sub_actions[j].item())
                    is_greedy[i] = bool(sub_greedy[j].item())
                q_parts.append(sub_logits)
            q_proxy = torch.cat(q_parts, dim=0).tolist()
            return actions, q_proxy, masks, is_greedy

        self._ensure_weights(self.pool.sample())
        obs_t = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks_t = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        actions, is_greedy, q_proxy = self._forward(obs_t, masks_t)
        return actions.tolist(), q_proxy.tolist(), masks, is_greedy.tolist()

    def _forward(self, obs_t, masks_t):
        match self.version:
            case 1:
                mu, logsig = self.brain(obs_t, None)
                phi = mu
            case 2 | 3 | 4:
                phi = self.brain(obs_t)

        logits, _values = self.actor_critic(phi, masks_t)
        masked_logits = logits.masked_fill(~masks_t, -torch.inf)

        if self.eval_mode:
            actions = masked_logits.argmax(-1)
            is_greedy = torch.ones(obs_t.shape[0], dtype=torch.bool, device=self.device)
        else:
            actions = Categorical(logits=masked_logits).sample()
            is_greedy = torch.zeros(obs_t.shape[0], dtype=torch.bool, device=self.device)
            illegal = ~masks_t.gather(1, actions.unsqueeze(1)).squeeze(1)
            if illegal.any():
                actions[illegal] = masked_logits[illegal].argmax(-1)

        return actions, is_greedy, logits.detach()
