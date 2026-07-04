"""PPO opponent engine with per-game checkpoint sampling from OpponentPool."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import torch

from model import ActorCritic, Brain
from opponent_pool import OpponentPool
from ppo_engine import pick_actions_from_logits


class PPOOpponentPoolEngine:
    engine_type = 'mortal'

    _GAME_CKPT_LIMIT = 1000

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
        self.pool = pool
        self.is_oracle = is_oracle
        self.version = version
        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name
        self.eval_mode = eval_mode
        self._conv_channels = brain.encoder.net[0].out_channels
        self._num_blocks = len([
            layer for layer in brain.encoder.net
            if hasattr(layer, 'res_unit')
        ])
        self._tau = actor_critic.tau
        self._models: dict[Path | None, tuple[Brain, ActorCritic]] = {}
        self._game_ckpt: dict[str, Path | None] = {}
        self.illegal_action_fallback_count = 0

    def _new_model_pair(self) -> tuple[Brain, ActorCritic]:
        brain = Brain(
            version=self.version,
            conv_channels=self._conv_channels,
            num_blocks=self._num_blocks,
            is_oracle=self.is_oracle,
        )
        actor_critic = ActorCritic(version=self.version, tau=self._tau)
        return brain.to(self.device).eval(), actor_critic.to(self.device).eval()

    def _cache_limit(self) -> int:
        return self.pool.past_k + 2

    def _evict_models_if_needed(self):
        limit = self._cache_limit()
        if len(self._models) <= limit:
            return
        pool_ckpts = set(self.pool.list_checkpoints())
        if self.pool.fallback_checkpoint is not None:
            pool_ckpts.add(self.pool.fallback_checkpoint)
        evictable = [ckpt for ckpt in self._models if ckpt not in pool_ckpts]
        for ckpt in evictable:
            del self._models[ckpt]
            if len(self._models) <= limit:
                return
        while len(self._models) > limit:
            oldest = next(iter(self._models))
            del self._models[oldest]

    def _get_model(self, ckpt: Path | None) -> tuple[Brain, ActorCritic]:
        if ckpt not in self._models:
            brain, actor_critic = self._new_model_pair()
            self.pool.load_ppo(ckpt, brain, actor_critic, map_location=self.device)
            self._models[ckpt] = (brain, actor_critic)
            self._evict_models_if_needed()
        return self._models[ckpt]

    def _ckpt_for_game(self, game_key: str) -> Path | None:
        if game_key not in self._game_ckpt:
            self._game_ckpt[game_key] = self.pool.sample()
        return self._game_ckpt[game_key]

    def _prune_game_ckpt(self, step_meta):
        if len(self._game_ckpt) <= self._GAME_CKPT_LIMIT:
            return
        active = {
            game_key or f'__anon_{i}'
            for i, (game_key, _seq, _record) in enumerate(step_meta)
        }
        for key in list(self._game_ckpt):
            if key not in active:
                del self._game_ckpt[key]

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
            self._prune_game_ckpt(step_meta)
            groups: dict[Path | None, list[int]] = {}
            for i, (game_key, _seq, _record) in enumerate(step_meta):
                ckpt = self._ckpt_for_game(game_key or f'__anon_{i}')
                groups.setdefault(ckpt, []).append(i)
            actions = [0] * batch_size
            is_greedy = [False] * batch_size
            q_parts: list[torch.Tensor] = []
            for ckpt, idxs in groups.items():
                brain, actor_critic = self._get_model(ckpt)
                sub_obs = torch.as_tensor(np.stack([obs[i] for i in idxs], axis=0), device=self.device)
                sub_masks = torch.as_tensor(np.stack([masks[i] for i in idxs], axis=0), device=self.device)
                sub_actions, sub_greedy, sub_logits = self._forward(brain, actor_critic, sub_obs, sub_masks)
                for j, i in enumerate(idxs):
                    actions[i] = int(sub_actions[j].item())
                    is_greedy[i] = bool(sub_greedy[j].item())
                q_parts.append(sub_logits)
            q_proxy = torch.cat(q_parts, dim=0).tolist()
            return actions, q_proxy, masks, is_greedy

        brain, actor_critic = self._get_model(self.pool.sample())
        obs_t = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks_t = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        actions, is_greedy, q_proxy = self._forward(brain, actor_critic, obs_t, masks_t)
        return actions.tolist(), q_proxy.tolist(), masks, is_greedy.tolist()

    def _forward(self, brain, actor_critic, obs_t, masks_t):
        match self.version:
            case 1:
                mu, logsig = brain(obs_t, None)
                phi = mu
            case 2 | 3 | 4:
                phi = brain(obs_t)

        logits, _values = actor_critic(phi, masks_t)
        actions, fallback = pick_actions_from_logits(logits, masks_t, eval_mode=self.eval_mode)
        self.illegal_action_fallback_count += fallback
        is_greedy = (
            torch.ones(obs_t.shape[0], dtype=torch.bool, device=self.device)
            if self.eval_mode
            else torch.zeros(obs_t.shape[0], dtype=torch.bool, device=self.device)
        )
        return actions, is_greedy, logits.detach()
