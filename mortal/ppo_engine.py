"""PPO inference engine: saves logp_old at action time for trajectory transport."""

import traceback

import numpy as np
import torch
from torch.distributions import Categorical

from model import ActorCritic, Brain
from ppo import action_log_probs


class PPOEngine:
    engine_type = 'mortal'

    def __init__(
        self,
        brain: Brain,
        actor_critic: ActorCritic,
        *,
        is_oracle=False,
        version,
        device=None,
        enable_amp=False,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=False,
        name='NoName',
        eval_mode=False,
        record_trajectory=True,
    ):
        self.device = device or torch.device('cpu')
        self.brain = brain.to(self.device).eval()
        self.actor_critic = actor_critic.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name
        self.eval_mode = eval_mode
        self.record_trajectory = record_trajectory
        self.pending_by_game: dict[str, dict[int, dict]] = {}
        self.pending_steps: list[dict] = []

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
        obs_t = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks_t = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        batch_size = obs_t.shape[0]

        match self.version:
            case 1:
                mu, logsig = self.brain(obs_t, invisible_obs)
                phi = mu
            case 2 | 3 | 4:
                phi = self.brain(obs_t)

        logits, _values = self.actor_critic(phi, masks_t)
        masked_logits = logits.masked_fill(~masks_t, -torch.inf)

        if self.eval_mode:
            actions = masked_logits.argmax(-1)
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        else:
            actions = Categorical(logits=masked_logits).sample()
            is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        logp_old = action_log_probs(logits, masks_t, actions)

        if self.record_trajectory:
            obs_cpu = obs_t.float().cpu().numpy()
            masks_cpu = masks_t.cpu().numpy()
            actions_cpu = actions.cpu().numpy()
            logp_cpu = logp_old.float().cpu().numpy()
            use_keys = step_meta is not None and len(step_meta) == batch_size
            for i in range(batch_size):
                if use_keys:
                    game_id, seq, record = step_meta[i]
                    if not record or not game_id:
                        continue
                    self.pending_by_game.setdefault(game_id, {})[int(seq)] = {
                        'obs': obs_cpu[i],
                        'action': int(actions_cpu[i]),
                        'logp_old': float(logp_cpu[i]),
                        'mask': masks_cpu[i],
                        'game_id': game_id,
                        'seq': int(seq),
                    }
                else:
                    self.pending_steps.append({
                        'obs': obs_cpu[i],
                        'action': int(actions_cpu[i]),
                        'logp_old': float(logp_cpu[i]),
                        'mask': masks_cpu[i],
                    })

        q_proxy = logits.detach()
        return (
            actions.tolist(),
            q_proxy.tolist(),
            masks_t.tolist(),
            is_greedy.tolist(),
        )

    def drain_pending(self) -> dict[str, list[dict]] | list[dict]:
        if self.pending_by_game:
            out = {}
            for game_id, steps_by_seq in self.pending_by_game.items():
                out[game_id] = [steps_by_seq[k] for k in sorted(steps_by_seq)]
            self.pending_by_game = {}
            return out

        steps = self.pending_steps
        self.pending_steps = []
        return steps
