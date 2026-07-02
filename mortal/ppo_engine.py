"""PPO inference engine: saves logp_old at action time for trajectory transport."""

import traceback

import numpy as np
import torch

from engine import sample_top_p
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
        boltzmann_epsilon=0,
        top_p=1,
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
        self.boltzmann_epsilon = boltzmann_epsilon
        self.top_p = top_p
        self.pending_steps: list[dict] = []

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}') from ex

    def _react_batch(self, obs, masks, invisible_obs):
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

        if self.boltzmann_epsilon > 0:
            is_greedy = torch.full(
                (batch_size,), 1 - self.boltzmann_epsilon, device=self.device,
            ).bernoulli().to(torch.bool)
            sampled = sample_top_p(logits, self.top_p)
            greedy_actions = logits.masked_fill(~masks_t, -torch.inf).argmax(-1)
            actions = torch.where(is_greedy, greedy_actions, sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = logits.masked_fill(~masks_t, -torch.inf).argmax(-1)

        logp_old = action_log_probs(logits, masks_t, actions)

        obs_cpu = obs_t.float().cpu().numpy()
        masks_cpu = masks_t.cpu().numpy()
        actions_cpu = actions.cpu().numpy()
        logp_cpu = logp_old.float().cpu().numpy()
        for i in range(batch_size):
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

    def drain_pending(self) -> list[dict]:
        steps = self.pending_steps
        self.pending_steps = []
        return steps
