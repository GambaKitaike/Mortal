"""PPO inference engine: saves logp_old at action time for trajectory transport."""

import traceback

import numpy as np
import torch
from torch.distributions import Categorical

from model import ActorCritic, Brain
from ppo import action_log_probs


def pick_actions_from_logits(
    logits: torch.Tensor,
    masks_t: torch.Tensor,
    *,
    eval_mode: bool,
) -> tuple[torch.Tensor, int]:
    # float32: AMP 下の -inf mask が illegal action を argmax するのを防ぐ
    masked = logits.float().masked_fill(~masks_t, -1e9)
    if eval_mode:
        actions = masked.argmax(-1)
    else:
        actions = Categorical(logits=masked).sample()
    fallback = 0
    for i in range(masks_t.shape[0]):
        if not masks_t[i, actions[i]]:
            fallback += 1
            legal = masks_t[i].nonzero(as_tuple=True)[0]
            if legal.numel() == 0:
                actions[i] = 0
            else:
                actions[i] = legal[masked[i, legal].argmax()]
    return actions, fallback


def dump_engine_config(engine) -> dict:
    tau = getattr(getattr(engine, 'actor_critic', None), 'tau', None)
    return {
        'name': engine.name,
        'engine_type': getattr(engine, 'engine_type', None),
        'enable_amp': engine.enable_amp,
        'enable_quick_eval': engine.enable_quick_eval,
        'enable_rule_based_agari_guard': engine.enable_rule_based_agari_guard,
        'eval_mode': getattr(engine, 'eval_mode', None),
        'record_trajectory': getattr(engine, 'record_trajectory', None),
        'tau': tau,
        'has_pending_steps': hasattr(engine, 'pending_steps'),
        # Stage2 赤濃縮 (stage2_design.md §2). Absent on non-PPO / opponent
        # engines, which is intentionally equivalent to 0.0 (no-op) here and
        # on the Rust side (libriichi arena OneVsThree::py_vs_py).
        'p_enrich': getattr(engine, 'p_enrich', 0.0),
    }


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
        p_enrich=0.0,
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
        # Stage2 赤濃縮 (stage2_design.md §2). Read by the Rust arena
        # (OneVsThree::py_vs_py) off the *challenger* engine only — the
        # trainee client is always passed as `challenger`, so this attribute
        # has no effect when this engine is used as `champion`/opponent.
        self.p_enrich = p_enrich
        self.pending_by_game: dict[str, dict[int, dict]] = {}
        self.pending_steps: list[dict] = []
        self.illegal_action_fallback_count = 0

    def _pick_actions(self, logits: torch.Tensor, masks_t: torch.Tensor, *, eval_mode: bool) -> torch.Tensor:
        actions, fallback = pick_actions_from_logits(logits, masks_t, eval_mode=eval_mode)
        self.illegal_action_fallback_count += fallback
        return actions

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
        actions = self._pick_actions(logits, masks_t, eval_mode=self.eval_mode)
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device) if self.eval_mode else torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        logp_old = action_log_probs(logits, masks_t, actions)

        if self.record_trajectory:
            obs_cpu = obs_t.float().cpu().numpy()
            masks_cpu = masks_t.cpu().numpy()
            actions_cpu = actions.cpu().numpy()
            logp_cpu = logp_old.float().cpu().numpy()
            use_keys = step_meta is not None and len(step_meta) == batch_size
            for i in range(batch_size):
                if use_keys:
                    meta = step_meta[i]
                    game_id = meta[0]
                    seq = meta[1]
                    record = meta[2]
                    at_kyoku = int(meta[3]) if len(meta) > 3 else 0
                    if not record or not game_id:
                        continue
                    self.pending_by_game.setdefault(game_id, {})[int(seq)] = {
                        'obs': obs_cpu[i],
                        'action': int(actions_cpu[i]),
                        'logp_old': float(logp_cpu[i]),
                        'mask': masks_cpu[i],
                        'game_id': game_id,
                        'seq': int(seq),
                        'at_kyoku': at_kyoku,
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


def build_production_trainee_engine(
    brain: Brain,
    actor_critic: ActorCritic,
    *,
    version,
    device=None,
    name: str = 'trainee',
    p_enrich: float = 0.0,
) -> PPOEngine:
    """Same kwargs as mortal/client.py train-rollout PPOEngine.

    p_enrich defaults to 0.0 (natural haipai distribution). Only the actual
    train-rollout call site (mortal/client.py) should pass a non-zero value,
    read from config['ppo']['p_enrich'] — eval call sites (player.py
    TestPlayer, verify_ppo_p1.py) must never set it, keeping eval always on
    the natural distribution per stage2_design.md §2.
    """
    return PPOEngine(
        brain,
        actor_critic,
        is_oracle=False,
        version=version,
        device=device,
        enable_amp=True,
        enable_quick_eval=False,
        name=name,
        p_enrich=p_enrich,
    )
