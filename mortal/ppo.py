"""PPO core: masked policy, GAE, clipped surrogate + value + entropy losses."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    masked = logits.masked_fill(~mask, -torch.inf)
    probs = masked.softmax(-1)
    return probs * mask.to(probs.dtype)


def masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    masked = logits.masked_fill(~mask, -torch.inf)
    return masked.log_softmax(-1)


def action_log_probs(logits: Tensor, mask: Tensor, actions: Tensor) -> Tensor:
    logp_all = masked_log_softmax(logits, mask)
    return logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def policy_entropy(logits: Tensor, mask: Tensor) -> Tensor:
    probs = masked_softmax(logits, mask)
    logp = torch.log(probs.clamp(min=1e-12))
    ent = -(probs * logp).sum(-1)
    return ent


def normalize_advantages(advantages: Tensor) -> Tensor:
    if advantages.numel() <= 1:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """GAE for a single episode (kyoku).

    Args:
        rewards: (T,) step rewards
        values: (T+1,) value estimates; values[T] is bootstrap (0 at terminal kyoku end)
        dones: (T,) bool, True on terminal step of episode
    Returns:
        advantages (T,), returns R̂_GAE (T,) on raw reward scale
    """
    assert rewards.ndim == 1 and values.ndim == 1 and dones.ndim == 1
    assert values.shape[0] == rewards.shape[0] + 1

    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t].to(rewards.dtype)
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def ppo_loss(
    logits: Tensor,
    values: Tensor,
    actions: Tensor,
    masks: Tensor,
    logp_old: Tensor,
    advantages: Tensor,
    returns: Tensor,
    *,
    eps_clip: float = 0.2,
    c_vf: float = 0.5,
    c_ent: float = 0.01,
    huber_delta: float = 15.0,
) -> dict[str, Tensor]:
    """PPO clipped surrogate + Huber value + entropy bonus (no CQL)."""
    adv_norm = normalize_advantages(advantages)
    logp = action_log_probs(logits, masks, actions)
    ratio = (logp - logp_old).exp()
    surr1 = ratio * adv_norm
    surr2 = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip) * adv_norm
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.huber_loss(values, returns, delta=huber_delta)
    entropy = policy_entropy(logits, masks).mean()
    total = policy_loss + c_vf * value_loss - c_ent * entropy
    return {
        'total': total,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
    }


def compose_kyoku_reward(
    sotensu_delta: float,
    juni_delta: float,
    chip_delta: float = 0.0,
    *,
    alpha: float = 1.0,
    gamma_pt: float = 1.0,
    beta: float = 1.0,
    chip_value: float = 5.0,
) -> float:
    """reward_design_teacherfree.md §2 three-term blend (no opp term)."""
    return alpha * sotensu_delta + gamma_pt * juni_delta + beta * chip_delta * chip_value
