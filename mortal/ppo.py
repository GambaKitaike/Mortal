"""PPO core: masked policy, GAE, clipped surrogate + value + entropy losses."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

# Action-space / obs-layout constants shared by the trainer instrumentation
# (train_ppo.py) and the call-bonus helpers / verify checks below. Moved from
# train_ppo.py unchanged so the definitions have a single source.
CALL_ACTION_MIN = 38
CALL_ACTION_MAX = 42
RIICHI_ACTION = 37
AKA_OBS_ROWS = slice(4, 7)


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


def call_bonus_coeff(step: int, b: float, full_until_step: int, zero_at_step: int) -> float:
    """Stage3 anneal 付き鳴きボーナスの係数 b(step) (stage3_design.md §2).

    b == 0.0 (config キー不在の設計された OFF) では常に 0.0。それ以外は
    [0, full_until_step) で b 一定、[full_until_step, zero_at_step) で
    線形 anneal b→0、zero_at_step 以降は 0.0（正典報酬のみの判定窓）。
    """
    if b == 0.0:
        return 0.0
    if step < full_until_step:
        return b
    if step < zero_at_step:
        return b * (zero_at_step - step) / (zero_at_step - full_until_step)
    return 0.0


def apply_call_bonus(
    rewards: Tensor,
    obs: Tensor,
    masks: Tensor,
    actions: Tensor,
    b_now: float,
) -> tuple[Tensor, int]:
    """鳴き可能∧赤保持∧鳴き実行の decision step に b_now を加算 (stage3_design.md §2).

    選択集合 sel の定義は既存計装 (train_ppo.py action_mass / advantage_decomp
    の n_call_possible 系) と同一。b_now == 0.0 のときは入力 rewards テンソルを
    そのまま返す (コピーも加算もしない — OFF 時ビット不変の保証)。
    """
    call_possible = masks[:, CALL_ACTION_MIN:CALL_ACTION_MAX + 1].any(dim=1)
    aka_held = obs[:, AKA_OBS_ROWS, :].abs().sum(dim=(1, 2)) > 0
    call_taken = (actions >= CALL_ACTION_MIN) & (actions <= CALL_ACTION_MAX)
    sel = call_possible & aka_held & call_taken
    n_applied = int(sel.sum())
    if b_now == 0.0:
        return rewards, n_applied
    return rewards + b_now * sel.to(rewards.dtype), n_applied


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
