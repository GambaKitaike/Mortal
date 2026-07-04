#!/usr/bin/env python3
"""PPO P1 plumbing sanity checks (design §7.1)."""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mortal'))

from libriichi.consts import ACTION_SPACE, obs_shape
from libriichi.dataset import GameplayLoader
from chip_from_log import load_kyoku_chip_deltas_from_log
from model import (
    ActorCritic,
    Brain,
    DQN,
    GRP,
    dqn_a_head_logits,
    load_actor_critic_from_dqn_checkpoint,
    load_ppo_from_mortal_checkpoint,
)
from ppo import compose_kyoku_reward, compute_gae, masked_softmax, action_log_probs
from ppo_engine import PPOEngine, dump_engine_config
from ppo_dataloader import assign_rewards_and_dones, recompute_logp_old
from ppo_transport import TrajectoryBatch, pack_trajectory, unpack_trajectory
from reward_calculator import RewardCalculator


DEFAULT_CKPT = '/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth'
DEFAULT_ONLINE_LOG = '/home/gamba/mahjong/runs_archive_0620/test_play/10299_8192_d.json.gz'


def log(msg: str, buf: StringIO):
    print(msg)
    buf.write(msg + '\n')


def check_pi0_match(ckpt: str, buf: StringIO, *, tau: float = 1.0, n: int = 32):
    log('(1) π₀ consistency', buf)
    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    version = state['config']['control']['version']
    conv_channels = state['config']['resnet']['conv_channels']
    num_blocks = state['config']['resnet']['num_blocks']

    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    mortal.load_state_dict(state['mortal'])
    dqn = DQN(version=version).eval()
    dqn.load_state_dict(state['current_dqn'], strict=False)
    ac = ActorCritic(version=version, tau=tau).eval()
    load_actor_critic_from_dqn_checkpoint(ac, state['current_dqn'], version=version)

    c, w = obs_shape(version)
    obs = torch.randn(n, c, w)
    masks = torch.zeros(n, ACTION_SPACE, dtype=torch.bool)
    for i in range(n):
        legal = torch.randperm(ACTION_SPACE)[:8]
        masks[i, legal] = True

    with torch.inference_mode():
        phi = mortal(obs)
        a_logits = dqn_a_head_logits(dqn, phi) / tau
        pi_logits, _ = ac(phi, masks)
        p_ref = masked_softmax(a_logits, masks)
        p_new = masked_softmax(pi_logits, masks)
        assert torch.allclose(p_ref, p_new, atol=1e-5), 'π₀ mismatch'
    log('  PASS: softmax(policy/τ) ≡ softmax(a_head/τ)', buf)


def check_mask(buf: StringIO):
    log('(2) illegal action mask', buf)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, False, True, False]])
    probs = masked_softmax(logits, mask)
    assert (probs[~mask] == 0).all()
    assert torch.allclose(probs[mask].sum(), torch.tensor(1.0), atol=1e-6)
    log('  PASS: illegal actions have π=0', buf)


def check_gae(buf: StringIO):
    log('(3) GAE 3-step hand calc', buf)
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 1.0, 2.0, 0.0])
    dones = torch.tensor([False, False, True])
    adv, ret = compute_gae(rewards, values, dones, gamma=1.0, lam=0.95)

    delta2 = rewards[2] - values[2]
    adv2 = delta2
    delta1 = rewards[1] + values[2] - values[1]
    adv1 = delta1 + 0.95 * adv2
    delta0 = rewards[0] + values[1] - values[0]
    adv0 = delta0 + 0.95 * adv1
    expected_adv = torch.tensor([adv0, adv1, adv2])
    expected_ret = expected_adv + values[:-1]
    assert torch.allclose(adv, expected_adv, atol=1e-6), f'adv {adv} vs {expected_adv}'
    assert torch.allclose(ret, expected_ret, atol=1e-6)
    log(f'  adv={adv.tolist()}', buf)
    log('  PASS: GAE matches manual 3-step', buf)


def check_logp_old(ckpt: str, buf: StringIO):
    log('(4) logp_old client vs trainer', buf)
    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    version = state['config']['control']['version']
    cfg = state['config']['resnet']
    device = torch.device('cpu')

    mortal = Brain(version=version, **cfg).eval()
    mortal.load_state_dict(state['mortal'])
    ac = ActorCritic(version=version, tau=1.0).eval()
    load_actor_critic_from_dqn_checkpoint(ac, state['current_dqn'], version=version)

    c, w = obs_shape(version)
    obs_np = np.random.randn(4, c, w).astype(np.float32)
    masks_np = np.zeros((4, ACTION_SPACE), dtype=bool)
    for i in range(4):
        masks_np[i, np.random.choice(ACTION_SPACE, 6, replace=False)] = True

    engine = PPOEngine(
        mortal, ac, version=version, device=device,
    )
    actions, _, _, _ = engine.react_batch(
        list(obs_np), list(masks_np), None,
    )
    client_steps = engine.drain_pending()
    client_logp = np.array([s['logp_old'] for s in client_steps])

    batch = TrajectoryBatch(
        obs=torch.as_tensor(obs_np),
        action=torch.as_tensor(actions, dtype=torch.int64),
        logp_old=torch.as_tensor(client_logp),
        mask=torch.as_tensor(masks_np),
        reward=torch.zeros(4),
        done=torch.tensor([False, False, False, True]),
    )
    trainer_logp = recompute_logp_old(batch, mortal, ac, device).numpy()
    assert np.allclose(client_logp, trainer_logp, atol=1e-5), f'{client_logp} vs {trainer_logp}'
    log('  PASS: client logp_old == trainer recompute', buf)


def check_reward_compose(buf: StringIO, grp_state_file: str | None):
    log('(5) reward composition vs calc_delta_blend', buf)
    score_only = compose_kyoku_reward(0.5, 0.2, chip_delta=0.0)
    chip_only = compose_kyoku_reward(0.0, 0.0, chip_delta=1.0)
    mixed = compose_kyoku_reward(0.5, 0.2, chip_delta=1.0)
    assert score_only == 0.7
    assert chip_only == 5.0
    assert mixed == 5.7
    log('  PASS: compose helper unit cases', buf)

    if grp_state_file is None or not Path(grp_state_file).exists():
        log('  SKIP: calc_delta_blend cross-check (grp state missing)', buf)
        return

    grp = GRP(hidden_size=64, num_layers=2).eval()
    grp.load_state_dict(torch.load(grp_state_file, weights_only=True, map_location='cpu')['model'])
    calc = RewardCalculator(grp, pts=[35, 5, -15, -25], alpha=1.0, gamma_pt=1.0)
    player_id = 0
    grp_feature = np.array([
        [0, 0, 0, 2.5, 2.5, 2.5, 2.5],
        [1, 0, 0, 2.5, 2.5, 2.5, 2.6],
    ], dtype=np.float64)
    rank_by_player = np.array([0, 1, 2, 3])
    final_scores = np.array([26000, 25000, 25000, 24000])

    sotensu = calc.calc_delta_points(player_id, grp_feature, final_scores) / 1000.0
    juni = calc.calc_delta_pt(player_id, grp_feature, rank_by_player)

    chip_zeros = np.zeros(len(grp_feature))
    blend_score = calc.calc_delta_blend(
        player_id, grp_feature, rank_by_player, final_scores,
        chip_deltas=chip_zeros, beta=1.0, chip_value=5.0, lambda_opp=0.0,
    )
    assert np.isclose(
        blend_score[0],
        compose_kyoku_reward(sotensu[0], juni[0], 0.0),
        atol=1e-9,
    ), 'score-only kyoku mismatch'

    chip_one = np.array([0.0, 1.0])
    blend_chip = calc.calc_delta_blend(
        player_id, grp_feature, rank_by_player, final_scores,
        chip_deltas=chip_one, beta=1.0, chip_value=5.0, lambda_opp=0.0,
    )
    assert np.isclose(
        blend_chip[1],
        compose_kyoku_reward(sotensu[1], juni[1], 1.0),
        atol=1e-9,
    ), 'mixed kyoku mismatch'

    chip_only_arr = np.array([1.0, 0.0])
    blend_chip_only = calc.calc_delta_blend(
        player_id, grp_feature, rank_by_player, final_scores,
        alpha=0.0, gamma_pt=0.0,
        chip_deltas=chip_only_arr, beta=1.0, chip_value=5.0, lambda_opp=0.0,
    )
    assert np.isclose(blend_chip_only[0], 5.0, atol=1e-9), 'chip-only kyoku mismatch'
    log('  PASS: calc_delta_blend ≡ compose (score / chip / mixed)', buf)


def check_sampling(ckpt: str, buf: StringIO, *, n_samples: int = 10000):
    log('(7) pure π sampling', buf)
    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    version = state['config']['control']['version']
    cfg = state['config']['resnet']
    device = torch.device('cpu')

    mortal = Brain(version=version, **cfg).eval()
    mortal.load_state_dict(state['mortal'])
    ac = ActorCritic(version=version, tau=1.0).eval()
    load_actor_critic_from_dqn_checkpoint(ac, state['current_dqn'], version=version)

    c, w = obs_shape(version)
    obs_np = np.random.randn(c, w).astype(np.float32)
    mask_np = np.zeros(ACTION_SPACE, dtype=bool)
    legal = np.random.choice(ACTION_SPACE, 12, replace=False)
    mask_np[legal] = True

    engine = PPOEngine(mortal, ac, version=version, device=device)
    counts = np.zeros(ACTION_SPACE, dtype=np.int64)
    illegal = 0
    for _ in range(n_samples):
        actions, _, _, is_greedy = engine.react_batch([obs_np], [mask_np], None)
        engine.drain_pending()
        action = actions[0]
        if not mask_np[action]:
            illegal += 1
        counts[action] += 1
        assert not is_greedy[0], 'training client must sample, not argmax'

    obs_t = torch.as_tensor(obs_np).unsqueeze(0)
    mask_t = torch.as_tensor(mask_np).unsqueeze(0)
    with torch.inference_mode():
        phi = mortal(obs_t)
        logits, _ = ac(phi, mask_t)
        expected = masked_softmax(logits, mask_t)[0].numpy()

    empirical = counts / n_samples
    tv = 0.5 * np.abs(empirical - expected).sum()
    assert illegal == 0, f'illegal actions sampled: {illegal}'
    assert tv < 0.02, f'TV distance {tv:.4f} >= 0.02'
    log(f'  TV distance={tv:.5f}, illegal={illegal}', buf)
    log('  PASS: samples match softmax(masked logits), no illegal actions', buf)


def check_online_chips(buf: StringIO, log_path: str, version: int):
    log('(8) online chip from log', buf)
    path = Path(log_path)
    assert path.exists(), f'log not found: {path}'

    loader = GameplayLoader(version=version, player_names=None)
    data = loader.load_gz_log_files([str(path)])
    assert data[0], f'no games parsed from {path}'
    game = data[0][0]
    player_id = game.take_player_id()
    grp_feature = game.take_grp().take_feature()
    n_kyoku = len(grp_feature)

    chip_deltas = load_kyoku_chip_deltas_from_log(path, player_id, n_kyoku)
    assert np.any(chip_deltas != 0), f'all-zero chip_deltas from {path}'
    nonzero_kyoku = int(np.count_nonzero(chip_deltas))
    log(
        f'  player_id={player_id} n_kyoku={n_kyoku} '
        f'nonzero_kyoku={nonzero_kyoku} sum={chip_deltas.sum():.1f}',
        buf,
    )
    log('  PASS: log-derived chip_delta has non-zero entries', buf)


def check_dqn_one_vs_three(ckpt: str, buf: StringIO):
    log('(9) DQN MortalEngine OneVsThree regression (1 hanchan)', buf)
    from engine import MortalEngine
    from libriichi.arena import OneVsThree

    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    version = state['config']['control']['version']
    cfg = state['config']['resnet']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mortal = Brain(version=version, **cfg).eval()
    mortal.load_state_dict(state['mortal'])
    dqn = DQN(version=version).eval()
    dqn.load_state_dict(state['current_dqn'], strict=False)

    baseline_state = torch.load('/home/gamba/mahjong/runs/grp_baseline.pth', weights_only=True, map_location='cpu')
    bcfg = baseline_state['config']
    bversion = bcfg['control'].get('version', 1)
    b_mortal = Brain(
        version=bversion,
        conv_channels=bcfg['resnet']['conv_channels'],
        num_blocks=bcfg['resnet']['num_blocks'],
    ).eval()
    b_dqn = DQN(version=bversion).eval()
    b_mortal.load_state_dict(baseline_state['mortal'])
    b_dqn.load_state_dict(baseline_state['current_dqn'], strict=False)

    engine = MortalEngine(
        mortal, dqn,
        is_oracle=False,
        version=version,
        device=device,
        enable_amp=True,
        enable_rule_based_agari_guard=True,
        name='mortal',
    )
    baseline = MortalEngine(
        b_mortal, b_dqn,
        is_oracle=False,
        version=bversion,
        device=device,
        enable_amp=True,
        enable_rule_based_agari_guard=True,
        name='baseline',
    )

    with tempfile.TemporaryDirectory(prefix='ppo_p1_dqn_1v3_') as tmp:
        log_dir = Path(tmp)
        env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
        env.py_vs_py(
            challenger=engine,
            champion=baseline,
            seed_start=(99999, 0x2000),
            seed_count=1,
        )
        n_logs = sum(1 for _ in log_dir.glob('*.json.gz'))
        assert n_logs > 0, f'no json.gz produced in {log_dir}'
    log(f'  PASS: grp_baseline path, 1 hanchan, json.gz={n_logs}', buf)


def check_train_engine_config(ckpt: str, buf: StringIO):
    log('(10) train client engine config (guard OFF / eval_mode False / record_trajectory)', buf)
    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    version = state['config']['control']['version']
    cfg = state['config']['resnet']
    device = torch.device('cpu')

    mortal = Brain(version=version, **cfg).eval()
    mortal.load_state_dict(state['mortal'])
    ac = ActorCritic(version=version, tau=1.0).eval()
    load_actor_critic_from_dqn_checkpoint(ac, state['current_dqn'], version=version)

    engine = PPOEngine(
        mortal, ac, version=version, device=device,
        enable_amp=True, enable_quick_eval=False, name='trainee',
    )
    dumped = dump_engine_config(engine)
    log(f'  trainee dump: {dumped}', buf)
    assert not engine.enable_rule_based_agari_guard
    assert not engine.eval_mode
    assert engine.record_trajectory
    log('  PASS: trainee guard=False eval_mode=False record_trajectory=True', buf)


def check_pool_engine_no_trajectory(buf: StringIO):
    log('(11) opponent pool engine has no pending_steps', buf)
    from opponent_pool import OpponentPool
    from ppo_pool_engine import PPOOpponentPoolEngine

    with tempfile.TemporaryDirectory(prefix='ppo_p1_pool_') as tmp:
        ckpt_dir = Path(tmp)
        init0 = ckpt_dir / 'step_000000.pth'
        torch.save({'mortal': {}, 'actor_critic': {}, 'steps': 0}, init0)
        pool = OpponentPool(ckpt_dir, past_k=5, latest_prob=0.5, fallback_checkpoint=init0)
        mortal = Brain(version=4, conv_channels=192, num_blocks=40).eval()
        ac = ActorCritic(version=4, tau=1.0).eval()
        opp = PPOOpponentPoolEngine(
            mortal, ac, pool, version=4, device=torch.device('cpu'),
            enable_amp=False, name='opp_pool', eval_mode=False,
        )
        dumped = dump_engine_config(opp)
        log(f'  pool dump: {dumped}', buf)
        assert not opp.enable_rule_based_agari_guard
        assert not hasattr(opp, 'pending_steps')
        assert dumped['has_pending_steps'] is False
    log('  PASS: pool engine guard=False, no pending_steps / record_trajectory', buf)


def check_pool_cache_logits_parity(buf: StringIO):
    log('(12) pool engine resident cache logits parity (old load vs cache)', buf)
    from opponent_pool import OpponentPool
    from ppo_pool_engine import PPOOpponentPoolEngine

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    version = 4
    conv_channels = 192
    num_blocks = 40

    with tempfile.TemporaryDirectory(prefix='ppo_p1_pool_cache_') as tmp:
        ckpt_dir = Path(tmp)
        mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        ac = ActorCritic(version=version, tau=1.0).eval()
        ckpts: list[Path] = []
        for step in (0, 1):
            if step == 1:
                with torch.no_grad():
                    for p in mortal.parameters():
                        p.add_(torch.randn_like(p) * 0.01)
                    for p in ac.parameters():
                        p.add_(torch.randn_like(p) * 0.01)
            path = ckpt_dir / f'step_{step:06d}.pth'
            torch.save({
                'mortal': mortal.state_dict(),
                'actor_critic': ac.state_dict(),
                'steps': step,
            }, path)
            ckpts.append(path)

        pool = OpponentPool(ckpt_dir, past_k=5, latest_prob=0.5, fallback_checkpoint=ckpts[0])
        engine = PPOOpponentPoolEngine(
            Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval(),
            ActorCritic(version=version, tau=1.0).eval(),
            pool,
            version=version,
            device=device,
            enable_amp=False,
            eval_mode=True,
        )

        c, w = obs_shape(version)
        n = 8
        obs_t = torch.randn(n, c, w, device=device)
        masks_t = torch.zeros(n, ACTION_SPACE, dtype=torch.bool, device=device)
        for i in range(n):
            legal = torch.randperm(ACTION_SPACE, device=device)[:8]
            masks_t[i, legal] = True

        reload_brain = Brain(
            version=version, conv_channels=conv_channels, num_blocks=num_blocks,
        ).eval().to(device)
        reload_ac = ActorCritic(version=version, tau=1.0).eval().to(device)

        for ckpt in ckpts:
            with torch.inference_mode():
                pool.load_ppo(ckpt, reload_brain, reload_ac, map_location=device)
                phi_old = reload_brain(obs_t)
                logits_old, _ = reload_ac(phi_old, masks_t)

                brain, ac_cached = engine._get_model(ckpt)
                phi_new = brain(obs_t)
                logits_new, _ = ac_cached(phi_new, masks_t)

            assert torch.allclose(logits_old.float(), logits_new.float(), atol=1e-6), (
                f'logits mismatch for {ckpt.name}'
            )
            log(f'  PASS: {ckpt.name} old-load ≡ resident cache (atol=1e-6)', buf)
    log('  PASS: pool cache logits parity for 2 checkpoints', buf)


def check_trajectory_join(ckpt: str, grp_state: str, buf: StringIO, *, seed_count: int = 13):
    """(13) Single-client self-play: every log file must join a trajectory (100%)."""
    import importlib
    import logging
    import os

    log(f'(13) trajectory join rate — single client ~{seed_count * 4}-game self-play', buf)

    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    version = state['config']['control']['version']
    resnet = state['config']['resnet']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with tempfile.TemporaryDirectory(prefix='ppo_p1_join_') as tmp:
        tmp_path = Path(tmp)
        log_dir = tmp_path / 'train_play'
        log_dir.mkdir(parents=True)
        cfg_path = tmp_path / 'config.toml'
        cfg_path.write_text(f"""
[control]
version = {version}

[grp]
state_file = '{grp_state}'

[grp.network]
hidden_size = 64
num_layers = 2

[env]
pts = [35, 5, -15, -25]
alpha = 1.0
gamma_pt = 1.0
beta = 1.0
chip_value = 5.0

[resnet]
conv_channels = {resnet['conv_channels']}
num_blocks = {resnet['num_blocks']}

[ppo]
tau_init = 1.0
""", encoding='utf-8')

        os.environ['MORTAL_CFG'] = str(cfg_path)
        import config as mortal_config
        importlib.reload(mortal_config)
        from client import _finalize_ppo_trajectories
        from libriichi.arena import OneVsThree

        mortal = Brain(version=version, **resnet).eval().to(device)
        mortal.load_state_dict(state['mortal'])
        ac = ActorCritic(version=version, tau=1.0).eval().to(device)
        load_actor_critic_from_dqn_checkpoint(ac, state['current_dqn'], version=version)

        engine = PPOEngine(
            mortal, ac, version=version, device=device,
            enable_amp=False, enable_quick_eval=False, name='trainee',
        )
        assert not engine.enable_rule_based_agari_guard
        assert engine.record_trajectory

        clone_m = Brain(version=version, **resnet).eval().to(device)
        clone_ac = ActorCritic(version=version, tau=1.0).eval().to(device)
        clone_m.load_state_dict(mortal.state_dict())
        clone_ac.load_state_dict(ac.state_dict())
        champion = PPOEngine(
            clone_m, clone_ac, version=version, device=device,
            enable_amp=False, enable_quick_eval=False,
            name='trainee_clone', record_trajectory=False, eval_mode=False,
        )

        env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
        env.py_vs_py(
            challenger=engine,
            champion=champion,
            seed_start=(90000, 0xBEEF),
            seed_count=seed_count,
        )

        file_list = sorted(str(p) for p in log_dir.glob('*.json.gz'))
        assert file_list, 'no json.gz logs produced'

        log_messages = []
        handler = logging.Handler()
        handler.emit = lambda rec: log_messages.append(rec.getMessage())
        root = logging.getLogger()
        old_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        try:
            trajectories = _finalize_ppo_trajectories(
                engine, file_list, param_version=0, client_label='verify13',
            )
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)

        n_games = len(file_list)
        n_joined = len(trajectories)
        n_key_missing = sum(1 for w in log_messages if 'game key missing' in w)
        n_mismatch = sum(1 for w in log_messages if 'step count mismatch' in w)
        n_orphan = sum(1 for w in log_messages if 'trajectory orphan' in w)
        n_loader_delta = sum(1 for w in log_messages if 'loader size delta' in w)

        log(
            f'  games={n_games} joined={n_joined} '
            f'key_missing={n_key_missing} mismatch={n_mismatch} '
            f'orphan={n_orphan} loader_delta={n_loader_delta}',
            buf,
        )
        assert n_joined == n_games, f'join rate {n_joined}/{n_games} != 100%'
        assert n_key_missing == 0, f'key_missing={n_key_missing} (expected 0)'
        assert n_mismatch == 0, f'mismatch={n_mismatch} (expected 0)'
        assert n_orphan == 0, f'orphan={n_orphan} (expected 0)'
        if n_loader_delta:
            log(f'  INFO: loader_delta={n_loader_delta} (non-fatal; runtime monitor only)', buf)
    log('  PASS: trajectory join rate 100%', buf)


def check_checkpoint_load(ckpt: str, buf: StringIO):
    log('(6) legacy checkpoint load after chip head removal', buf)
    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    version = state['config']['control']['version']
    dqn = DQN(version=version)
    missing, unexpected = dqn.load_state_dict(state['current_dqn'], strict=False)
    assert not missing, f'missing keys: {missing}'
    assert set(unexpected) == {'chip_net.weight', 'chip_net.bias'}, unexpected
    ac = ActorCritic(version=version)
    load_ppo_from_mortal_checkpoint(ac, ckpt, map_location='cpu')
    packed = pack_trajectory(TrajectoryBatch(
        obs=torch.zeros(1, *obs_shape(version)),
        action=torch.zeros(1, dtype=torch.int64),
        logp_old=torch.zeros(1),
        mask=torch.ones(1, ACTION_SPACE, dtype=torch.bool),
        reward=torch.zeros(1),
        done=torch.ones(1, dtype=torch.bool),
    ))
    roundtrip = unpack_trajectory(packed, map_location='cpu')
    assert roundtrip.obs.shape[0] == 1
    log('  PASS: strict=False ignores chip_net; ActorCritic loads main heads', buf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=DEFAULT_CKPT)
    parser.add_argument('--grp-state', default='/home/gamba/mahjong/runs/grp.pth')
    parser.add_argument('--online-log', default=DEFAULT_ONLINE_LOG)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    version = state['config']['control']['version']

    buf = StringIO()
    now = datetime.now(timezone.utc).astimezone()
    log(f'PPO P1 sanity verification (re-run {now.isoformat(timespec="seconds")})', buf)
    log(f'checkpoint: {args.checkpoint}', buf)
    log(f'online log: {args.online_log}', buf)
    log('', buf)

    check_pi0_match(args.checkpoint, buf)
    check_mask(buf)
    check_gae(buf)
    check_logp_old(args.checkpoint, buf)
    check_reward_compose(buf, args.grp_state)
    check_checkpoint_load(args.checkpoint, buf)
    check_sampling(args.checkpoint, buf)
    check_online_chips(buf, args.online_log, version)
    check_dqn_one_vs_three(args.checkpoint, buf)
    check_train_engine_config(args.checkpoint, buf)
    check_pool_engine_no_trajectory(buf)
    check_pool_cache_logits_parity(buf)
    check_trajectory_join(args.checkpoint, args.grp_state, buf)

    log('', buf)
    log('ALL 13 CHECKS PASSED', buf)
    out_path = ROOT / 'freeparlor' / 'docs' / 'ppo_p1_verify_log.txt'
    out_path.write_text(buf.getvalue(), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
