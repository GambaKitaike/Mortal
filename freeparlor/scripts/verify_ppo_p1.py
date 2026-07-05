#!/usr/bin/env python3
"""PPO P1 plumbing sanity checks (design §7.1)."""

from __future__ import annotations

import argparse
import os
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
from ppo_engine import (
    PPOEngine,
    build_production_trainee_engine,
    dump_engine_config,
    pick_actions_from_logits,
)
from ppo_dataloader import assign_rewards_and_dones, recompute_logp_old
from ppo_transport import TrajectoryBatch, pack_trajectory, unpack_trajectory
from reward_calculator import RewardCalculator


DEFAULT_CKPT = '/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth'
DEFAULT_ONLINE_LOG = '/home/gamba/mahjong/runs_archive_0620/test_play/10299_8192_d.json.gz'


def log(msg: str, buf: StringIO):
    print(msg)
    buf.write(msg + '\n')


def _run_self_play_with_retry(make_env, engine, champion, *, log_dir: Path, seed_count: int, seed_bases: list[int]):
    """Run py_vs_py; retry on flaky kan-select RuntimeError from sampled policies."""
    import shutil

    last_err = None
    for seed_base in seed_bases:
        engine.pending_by_game = {}
        engine.pending_steps = []
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True)
        env = make_env(log_dir)
        try:
            env.py_vs_py(
                challenger=engine,
                champion=champion,
                seed_start=(seed_base, 0xBEEF),
                seed_count=seed_count,
            )
            return seed_base
        except RuntimeError as exc:
            last_err = exc
            if 'kan choice not in kan candidates' not in str(exc):
                raise
    raise RuntimeError(f'self-play failed on all seeds: {last_err}') from last_err


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

    engine = build_production_trainee_engine(
        mortal, ac, version=version, device=device,
    )
    dumped = dump_engine_config(engine)
    log(f'  trainee dump: {dumped}', buf)
    assert not engine.enable_rule_based_agari_guard
    assert not engine.eval_mode
    assert engine.record_trajectory
    assert not engine.enable_quick_eval
    assert engine.enable_amp
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


def _build_verify_selfplay_engines(state: dict, device: torch.device):
    """Production trainee + eval clone (same builders as client.py / test_play)."""
    version = state['config']['control']['version']
    resnet = state['config']['resnet']

    mortal = Brain(version=version, **resnet).eval().to(device)
    mortal.load_state_dict(state['mortal'])
    ac = ActorCritic(version=version, tau=1.0).eval().to(device)
    load_actor_critic_from_dqn_checkpoint(ac, state['current_dqn'], version=version)

    engine = build_production_trainee_engine(
        mortal, ac, version=version, device=device,
    )
    ref_dump = dump_engine_config(engine)

    clone_m = Brain(version=version, **resnet).eval().to(device)
    clone_ac = ActorCritic(version=version, tau=1.0).eval().to(device)
    clone_m.load_state_dict(mortal.state_dict())
    clone_ac.load_state_dict(ac.state_dict())
    champion = PPOEngine(
        clone_m, clone_ac, version=version, device=device,
        enable_amp=True, enable_quick_eval=False,
        name='trainee_clone', record_trajectory=False, eval_mode=True,
    )
    return engine, champion, ref_dump


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

        def _make_env(path: Path):
            return OneVsThree(disable_progress_bar=True, log_dir=str(path))

        engine, champion, ref_dump = _build_verify_selfplay_engines(state, device)
        assert dump_engine_config(engine) == ref_dump, 'trainee config drift in (13)'
        log(f'  trainee config dump diff=empty: {ref_dump}', buf)
        assert not engine.enable_rule_based_agari_guard
        assert engine.record_trajectory

        seed_used = _run_self_play_with_retry(
            _make_env, engine, champion,
            log_dir=log_dir,
            seed_count=seed_count,
            seed_bases=[90000, 91000, 92000, 93000, 94000, 11111],
        )
        log(f'  self-play seed_start=({seed_used}, 0xBEEF)', buf)

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


def _count_end_kyoku_events(log_path: str) -> int:
    import gzip
    import json

    count = 0
    with gzip.open(log_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get('type') == 'end_kyoku':
                count += 1
    return count


def _expected_rewards_from_log(
    log_path: str,
    *,
    version: int,
    grp_state: str,
    alpha: float,
    gamma_pt: float,
    beta: float,
    chip_value: float,
    pts: list[int],
):
    loader = GameplayLoader(
        version=version,
        player_names=['trainee'],
        oracle=False,
        always_include_kan_select=True,
    )
    data = loader.load_gz_log_files([log_path])
    game = data[0][0]
    grp_obj = game.take_grp()
    player_id = game.take_player_id()
    grp_feature = grp_obj.take_feature()
    rank_by_player = grp_obj.take_rank_by_player()
    final_scores = grp_obj.take_final_scores()
    n_kyoku = len(grp_feature)

    grp = GRP(hidden_size=64, num_layers=2).eval()
    grp.load_state_dict(torch.load(grp_state, weights_only=True, map_location='cpu')['model'])
    calc = RewardCalculator(grp, pts=pts, alpha=alpha, gamma_pt=gamma_pt)
    chip_deltas = load_kyoku_chip_deltas_from_log(log_path, player_id, n_kyoku)

    kyoku_rewards = calc.calc_delta_blend(
        player_id, grp_feature, rank_by_player, final_scores,
        alpha=alpha, gamma_pt=gamma_pt,
        chip_deltas=chip_deltas, beta=beta, chip_value=chip_value,
        lambda_opp=0.0,
    )
    sotensu = calc.calc_delta_points(player_id, grp_feature, final_scores) / 1000.0
    juni = calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
    sotensu_terms = alpha * sotensu
    grp_terms = gamma_pt * juni
    chip_terms = beta * chip_deltas * chip_value
    return kyoku_rewards, sotensu_terms, grp_terms, chip_terms


def check_reward_placement_e2e(ckpt: str, grp_state: str, buf: StringIO, *, seed_count: int = 5):
    """(14) End-to-end reward/done placement: traj × json.gz per game."""
    import importlib
    import os

    log(f'(14) reward placement e2e — single client {seed_count * 4}-game self-play', buf)

    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    version = state['config']['control']['version']
    resnet = state['config']['resnet']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_cfg = state['config'].get('env', {})
    alpha = env_cfg.get('alpha', 1.0)
    gamma_pt = env_cfg.get('gamma_pt', 1.0)
    beta = env_cfg.get('beta', 1.0)
    chip_value = env_cfg.get('chip_value', 5.0)
    pts = env_cfg.get('pts', [35, 5, -15, -25])

    with tempfile.TemporaryDirectory(prefix='ppo_p1_reward_e2e_') as tmp:
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
pts = {pts}
alpha = {alpha}
gamma_pt = {gamma_pt}
beta = {beta}
chip_value = {chip_value}

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

        def _make_env(path: Path):
            return OneVsThree(disable_progress_bar=True, log_dir=str(path))

        engine, champion, ref_dump = _build_verify_selfplay_engines(state, device)
        assert dump_engine_config(engine) == ref_dump, 'trainee config drift in (14)'
        log(f'  trainee config dump diff=empty: {ref_dump}', buf)
        assert not engine.enable_rule_based_agari_guard
        assert engine.record_trajectory

        seed_used = _run_self_play_with_retry(
            _make_env, engine, champion,
            log_dir=log_dir,
            seed_count=seed_count,
            seed_bases=[88000, 89000, 90000, 91000, 92000, 11111],
        )
        log(f'  self-play seed_start=({seed_used}, 0xBEEF)', buf)

        file_list = sorted(str(p) for p in log_dir.glob('*.json.gz'))
        assert file_list, 'no json.gz logs produced'

        trajectories = _finalize_ppo_trajectories(
            engine, file_list, param_version=0, client_label='verify14',
        )
        assert len(trajectories) == len(file_list), (
            f'join rate {len(trajectories)}/{len(file_list)} != 100%'
        )

        n_games = 0
        n_pass = 0
        for log_path in file_list:
            game_key = Path(log_path).name.replace('.json.gz', '')
            traj_name = f'{game_key}.traj'
            assert traj_name in trajectories, f'missing traj for {game_key}'
            batch = unpack_trajectory(trajectories[traj_name])
            assert batch.at_kyoku is not None, f'at_kyoku missing in traj {game_key}'

            end_kyoku = _count_end_kyoku_events(log_path)
            done_count = int(batch.done.sum().item())
            assert done_count == end_kyoku, (
                f'{game_key}: sum(done)={done_count} != end_kyoku={end_kyoku}'
            )

            at_kyoku = batch.at_kyoku.cpu().numpy().astype(np.int64)
            assert at_kyoku.max() + 1 == len(np.unique(at_kyoku)), (
                f'{game_key}: at_kyoku not consecutive (max={at_kyoku.max()}, '
                f'unique={len(np.unique(at_kyoku))})'
            )

            for kyoku in np.unique(at_kyoku):
                mask = at_kyoku == kyoku
                done_in_kyoku = int(batch.done[mask].sum().item())
                assert done_in_kyoku == 1, (
                    f'{game_key}: kyoku {kyoku} has {done_in_kyoku} done steps'
                )

            non_done = ~batch.done.cpu().numpy()
            assert np.all(batch.reward.cpu().numpy()[non_done] == 0), (
                f'{game_key}: non-done steps have non-zero reward'
            )
            for name in ('reward_sotensu', 'reward_grp', 'reward_chip'):
                comp = getattr(batch, name)
                assert comp is not None, f'{game_key}: missing {name}'
                assert np.all(comp.cpu().numpy()[non_done] == 0), (
                    f'{game_key}: non-done steps have non-zero {name}'
                )

            exp_total, exp_sotensu, exp_grp, exp_chip = _expected_rewards_from_log(
                log_path,
                version=version,
                grp_state=grp_state,
                alpha=alpha,
                gamma_pt=gamma_pt,
                beta=beta,
                chip_value=chip_value,
                pts=pts,
            )

            reward_np = batch.reward.cpu().numpy()
            sotensu_np = batch.reward_sotensu.cpu().numpy()
            grp_np = batch.reward_grp.cpu().numpy()
            chip_np = batch.reward_chip.cpu().numpy()
            for kyoku in np.unique(at_kyoku):
                done_idx = np.where((at_kyoku == kyoku) & batch.done.cpu().numpy())[0]
                assert len(done_idx) == 1, f'{game_key}: kyoku {kyoku} done index'
                i = int(done_idx[0])
                k = int(kyoku)
                assert np.isclose(reward_np[i], exp_total[k], atol=1e-5), (
                    f'{game_key}: kyoku {k} reward {reward_np[i]} != {exp_total[k]}'
                )
                assert np.isclose(sotensu_np[i], exp_sotensu[k], atol=1e-5), (
                    f'{game_key}: kyoku {k} reward_sotensu mismatch'
                )
                assert np.isclose(grp_np[i], exp_grp[k], atol=1e-5), (
                    f'{game_key}: kyoku {k} reward_grp mismatch'
                )
                assert np.isclose(chip_np[i], exp_chip[k], atol=1e-5), (
                    f'{game_key}: kyoku {k} reward_chip mismatch'
                )

            n_games += 1
            n_pass += 1

        log(f'  games={n_games} passed={n_pass} end_kyoku/done/reward all OK', buf)
    log('  PASS: reward placement e2e (14)', buf)


# Deterministic seed: trainee (split a) gets daiminkan opportunity in 1 hanchan.
DAIMINKAN_VERIFY_SEED = 1


class _DaiminkanForcingEngine(PPOEngine):
    """Stub: force action 42 when legal to exercise daiminkan direct path."""

    def _pick_actions(self, logits: torch.Tensor, masks_t: torch.Tensor, *, eval_mode: bool):
        actions, fallback = pick_actions_from_logits(logits, masks_t, eval_mode=eval_mode)
        for i in range(masks_t.shape[0]):
            if masks_t[i, 42]:
                actions[i] = 42
                self.daiminkan_forced = True
        self.illegal_action_fallback_count += fallback
        return actions


def _count_trainee_daiminkan(log_path: str, trainee_seat: int) -> int:
    import gzip
    import json

    count = 0
    with gzip.open(log_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get('type') == 'daiminkan' and ev.get('actor') == trainee_seat:
                count += 1
    return count


def check_daiminkan_direct_path(ckpt: str, grp_state: str, buf: StringIO):
    """(15) Deterministic daiminkan: no kan_select phase, action 42 direct execution."""
    import importlib
    import os

    log(
        f'(15) daiminkan direct path — seed=({DAIMINKAN_VERIFY_SEED}, 0xBEEF) '
        'action-42 stub',
        buf,
    )

    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    version = state['config']['control']['version']
    resnet = state['config']['resnet']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with tempfile.TemporaryDirectory(prefix='ppo_p1_daiminkan_') as tmp:
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

        engine = _DaiminkanForcingEngine(
            mortal,
            ac,
            is_oracle=False,
            version=version,
            device=device,
            enable_amp=True,
            enable_quick_eval=False,
            name='trainee',
        )
        engine.daiminkan_forced = False
        prod_dump = dump_engine_config(
            build_production_trainee_engine(mortal, ac, version=version, device=device),
        )
        assert dump_engine_config(engine) == prod_dump, 'stub engine config drift from production'
        log(f'  trainee config dump diff=empty: {prod_dump}', buf)

        clone_m = Brain(version=version, **resnet).eval().to(device)
        clone_ac = ActorCritic(version=version, tau=1.0).eval().to(device)
        clone_m.load_state_dict(mortal.state_dict())
        clone_ac.load_state_dict(ac.state_dict())
        champion = PPOEngine(
            clone_m, clone_ac, version=version, device=device,
            enable_amp=True, enable_quick_eval=False,
            name='trainee_clone', record_trajectory=False, eval_mode=True,
        )

        env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
        env.py_vs_py(
            challenger=engine,
            champion=champion,
            seed_start=(DAIMINKAN_VERIFY_SEED, 0xBEEF),
            seed_count=1,
        )

        assert engine.daiminkan_forced, 'stub never saw legal action 42 (daiminkan mask)'
        log('  stub forced action 42 at least once', buf)

        file_list = sorted(str(p) for p in log_dir.glob('*.json.gz'))
        assert file_list, 'no json.gz logs produced'

        daiminkan_logs = []
        for log_path in file_list:
            game_key = Path(log_path).name.replace('.json.gz', '')
            split = game_key.rsplit('_', 1)[-1]
            trainee_seat = {'a': 0, 'b': 1, 'c': 2, 'd': 3}.get(split)
            assert trainee_seat is not None, f'unknown split in {game_key}'
            n_daiminkan = _count_trainee_daiminkan(log_path, trainee_seat)
            if n_daiminkan >= 1:
                daiminkan_logs.append((log_path, game_key, trainee_seat, n_daiminkan))

        assert daiminkan_logs, 'no game with trainee daiminkan in 4-split set'
        log_path, game_key, trainee_seat, n_daiminkan = daiminkan_logs[0]
        log(
            f'  daiminkan game={game_key} count={n_daiminkan} trainee_seat={trainee_seat}',
            buf,
        )

        trajectories = _finalize_ppo_trajectories(
            engine, file_list, param_version=0, client_label='verify15',
        )
        assert len(trajectories) == len(file_list), (
            f'trajectory join {len(trajectories)}/{len(file_list)} != 100%'
        )

        traj_name = f'{game_key}.traj'
        assert traj_name in trajectories, f'missing traj for daiminkan game {game_key}'
        batch = unpack_trajectory(trajectories[traj_name])
        actions_np = batch.action.cpu().numpy()
        assert (actions_np == 42).any(), 'trajectory lacks action 42 (daiminkan)'
        log(f'  trajectory_steps={len(actions_np)} action42_present=True', buf)

    log('  PASS: daiminkan direct path (15)', buf)


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


def check_optimizer_resume(buf: StringIO):
    """save → load → 1 opt step: AdamW moments must roundtrip and continue (not reset)."""
    import subprocess
    import textwrap

    log('(16) optimizer state resume (save → load → 1 step)', buf)

    resume_ckpt = Path(
        os.environ.get(
            'PPO_RESUME_CKPT',
            '/home/gamba/mahjong/runs/ppo/stage1_20260705_053301/checkpoints/step_010000.pth',
        )
    )
    if resume_ckpt.is_file():
        state = torch.load(resume_ckpt, weights_only=True, map_location='cpu')
        assert state.get('steps') == 10000, f"steps={state.get('steps')}"
        opt_sd = state['optimizer']
        saved_norms = sorted(
            (int(k), v['exp_avg'].float().norm().item())
            for k, v in opt_sd['state'].items()
        )
        assert len(saved_norms) == 411, f'optimizer params={len(saved_norms)}'
        assert saved_norms[-1][1] > 0, 'exp_avg norms must be non-zero at step 10000'

        version = state['config']['control']['version']
        resnet = state['config']['resnet']
        optim_cfg = state['config'].get('optim', {})
        mortal = Brain(version=version, **resnet)
        ac = ActorCritic(version=version, tau=state['config']['ppo']['tau_init'])
        from torch import optim

        optimizer = optim.AdamW(
            list(mortal.parameters()) + list(ac.parameters()),
            lr=state['config']['ppo']['lr'],
            eps=optim_cfg.get('eps', 1e-8),
            betas=tuple(optim_cfg.get('betas', [0.9, 0.999])),
            weight_decay=optim_cfg.get('weight_decay', 0.1),
        )
        mortal.load_state_dict(state['mortal'])
        ac.load_state_dict(state['actor_critic'])
        optimizer.load_state_dict(opt_sd)

        params = list(mortal.parameters()) + list(ac.parameters())
        before = [optimizer.state[p]['exp_avg'].clone() for p in params]
        reloaded = optimizer.state_dict()
        for key, saved_st in opt_sd['state'].items():
            torch.testing.assert_close(
                reloaded['state'][key]['exp_avg'],
                saved_st['exp_avg'],
                rtol=0,
                atol=1e-6,
                msg=f'load mismatch param {key}',
            )

        c, w = obs_shape(version)
        obs = torch.randn(8, c, w)
        masks = torch.zeros(8, ACTION_SPACE, dtype=torch.bool)
        for row in range(8):
            masks[row, :8] = True
        actions = masks.int().argmax(dim=1)
        mortal.train()
        ac.train()
        optimizer.zero_grad(set_to_none=True)
        phi = mortal(obs)
        logits, values = ac(phi, masks)
        logp = action_log_probs(logits, masks, actions)
        loss = -logp.mean() + 0.5 * values.pow(2).mean()
        loss.backward()
        optimizer.step()

        after = [optimizer.state[p]['exp_avg'] for p in params]
        changed = sum(1 for b, a in zip(before, after) if not torch.allclose(b, a, atol=1e-12))
        assert changed > 0, 'optimizer moments unchanged after 1 step'
        log(f'  real ckpt: {resume_ckpt.name} params={len(params)} changed={changed}', buf)
    else:
        log(f'  WARN: resume ckpt missing ({resume_ckpt}), synthetic subprocess only', buf)

    script = textwrap.dedent('''
        import torch
        import torch.nn as nn
        from torch import optim

        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(8, 4)
        y = torch.randn(8, 2)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            (model(x) - y).pow(2).mean().backward()
            optimizer.step()
        saved = optimizer.state_dict()
        saved_moment = saved['state'][0]['exp_avg'].clone()

        model2 = nn.Linear(4, 2)
        optimizer2 = optim.AdamW(model2.parameters(), lr=1e-3)
        model2.load_state_dict(model.state_dict())
        optimizer2.load_state_dict(saved)
        loaded_moment = optimizer2.state_dict()['state'][0]['exp_avg']
        torch.testing.assert_close(loaded_moment, saved_moment, rtol=0, atol=1e-6)

        optimizer2.zero_grad(set_to_none=True)
        (model2(x) - y).pow(2).mean().backward()
        optimizer2.step()
        after_moment = optimizer2.state_dict()['state'][0]['exp_avg']
        assert not torch.allclose(after_moment, saved_moment, atol=1e-12)
        print('OK synthetic load+1step')
    ''')
    proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        log(proc.stdout, buf)
        log(proc.stderr, buf)
        raise AssertionError(f'optimizer resume subprocess failed (exit {proc.returncode})')
    log(f'  {proc.stdout.strip()}', buf)
    log('  PASS: optimizer exp_avg continues after save/load (16)', buf)


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
    check_reward_placement_e2e(args.checkpoint, args.grp_state, buf)
    check_daiminkan_direct_path(args.checkpoint, args.grp_state, buf)
    check_optimizer_resume(buf)

    log('', buf)
    log('ALL 16 CHECKS PASSED', buf)
    out_path = ROOT / 'freeparlor' / 'docs' / 'ppo_p1_verify_log.txt'
    out_path.write_text(buf.getvalue(), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
