import prelude

import logging
import os
import socket
import torch
import numpy as np
import time
import gc
from os import path
from model import Brain, DQN, ActorCritic, load_actor_critic_from_dqn_checkpoint
from player import TrainPlayer
from common import send_msg, recv_msg
from config import config

def _ppo_enabled():
    return config.get('ppo', {}).get('enabled', False)


def _load_heads(rsp, version, device):
    if _ppo_enabled():
        ac = ActorCritic(version=version, tau=config['ppo']['tau_init']).to(device)
        if 'actor_critic' in rsp:
            ac.load_state_dict(rsp['actor_critic'])
        elif 'dqn' in rsp:
            load_actor_critic_from_dqn_checkpoint(ac, rsp['dqn'], version=version)
        return ac
    dqn = DQN(version=version).to(device)
    dqn.load_state_dict(rsp['dqn'], strict=False)
    return dqn


def _log_file_game_key(file_path: str) -> str:
    base = path.basename(file_path)
    if base.endswith('.json.gz'):
        return base[:-len('.json.gz')]
    if base.endswith('.mjson'):
        return base[:-len('.mjson')]
    return path.splitext(base)[0]


def _parse_game_key_meta(game_key: str) -> dict:
    """Parse `{seed}_{key}_{split}` from log filename / game_key."""
    parts = game_key.rsplit('_', 2)
    if len(parts) == 3:
        seed, key, split = parts
        seat = {'a': 0, 'b': 1, 'c': 2, 'd': 3}.get(split)
        return {'seed': seed, 'key': key, 'split': split, 'trainee_seat': seat}
    return {'seed': None, 'key': None, 'split': None, 'trainee_seat': None}


def _pending_key_status(pending_by_game: dict | None, game_key: str) -> tuple[bool, int]:
    if pending_by_game is None:
        return False, 0
    if game_key in pending_by_game:
        return True, len(pending_by_game[game_key])
    partial = sum(
        1 for k in pending_by_game
        if k.startswith(game_key) or game_key.startswith(k)
    )
    return False, partial


def _finalize_ppo_trajectories(engine, file_list, param_version, *, client_label=''):
    from libriichi.dataset import GameplayLoader
    from chip_from_log import load_kyoku_chip_deltas_from_log
    from model import GRP
    from reward_calculator import RewardCalculator
    from ppo_dataloader import assign_rewards_and_dones, assign_kyoku_terms
    from ppo_transport import numpy_trajectory_to_batch, pack_trajectory

    pending = engine.drain_pending()
    pending_by_game = pending if isinstance(pending, dict) else None
    pending_flat = pending if isinstance(pending, list) else []
    if not pending_by_game and not pending_flat:
        return {}

    grp = GRP(**config['grp']['network'])
    grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location='cpu')
    grp.load_state_dict(grp_state['model'])
    reward_calc = RewardCalculator(
        grp, config['env']['pts'],
        alpha=config['env'].get('alpha', 1.0),
        gamma_pt=config['env'].get('gamma_pt', 1.0),
    )
    beta = config['env'].get('beta', 1.0)
    chip_value = config['env'].get('chip_value', 5.0)

    loader = GameplayLoader(
        version=config['control']['version'],
        player_names=['trainee'],
        oracle=False,
        always_include_kan_select=True,
    )
    trajectories = {}
    cursor = 0
    for file_path in sorted(file_list):
        game_key = _log_file_game_key(file_path)
        data = loader.load_gz_log_files([file_path])
        for game in data[0]:
            loader_game_size = len(game.take_obs())
            if pending_by_game is not None:
                steps = pending_by_game.pop(game_key, None)
                if steps is None:
                    had_key, partial = _pending_key_status(pending_by_game, game_key)
                    meta = _parse_game_key_meta(game_key)
                    logging.warning(
                        'trajectory game key missing, skipping game '
                        'client=%s game_key=%s expected_game_size=%s actual_steps=0 '
                        'pending_had_key=%s pending_partial_match=%s '
                        'seed=%s split=%s trainee_seat=%s file=%s',
                        client_label, game_key, loader_game_size,
                        had_key, partial,
                        meta['seed'], meta['split'], meta['trainee_seat'], file_path,
                    )
                    continue
            else:
                steps = pending_flat[cursor:cursor + loader_game_size]
                cursor += loader_game_size

            game_size = len(steps)
            at_kyoku = [int(s['at_kyoku']) for s in steps]
            loader_delta = game_size - loader_game_size
            if loader_delta != 0:
                meta = _parse_game_key_meta(game_key)
                logging.info(
                    'trajectory loader size delta=%s client=%s game_key=%s '
                    'loader_game_size=%s recorded_steps=%s '
                    'seed=%s split=%s trainee_seat=%s file=%s',
                    loader_delta, client_label, game_key,
                    loader_game_size, game_size,
                    meta['seed'], meta['split'], meta['trainee_seat'], file_path,
                )

            grp_obj = game.take_grp()
            player_id = game.take_player_id()
            grp_feature = grp_obj.take_feature()
            rank_by_player = grp_obj.take_rank_by_player()
            final_scores = grp_obj.take_final_scores()

            try:
                chip_deltas = load_kyoku_chip_deltas_from_log(
                    file_path, player_id, len(grp_feature),
                )
            except RuntimeError as exc:
                logging.error('online chip resolution failed for %s: %s', file_path, exc)
                raise

            kyoku_rewards = reward_calc.calc_delta_blend(
                player_id, grp_feature, rank_by_player, final_scores,
                alpha=reward_calc.alpha, gamma_pt=reward_calc.gamma_pt,
                chip_deltas=chip_deltas, beta=beta, chip_value=chip_value,
                lambda_opp=0.0,
            )
            rank_prob = reward_calc.calc_rank_prob(player_id, grp_feature, rank_by_player)
            grp_pred_rank = int(rank_prob[-2].argmax())
            grp_actual_rank = int(rank_by_player[player_id])
            sotensu = reward_calc.calc_delta_points(player_id, grp_feature, final_scores) / 1000.0
            juni = reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
            sotensu_terms = reward_calc.alpha * sotensu
            grp_terms = reward_calc.gamma_pt * juni
            chip_terms = beta * chip_deltas * chip_value

            rewards, dones = assign_rewards_and_dones(at_kyoku, kyoku_rewards, game_size)
            reward_sotensu = assign_kyoku_terms(at_kyoku, sotensu_terms, game_size)
            reward_grp = assign_kyoku_terms(at_kyoku, grp_terms, game_size)
            reward_chip = assign_kyoku_terms(at_kyoku, chip_terms, game_size)
            for i in range(game_size):
                steps[i]['reward'] = float(rewards[i])
                steps[i]['done'] = bool(dones[i])
                steps[i]['reward_sotensu'] = float(reward_sotensu[i])
                steps[i]['reward_grp'] = float(reward_grp[i])
                steps[i]['reward_chip'] = float(reward_chip[i])
                steps[i]['grp_pred_rank'] = float(grp_pred_rank)
                steps[i]['grp_actual_rank'] = float(grp_actual_rank)

            batch = numpy_trajectory_to_batch(steps, param_version=param_version)
            traj_name = path.basename(file_path).replace('.json.gz', '.traj').replace('.mjson', '.traj')
            trajectories[traj_name] = pack_trajectory(batch)

    if pending_by_game:
        for game_key, orphan_steps in pending_by_game.items():
            logging.warning(
                'trajectory orphan steps (%s steps) for game_id=%s client=%s',
                len(orphan_steps), game_key, client_label,
            )

    return trajectories


def main():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']
    use_ppo = _ppo_enabled()
    client_label = os.environ.get('TRAIN_PLAY_PROFILE', 'default')

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).to(device).eval()
    head = None
    if not use_ppo:
        head = DQN(version=version).to(device)

    if config['online']['enable_compile']:
        mortal.compile()
        if head is not None:
            head.compile()

    train_player = TrainPlayer()
    param_version = -1

    pts = np.array([90, 45, 0, -135])
    history_window = config['online']['history_window']
    history = []

    while True:
        while True:
            with socket.socket() as conn:
                conn.connect(remote)
                msg = {
                    'type': 'get_param',
                    'param_version': param_version,
                }
                send_msg(conn, msg)
                rsp = recv_msg(conn, map_location=device)
                if rsp['status'] == 'ok':
                    param_version = rsp['param_version']
                    break
                time.sleep(3)
        mortal.load_state_dict(rsp['mortal'])
        head = _load_heads(rsp, version, device)
        beta_sel = rsp.get('beta_sel', 0.0)
        logging.info(f'param has been updated (beta_sel={beta_sel}, ppo={use_ppo})')

        if use_ppo:
            from ppo_engine import build_production_trainee_engine, dump_engine_config
            engine = build_production_trainee_engine(
                mortal,
                head,
                version=version,
                device=device,
                p_enrich=config['ppo'].get('p_enrich', 0.0),
            )
            trainee_cfg = dump_engine_config(engine)
            logging.info(f'trainee engine config dump: {trainee_cfg}')
            assert not engine.enable_rule_based_agari_guard, 'train rollout: guard must be OFF'
            assert not engine.eval_mode, 'train rollout: eval_mode must be False'
            assert engine.record_trajectory, 'train rollout: record_trajectory must be enabled'
            rankings, file_list = train_player.train_play_ppo(engine, device)
            fb = engine.illegal_action_fallback_count
            if fb:
                logging.warning(f'illegal_action_fallback_count={fb} (expected 0)')
            else:
                logging.info(f'illegal_action_fallback_count={fb}')
            logs = _finalize_ppo_trajectories(
                engine, file_list, param_version, client_label=client_label,
            )
        else:
            rankings, file_list = train_player.train_play(mortal, head, device, beta_sel=beta_sel)
            logs = {}
            for filename in file_list:
                with open(filename, 'rb') as f:
                    logs[path.basename(filename)] = f.read()

        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ pts / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = sum_rankings @ np.arange(1, 5) / sum_rankings.sum()
        ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()

        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {
                'type': 'submit_replay',
                'logs': logs,
                'param_version': param_version,
            })
            logging.info('logs have been submitted')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
