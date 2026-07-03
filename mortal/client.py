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

    loader = GameplayLoader(version=config['control']['version'], player_names=['trainee'])
    trajectories = {}
    cursor = 0
    for file_path in sorted(file_list):
        game_key = _log_file_game_key(file_path)
        data = loader.load_gz_log_files([file_path])
        for game in data[0]:
            game_size = len(game.take_obs())
            if pending_by_game is not None:
                steps = pending_by_game.pop(game_key, None)
                if steps is None:
                    logging.warning(
                        'trajectory game key missing (%s), skipping game client=%s file=%s',
                        game_key, client_label, file_path,
                    )
                    continue
            else:
                steps = pending_flat[cursor:cursor + game_size]
                cursor += game_size

            if len(steps) != game_size:
                logging.warning(
                    'trajectory step count mismatch (%s/%s), skipping game client=%s file=%s',
                    len(steps), game_size, client_label, file_path,
                )
                continue

            grp_obj = game.take_grp()
            player_id = game.take_player_id()
            at_kyoku = game.take_at_kyoku()
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
            from ppo_engine import PPOEngine
            engine = PPOEngine(
                mortal,
                head,
                is_oracle=False,
                version=version,
                device=device,
                enable_amp=True,
                enable_quick_eval=False,
                enable_rule_based_agari_guard=True,
                name='trainee',
            )
            rankings, file_list = train_player.train_play_ppo(engine, device)
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
