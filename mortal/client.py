import prelude

import logging
import socket
import torch
import numpy as np
import time
import gc
from os import path
from model import Brain, DQN, ActorCritic, load_actor_critic_from_dqn_checkpoint
from player import TrainPlayer
from ppo_dataloader import assign_rewards_and_dones
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


def _finalize_ppo_trajectories(engine, file_list, param_version):
    from pathlib import Path
    from libriichi.dataset import GameplayLoader
    from model import GRP
    from reward_calculator import RewardCalculator
    from ppo_transport import numpy_trajectory_to_batch, pack_trajectory

    pending = engine.drain_pending()
    if not pending:
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
    chip_dir = config['env'].get('chip_dir', '/home/gamba/mahjong/data/tenhou/chips')

    loader = GameplayLoader(version=config['control']['version'], player_names=['trainee'])
    trajectories = {}
    cursor = 0
    for file_path in file_list:
        data = loader.load_gz_log_files([file_path])
        for game in data[0]:
            game_size = len(game.take_obs())
            steps = pending[cursor:cursor + game_size]
            cursor += game_size
            if len(steps) != game_size:
                logging.warning('trajectory step count mismatch, skipping game')
                continue

            grp_obj = game.take_grp()
            player_id = game.take_player_id()
            at_kyoku = game.take_at_kyoku()
            grp_feature = grp_obj.take_feature()
            rank_by_player = grp_obj.take_rank_by_player()
            final_scores = grp_obj.take_final_scores()

            chip_path = Path(chip_dir) / f'{Path(file_path).name}.npz'
            if chip_path.exists():
                chips = np.load(chip_path)['chips']
                chip_deltas = chips[:len(grp_feature), player_id].astype(np.float64)
            else:
                chip_deltas = np.zeros(len(grp_feature), dtype=np.float64)

            kyoku_rewards = reward_calc.calc_delta_blend(
                player_id, grp_feature, rank_by_player, final_scores,
                alpha=reward_calc.alpha, gamma_pt=reward_calc.gamma_pt,
                chip_deltas=chip_deltas, beta=beta, chip_value=chip_value,
                lambda_opp=0.0,
            )

            rewards, dones = assign_rewards_and_dones(at_kyoku, kyoku_rewards, game_size)
            for i in range(game_size):
                steps[i]['reward'] = float(rewards[i])
                steps[i]['done'] = bool(dones[i])

            batch = numpy_trajectory_to_batch(steps, param_version=param_version)
            traj_name = path.basename(file_path).replace('.json.gz', '.traj').replace('.mjson', '.traj')
            trajectories[traj_name] = pack_trajectory(batch)

    return trajectories


def main():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']
    use_ppo = _ppo_enabled()

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
            profile = config['train_play']['default']
            engine = PPOEngine(
                mortal,
                head,
                is_oracle=False,
                version=version,
                boltzmann_epsilon=profile['boltzmann_epsilon'],
                top_p=profile['top_p'],
                device=device,
                enable_amp=True,
                name='trainee',
            )
            rankings, file_list = train_player.train_play_ppo(engine, device)
            logs = _finalize_ppo_trajectories(engine, file_list, param_version)
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
