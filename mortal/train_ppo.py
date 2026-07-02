def train_ppo():
    import prelude

    import gc
    import logging
    import os
    import shutil
    import sys
    from datetime import datetime
    from pathlib import Path

    import torch
    from torch import optim
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.tensorboard import SummaryWriter

    from common import drain, parameter_count, submit_param, tqdm
    from config import config
    from model import ActorCritic, Brain, load_ppo_from_mortal_checkpoint
    from player import TestPlayer
    from ppo import action_log_probs, compute_gae, ppo_loss
    from ppo_dataloader import collate_trajectory_batches, load_trajectory_file
    from ppo_transport import TrajectoryBatch

    ppo_cfg = config['ppo']
    version = config['control']['version']
    device = torch.device(config['control']['device'])
    enable_amp = config['control']['enable_amp']
    max_grad_norm = config['optim']['max_grad_norm']
    online = config['control']['online']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    max_steps = ppo_cfg.get('max_steps') or 0
    state_file = config['control']['state_file']
    eps_clip = ppo_cfg['eps_clip']

    mortal = Brain(version=version, **config['resnet']).to(device)
    actor_critic = ActorCritic(
        version=version,
        tau=ppo_cfg['tau_init'],
    ).to(device)
    mortal.eval()
    actor_critic.eval()

    optimizer = optim.AdamW(
        list(mortal.parameters()) + list(actor_critic.parameters()),
        lr=ppo_cfg.get('lr', 3e-4),
        eps=config['optim']['eps'],
        betas=config['optim']['betas'],
        weight_decay=config['optim']['weight_decay'],
    )
    scaler = GradScaler(device.type, enabled=enable_amp)
    steps = 0

    if os.path.isfile(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        mortal.load_state_dict(state['mortal'])
        actor_critic.load_state_dict(state['actor_critic'])
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        steps = state['steps']
        logging.info(f'loaded checkpoint: steps={steps:,}')
    else:
        init_ckpt = ppo_cfg.get('init_checkpoint')
        if init_ckpt:
            load_ppo_from_mortal_checkpoint(actor_critic, init_ckpt, map_location=device)
            mortal_state = torch.load(init_ckpt, weights_only=True, map_location=device)['mortal']
            mortal.load_state_dict(mortal_state)
            logging.info(f'initialized from {init_ckpt}')

    logging.info(f'PPO mortal params: {parameter_count(mortal):,}')
    logging.info(f'PPO actor_critic params: {parameter_count(actor_critic):,}')

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    test_player = TestPlayer()
    stats = {
        'total': 0.0,
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'entropy': 0.0,
        'clip_fraction': 0.0,
        'explained_variance': 0.0,
    }
    stat_count = 0

    if online:
        submit_param(mortal, actor_critic, is_idle=True, beta_sel=0.0, use_ppo=True)
        logging.info('param has been submitted')

    def save_checkpoint():
        state = {
            'mortal': mortal.state_dict(),
            'actor_critic': actor_critic.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': steps,
            'timestamp': datetime.now().timestamp(),
            'config': config,
        }
        torch.save(state, state_file)

    def flush_stats():
        nonlocal stat_count
        if stat_count == 0:
            return
        n = stat_count
        writer.add_scalar('loss/total', stats['total'] / n, steps)
        writer.add_scalar('loss/policy_loss', stats['policy_loss'] / n, steps)
        writer.add_scalar('loss/value_loss', stats['value_loss'] / n, steps)
        writer.add_scalar('loss/entropy', stats['entropy'] / n, steps)
        writer.add_scalar('ppo/clip_fraction', stats['clip_fraction'] / n, steps)
        writer.add_scalar('ppo/explained_variance', stats['explained_variance'] / n, steps)
        writer.add_scalar('hparam/lr', optimizer.param_groups[0]['lr'], steps)
        writer.flush()
        for k in stats:
            stats[k] = 0.0
        stat_count = 0

    def run_test_play():
        stat = test_player.test_play_ppo(
            test_games // 4,
            mortal,
            actor_critic,
            device,
        )
        mortal.eval()
        actor_critic.eval()
        avg_pt = stat.avg_pt([90, 45, 0, -135])
        logging.info(f'avg rank: {stat.avg_rank:.6}')
        logging.info(f'avg pt: {avg_pt:.6}')
        logging.info(
            f'test_play behavior: agari={stat.agari_rate * 100:.2f}% '
            f'houjuu={stat.houjuu_rate * 100:.2f}% '
            f'fuuro={stat.fuuro_rate * 100:.2f}% '
            f'riichi={stat.riichi_rate * 100:.2f}%'
        )
        writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
        writer.add_scalar('test_play/avg_pt', avg_pt, steps)
        writer.add_scalar('test_play/behavior/houjuu', stat.houjuu_rate, steps)
        writer.flush()
        return stat

    def train_on_trajectories(traj: TrajectoryBatch):
        nonlocal steps
        nonlocal stat_count

        obs = traj.obs.to(device=device, dtype=torch.float32)
        actions = traj.action.to(device=device)
        masks = traj.mask.to(device=device)
        logp_old = traj.logp_old.to(device=device)
        rewards = traj.reward.to(device=device)
        dones = traj.done.to(device=device)

        with torch.inference_mode():
            phi_all = mortal(obs)
            _, values_all = actor_critic(phi_all, masks)
            values_np = values_all.cpu()
            rewards_np = rewards.cpu()
            dones_np = dones.cpu()

        episode_indices = torch.where(dones)[0].tolist()
        if not episode_indices or episode_indices[-1] != len(dones) - 1:
            raise ValueError('trajectory must be concatenated full kyoku episodes ending with done=True')

        start = 0
        adv_parts = []
        ret_parts = []
        for end in episode_indices:
            sl = slice(start, end + 1)
            v = torch.cat([values_np[sl], torch.zeros(1)])
            adv, ret = compute_gae(
                rewards_np[sl],
                v,
                dones_np[sl],
                gamma=ppo_cfg['gamma_disc'],
                lam=ppo_cfg['gae_lambda'],
            )
            adv_parts.append(adv)
            ret_parts.append(ret)
            start = end + 1

        advantages = torch.cat(adv_parts).to(device)
        returns = torch.cat(ret_parts).to(device)

        with torch.autocast(device.type, enabled=enable_amp):
            phi = mortal(obs)
            logits, values = actor_critic(phi, masks)
            losses = ppo_loss(
                logits,
                values,
                actions,
                masks,
                logp_old,
                advantages,
                returns,
                eps_clip=eps_clip,
                c_vf=ppo_cfg['c_vf'],
                c_ent=ppo_cfg['c_ent'],
                huber_delta=ppo_cfg['huber_delta'],
            )

        for name, val in losses.items():
            if not torch.isfinite(val).all():
                raise FloatingPointError(f'non-finite {name} at step {steps + 1}')

        with torch.inference_mode():
            logp = action_log_probs(logits, masks, actions)
            ratio = (logp - logp_old).exp()
            clip_fraction = ((ratio - 1.0).abs() > eps_clip).float().mean().item()
            ret_var = returns.var(unbiased=False)
            if ret_var > 0:
                explained_var = (1 - (returns - values).var(unbiased=False) / ret_var).item()
            else:
                explained_var = float('nan')

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses['total']).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(
                list(mortal.parameters()) + list(actor_critic.parameters()),
                max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        steps += 1

        stats['total'] += losses['total'].item()
        stats['policy_loss'] += losses['policy_loss'].item()
        stats['value_loss'] += losses['value_loss'].item()
        stats['entropy'] += losses['entropy'].item()
        stats['clip_fraction'] += clip_fraction
        stats['explained_variance'] += explained_var
        stat_count += 1

        logging.info(
            f'ppo step {steps}: total={losses["total"].item():.4f} '
            f'pi={losses["policy_loss"].item():.4f} vf={losses["value_loss"].item():.4f} '
            f'H={losses["entropy"].item():.4f} clip={clip_fraction:.4f} ev={explained_var:.4f}'
        )

        if online and steps % submit_every == 0:
            submit_param(mortal, actor_critic, is_idle=False, beta_sel=0.0, use_ppo=True)
            logging.info('param has been submitted')

        if steps % save_every == 0:
            flush_stats()
            save_checkpoint()
            before_next_test_play = (test_every - steps % test_every) % test_every
            logging.info(f'total steps: {steps:,} (~{before_next_test_play:,} to test_play)')

            if online and steps % submit_every != 0:
                submit_param(mortal, actor_critic, is_idle=False, beta_sel=0.0, use_ppo=True)
                logging.info('param has been submitted')

            if steps % test_every == 0:
                run_test_play()
                if online:
                    sys.exit(0)

    def train_epoch():
        drain_dir = None
        if online:
            drain_dir = drain()
            traj_files = sorted(
                path
                for path in Path(drain_dir).glob('*.traj')
                if path.is_file()
            )
        else:
            traj_glob = ppo_cfg.get('trajectory_glob')
            traj_files = sorted(Path().glob(traj_glob)) if traj_glob else []

        logging.info(f'trajectory file list size: {len(traj_files):,}')
        if not traj_files:
            logging.warning('empty trajectory list, skipping epoch')
            return drain_dir

        pb = tqdm(total=len(traj_files), desc='PPO TRAIN')
        for traj_path in traj_files:
            if max_steps and steps >= max_steps:
                break
            train_on_trajectories(load_trajectory_file(traj_path, map_location='cpu'))
            pb.update(1)
        pb.close()
        return drain_dir

    while True:
        if max_steps and steps >= max_steps:
            logging.info(f'reached max_steps={max_steps:,}, stopping')
            break
        train_epoch()
        if not online:
            break
        if max_steps and steps >= max_steps:
            break

    if stat_count:
        flush_stats()
    save_checkpoint()
    if online and steps % test_every != 0:
        run_test_play()
    if online and (max_steps and steps >= max_steps or steps % test_every == 0):
        sys.exit(0)


def main():
    import os
    import sys
    import time
    from subprocess import Popen

    from config import config

    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train_ppo()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
