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
    ppo_epochs = ppo_cfg.get('ppo_epochs', 1)
    minibatch_size = ppo_cfg.get('minibatch_size', 0)
    diag_log_path = Path(config['control']['state_file']).parent / 'logs' / 'ppo_diag.jsonl'
    diag_log_path.parent.mkdir(parents=True, exist_ok=True)
    trainer_param_version = 0

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
        trainer_param_version = 1
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

    def _append_diag(record: dict):
        import json
        record['trainer_step'] = steps
        with diag_log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def train_on_trajectories(traj: TrajectoryBatch):
        nonlocal steps
        nonlocal stat_count
        nonlocal trainer_param_version

        obs = traj.obs.to(device=device, dtype=torch.float32)
        actions = traj.action.to(device=device)
        masks = traj.mask.to(device=device)
        logp_old = traj.logp_old.to(device=device)
        rewards = traj.reward.to(device=device)
        dones = traj.done.to(device=device)
        param_lag = trainer_param_version - traj.param_version
        _append_diag({
            'event': 'batch_lag',
            'param_version': traj.param_version,
            'trainer_param_version': trainer_param_version,
            'lag': param_lag,
            'batch_size': int(obs.shape[0]),
        })
        logging.info(
            f'ppo batch lag={param_lag} (client pv={traj.param_version}, trainer pv={trainer_param_version})'
        )

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

        n = obs.shape[0]
        mb_size = minibatch_size if minibatch_size and minibatch_size < n else n
        epoch_metrics = []
        explained_var = float('nan')

        for epoch in range(ppo_epochs):
            perm = torch.randperm(n, device=device)
            epoch_clip = []
            epoch_ratio_mean = []
            epoch_ratio_std = []
            epoch_total = epoch_pi = epoch_vf = epoch_ent = 0.0
            mb_count = 0

            for mb_start in range(0, n, mb_size):
                idx = perm[mb_start:mb_start + mb_size]
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_masks = masks[idx]
                mb_logp_old = logp_old[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                with torch.autocast(device.type, enabled=enable_amp):
                    phi = mortal(mb_obs)
                    logits, values = actor_critic(phi, mb_masks)
                    losses = ppo_loss(
                        logits,
                        values,
                        mb_actions,
                        mb_masks,
                        mb_logp_old,
                        mb_adv,
                        mb_ret,
                        eps_clip=eps_clip,
                        c_vf=ppo_cfg['c_vf'],
                        c_ent=ppo_cfg['c_ent'],
                        huber_delta=ppo_cfg['huber_delta'],
                    )

                for name, val in losses.items():
                    if not torch.isfinite(val).all():
                        raise FloatingPointError(f'non-finite {name} at step {steps + 1} epoch {epoch + 1}')

                with torch.inference_mode():
                    logp = action_log_probs(logits, mb_masks, mb_actions)
                    ratio = (logp - mb_logp_old).exp()
                    clip_fraction = ((ratio - 1.0).abs() > eps_clip).float().mean().item()
                    ratio_mean = ratio.mean().item()
                    ratio_std = ratio.std(unbiased=False).item()
                    ret_var = mb_ret.var(unbiased=False)
                    if ret_var > 0:
                        explained_var = (1 - (mb_ret - values).var(unbiased=False) / ret_var).item()

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

                epoch_clip.append(clip_fraction)
                epoch_ratio_mean.append(ratio_mean)
                epoch_ratio_std.append(ratio_std)
                epoch_total += losses['total'].item()
                epoch_pi += losses['policy_loss'].item()
                epoch_vf += losses['value_loss'].item()
                epoch_ent += losses['entropy'].item()
                mb_count += 1

            if mb_count == 0:
                continue

            clip_avg = sum(epoch_clip) / len(epoch_clip)
            ratio_mean_avg = sum(epoch_ratio_mean) / len(epoch_ratio_mean)
            ratio_std_avg = sum(epoch_ratio_std) / len(epoch_ratio_std)
            epoch_metrics.append({
                'epoch': epoch + 1,
                'clip_fraction': clip_avg,
                'ratio_mean': ratio_mean_avg,
                'ratio_std': ratio_std_avg,
            })
            _append_diag({
                'event': 'ppo_epoch',
                'epoch': epoch + 1,
                'clip_fraction': clip_avg,
                'ratio_mean': ratio_mean_avg,
                'ratio_std': ratio_std_avg,
                'param_lag': param_lag,
            })
            logging.info(
                f'ppo step {steps + 1} epoch {epoch + 1}/{ppo_epochs}: '
                f'clip={clip_avg:.4f} ratio={ratio_mean_avg:.4f}±{ratio_std_avg:.4f}'
            )

            if epoch == 0:
                stats['total'] += epoch_total / mb_count
                stats['policy_loss'] += epoch_pi / mb_count
                stats['value_loss'] += epoch_vf / mb_count
                stats['entropy'] += epoch_ent / mb_count
                stats['clip_fraction'] += clip_avg
                stats['explained_variance'] += explained_var
                stat_count += 1

        steps += 1

        if epoch_metrics:
            logging.info(
                f'ppo step {steps}: epoch1_clip={epoch_metrics[0]["clip_fraction"]:.4f} '
                f'epoch{ppo_epochs}_clip={epoch_metrics[-1]["clip_fraction"]:.4f} '
                f'ev={explained_var:.4f}'
            )

        if online and steps % submit_every == 0:
            submit_param(mortal, actor_critic, is_idle=False, beta_sel=0.0, use_ppo=True)
            trainer_param_version += 1
            logging.info('param has been submitted')

        if steps % save_every == 0:
            flush_stats()
            save_checkpoint()
            before_next_test_play = (test_every - steps % test_every) % test_every
            logging.info(f'total steps: {steps:,} (~{before_next_test_play:,} to test_play)')

            if online and steps % submit_every != 0:
                submit_param(mortal, actor_critic, is_idle=False, beta_sel=0.0, use_ppo=True)
                trainer_param_version += 1
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
    inline_test_play = not (max_steps and test_every > max_steps)
    if online and inline_test_play and steps % test_every != 0:
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
