def train_ppo():
    import prelude

    import logging
    from pathlib import Path

    import torch
    from torch import optim
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_

    from common import drain, parameter_count, submit_param
    from config import config
    from model import Brain, ActorCritic, load_ppo_from_mortal_checkpoint
    from ppo import compute_gae, ppo_loss
    from ppo_dataloader import collate_trajectory_batches, load_trajectory_file
    from ppo_transport import TrajectoryBatch

    ppo_cfg = config['ppo']
    version = config['control']['version']
    device = torch.device(config['control']['device'])
    enable_amp = config['control']['enable_amp']
    max_grad_norm = config['optim']['max_grad_norm']

    mortal = Brain(version=version, **config['resnet']).to(device)
    actor_critic = ActorCritic(
        version=version,
        tau=ppo_cfg['tau_init'],
    ).to(device)

    init_ckpt = ppo_cfg.get('init_checkpoint')
    if init_ckpt:
        load_ppo_from_mortal_checkpoint(actor_critic, init_ckpt, map_location=device)
        mortal_state = torch.load(init_ckpt, weights_only=True, map_location=device)['mortal']
        mortal.load_state_dict(mortal_state)

    logging.info(f'PPO mortal params: {parameter_count(mortal):,}')
    logging.info(f'PPO actor_critic params: {parameter_count(actor_critic):,}')

    optimizer = optim.AdamW(
        list(mortal.parameters()) + list(actor_critic.parameters()),
        lr=ppo_cfg.get('lr', 3e-4),
        eps=config['optim']['eps'],
        betas=config['optim']['betas'],
        weight_decay=config['optim']['weight_decay'],
    )
    scaler = GradScaler(device.type, enabled=enable_amp)
    steps = 0

    def train_on_trajectories(traj: TrajectoryBatch):
        nonlocal steps
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
                eps_clip=ppo_cfg['eps_clip'],
                c_vf=ppo_cfg['c_vf'],
                c_ent=ppo_cfg['c_ent'],
                huber_delta=ppo_cfg['huber_delta'],
            )

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
        logging.info(
            f'ppo step {steps}: total={losses["total"].item():.4f} '
            f'pi={losses["policy_loss"].item():.4f} vf={losses["value_loss"].item():.4f} '
            f'H={losses["entropy"].item():.4f}'
        )

    online = config['control']['online']
    if online:
        submit_param(mortal, actor_critic, is_idle=True, beta_sel=0.0, use_ppo=True)
        drain_dir = drain()
        if drain_dir:
            traj_files = sorted(Path(drain_dir).glob('*.traj'))
            batches = [load_trajectory_file(p, map_location='cpu') for p in traj_files]
            if batches:
                train_on_trajectories(collate_trajectory_batches(batches))
    else:
        traj_glob = ppo_cfg.get('trajectory_glob')
        if traj_glob:
            batches = [
                load_trajectory_file(p, map_location='cpu')
                for p in sorted(Path().glob(traj_glob))
            ]
            if batches:
                train_on_trajectories(collate_trajectory_batches(batches))


def main():
    train_ppo()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
