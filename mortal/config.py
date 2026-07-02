import toml
import os

config_file = os.environ.get('MORTAL_CFG', 'config.toml')
with open(config_file, encoding='utf-8') as f:
    config = toml.load(f)

_env_defaults = {
    'lambda_opp': 0.0,
    'noten_factor': 0.0,
    'beta_sel_max': 0.3,
    'beta_sel_warmup_steps': 2000,
    'beta_sel_ramp_steps': 2000,
    'chip_n_step': 3,
    'chip_target_tau': 0.005,
    'chip_weight': 1.0,
}
config.setdefault('env', {}).update({k: v for k, v in _env_defaults.items() if k not in config['env']})
config.setdefault('dataset', {}).setdefault('games_per_batch', 4)
_control_defaults = {
    'dqn_loss': 'mse',
    'huber_delta': 15.0,
}
config.setdefault('control', {}).update({k: v for k, v in _control_defaults.items() if k not in config['control']})
_ppo_defaults = {
    'enabled': False,
    'eps_clip': 0.2,
    'c_vf': 0.5,
    'c_ent': 0.01,
    'gae_lambda': 0.95,
    'gamma_disc': 1.0,
    'tau_init': 1.0,
    'huber_delta': 15.0,
    'lr': 3e-4,
    'init_checkpoint': '',
    'trajectory_glob': '',
    'max_steps': 0,
}
config.setdefault('ppo', {}).update({k: v for k, v in _ppo_defaults.items() if k not in config.get('ppo', {})})
