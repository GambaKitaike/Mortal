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
