import toml
import os

config_file = os.environ.get('MORTAL_CFG', 'config.toml')
with open(config_file, encoding='utf-8') as f:
    config = toml.load(f)

_env_defaults = {
    'lambda_opp': 0.0,
    'noten_factor': 0.0,
}
config.setdefault('env', {}).update({k: v for k, v in _env_defaults.items() if k not in config['env']})
