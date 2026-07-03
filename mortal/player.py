import torch
import numpy as np
import os
import shutil
import secrets
import logging
from os import path
from model import Brain, DQN
from engine import MortalEngine
from libriichi.stat import Stat
from libriichi.arena import OneVsThree
from config import config

class TestPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['test']
        device = torch.device(baseline_cfg['device'])

        state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = DQN(version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(state['current_dqn'], strict=False)
        if baseline_cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()

        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle = False,
            version = version,
            device = device,
            enable_amp = True,
            enable_rule_based_agari_guard = True,
            name = 'baseline',
        )
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(config['test_play']['log_dir'])
        self.self_play = config['test_play'].get('self_play', False)
        self._last_chip_realize = None

    def chip_realize_stats(self):
        return self._last_chip_realize

    def _compute_chip_realize(self, log_dir):
        try:
            import sys
            from pathlib import Path
            scripts = Path(__file__).resolve().parents[1] / 'freeparlor' / 'scripts'
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            from analyze_chip_realize import analyze_eval_dir
            agg = analyze_eval_dir(Path(log_dir))
            if agg.aka_held_kyoku == 0:
                return None
            return {
                'chip_realize_rate': agg.chip_realize / agg.aka_held_kyoku,
                'call_win_rate': agg.call_win / agg.aka_held_kyoku,
            }
        except Exception as ex:
            logging.warning(f'chip_realize stats skipped: {ex}')
            return None

    def test_play(self, seed_count, mortal, dqn, device, beta_sel=0.0):
        torch.backends.cudnn.benchmark = False
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            device = device,
            enable_amp = True,
            beta_sel = beta_sel,
            name = 'mortal',
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        champion = engine_chal if self.self_play else self.baseline_engine
        env.py_vs_py(
            challenger = engine_chal,
            champion = champion,
            seed_start = (10000, 0x2000),
            seed_count = seed_count,
        )

        stat = Stat.from_dir(self.log_dir, 'mortal')
        self._last_chip_realize = self._compute_chip_realize(self.log_dir)
        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return stat

    def _make_ppo_eval_engine(self, mortal, actor_critic, device, *, name='mortal'):
        from ppo_engine import PPOEngine
        return PPOEngine(
            mortal,
            actor_critic,
            is_oracle=False,
            version=self.chal_version,
            device=device,
            enable_amp=False,
            enable_quick_eval=False,
            enable_rule_based_agari_guard=True,
            name=name,
            eval_mode=True,
            record_trajectory=False,
        )

    def _clone_ppo_eval_engine(self, mortal, actor_critic, device, *, name='mortal'):
        from model import ActorCritic, Brain
        from ppo_engine import PPOEngine

        clone_m = Brain(version=self.chal_version, **config['resnet']).to(device).eval()
        clone_ac = ActorCritic(
            version=self.chal_version,
            tau=config['ppo']['tau_init'],
        ).to(device).eval()
        clone_m.load_state_dict(mortal.state_dict())
        clone_ac.load_state_dict(actor_critic.state_dict())
        return PPOEngine(
            clone_m,
            clone_ac,
            is_oracle=False,
            version=self.chal_version,
            device=device,
            enable_amp=False,
            enable_quick_eval=False,
            enable_rule_based_agari_guard=True,
            name=name,
            eval_mode=True,
            record_trajectory=False,
        )

    def test_play_ppo(
        self,
        seed_count,
        mortal,
        actor_critic,
        device,
        *,
        seed_start=(10000, 0x2000),
        clear_log_dir=True,
    ):
        torch.backends.cudnn.benchmark = False
        engine_chal = self._make_ppo_eval_engine(mortal, actor_critic, device, name='mortal')

        if clear_log_dir:
            if path.isdir(self.log_dir):
                shutil.rmtree(self.log_dir)
        else:
            os.makedirs(self.log_dir, exist_ok=True)

        env = OneVsThree(
            disable_progress_bar=False,
            log_dir=self.log_dir,
        )
        if self.self_play:
            # challenger/champion は別インスタンス必須（arena 並列が同一 engine を叩く）
            champion = self._clone_ppo_eval_engine(mortal, actor_critic, device, name='mortal_clone')
        else:
            champion = self.baseline_engine
        env.py_vs_py(
            challenger=engine_chal,
            champion=champion,
            seed_start=seed_start,
            seed_count=seed_count,
        )

        stat = Stat.from_dir(self.log_dir, 'mortal')
        self._last_chip_realize = self._compute_chip_realize(self.log_dir)
        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return stat

class TrainPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['train']
        device = torch.device(baseline_cfg['device'])

        state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = DQN(version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(state['current_dqn'], strict=False)
        if baseline_cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()

        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle = False,
            version = version,
            device = device,
            enable_amp = True,
            enable_rule_based_agari_guard = True,
            name = 'baseline',
        )

        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        logging.info(f'using profile {profile}')
        cfg = config['train_play'][profile]
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(cfg['log_dir'])
        self.train_key = secrets.randbits(64)
        self.train_seed = 10000

        self.seed_count = cfg['games'] // 4
        self.boltzmann_epsilon = cfg.get('boltzmann_epsilon', 0.0)
        self.boltzmann_temp = cfg.get('boltzmann_temp', 0.05)
        self.top_p = cfg.get('top_p', 1.0)

        self.repeats = cfg['repeats']
        self.repeat_counter = 0

        pool_cfg = config.get('opponent_pool', {})
        self.opponent_pool_enabled = bool(pool_cfg.get('enabled', False))
        self._opp_pool_cfg = pool_cfg

    def _make_opponent_engine(self, device):
        if not self.opponent_pool_enabled:
            return self.baseline_engine
        from model import Brain, ActorCritic
        from opponent_pool import OpponentPool
        from ppo_pool_engine import PPOOpponentPoolEngine

        run_dir = path.dirname(path.dirname(self.log_dir))
        ckpt_dir = path.join(run_dir, 'checkpoints')
        init0 = path.join(ckpt_dir, 'step_000000.pth')
        fallback = init0 if path.isfile(init0) else config['ppo'].get('init_checkpoint')
        pool = OpponentPool(
            ckpt_dir,
            past_k=int(self._opp_pool_cfg.get('past_k', 5)),
            latest_prob=float(self._opp_pool_cfg.get('latest_prob', 0.5)),
            fallback_checkpoint=fallback,
        )
        version = self.chal_version
        conv_channels = config['resnet']['conv_channels']
        num_blocks = config['resnet']['num_blocks']
        brain = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks)
        actor_critic = ActorCritic(version=version, tau=config['ppo']['tau_init'])
        from ppo_engine import dump_engine_config

        opp = PPOOpponentPoolEngine(
            brain,
            actor_critic,
            pool,
            is_oracle=False,
            version=version,
            device=device,
            enable_amp=True,
            name='opp_pool',
            eval_mode=False,
        )
        opp_cfg = dump_engine_config(opp)
        logging.info(f'opponent pool engine config dump: {opp_cfg}')
        assert not opp.enable_rule_based_agari_guard, 'train rollout: pool guard must be OFF'
        assert not hasattr(opp, 'pending_steps'), 'pool engine must not record trajectories'
        return opp

    def train_play(self, mortal, dqn, device, beta_sel=0.0):
        torch.backends.cudnn.benchmark = False
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            boltzmann_epsilon = self.boltzmann_epsilon,
            boltzmann_temp = self.boltzmann_temp,
            top_p = self.top_p,
            device = device,
            enable_amp = True,
            beta_sel = beta_sel,
            name = 'trainee',
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        rankings = env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (self.train_seed, self.train_key),
            seed_count = self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = sorted(
            path.join(self.log_dir, p) for p in os.listdir(self.log_dir)
        )

        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return rankings, file_list

    def train_play_ppo(self, engine, device):
        torch.backends.cudnn.benchmark = False
        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        rankings = env.py_vs_py(
            challenger = engine,
            champion = self._make_opponent_engine(device),
            seed_start = (self.train_seed, self.train_key),
            seed_count = self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = sorted(
            path.join(self.log_dir, p) for p in os.listdir(self.log_dir)
        )

        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return rankings, file_list
