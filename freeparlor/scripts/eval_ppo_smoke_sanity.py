#!/usr/bin/env python3
"""Final 100-hanchan self-play sanity for PPO P2 smoke checkpoint."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
from config import config
from libriichi.stat import Stat
from model import ActorCritic, Brain, load_ppo_from_mortal_checkpoint
from player import TestPlayer
from ppo_engine import dump_engine_config


class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('eval_sanity')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh = FlushingFileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def count_logs(log_dir: Path) -> int:
    if not log_dir.is_dir():
        return 0
    return sum(1 for _ in log_dir.glob('*.json.gz'))


def load_checkpoint(state_file: Path, mortal: Brain, ac: ActorCritic, device: torch.device) -> int:
    state = torch.load(state_file, weights_only=True, map_location=device)
    mortal.load_state_dict(state['mortal'])
    if 'actor_critic' in state:
        ac.load_state_dict(state['actor_critic'])
    else:
        load_ppo_from_mortal_checkpoint(ac, str(state_file), map_location=device)
    return int(state.get('steps', 0))


def main():
    run_dir = Path(config['control']['state_file']).parent
    eval_label = os.environ.get('EVAL_LABEL', 'step400')
    state_file = Path(os.environ.get('EVAL_CHECKPOINT', config['control']['state_file']))
    log_path = run_dir / 'logs' / f'eval_sanity_{eval_label}.log'
    log = setup_logging(log_path)

    log.info('eval_sanity: 起動 label=%s', eval_label)
    log.info('config 読了 (MORTAL_CFG=%s)', os.environ.get('MORTAL_CFG', 'config.toml'))
    log.info('checkpoint=%s', state_file)
    log.info('device=%s games=%s log_dir=%s',
             config['control']['device'],
             config['test_play']['games'],
             config['test_play']['log_dir'])

    device = torch.device(config['control']['device'])
    total_hanchans = config['test_play']['games'] // 4
    batch_size = 10
    log_dir = Path(config['test_play']['log_dir'])

    log.info('checkpoint ロード開始: %s', state_file)
    version = config['control']['version']
    mortal = Brain(version=version, **config['resnet']).to(device).eval()
    ac = ActorCritic(version=version, tau=config['ppo']['tau_init']).to(device).eval()
    ckpt_steps = load_checkpoint(state_file, mortal, ac, device)
    log.info('checkpoint ロード完了: steps=%s', ckpt_steps)

    # 構成 dump（診断専用の使い捨てengine。tp.test_play_ppo が内部で
    # 構築する engine と同一 kwargs — player.py:_make_ppo_eval_engine
    # 参照）。p_enrich / call_bonus_b が eval 経路で常時 0 であることを
    # ここで assert する (stage2_design.md §2、stage3_design.md §2/§6)。
    from eval_grp_baseline_1v3 import build_challenger_engine
    _diag_engine, _ = build_challenger_engine(state_file, device, name='mortal')
    diag_cfg = dump_engine_config(_diag_engine)
    log.info('eval engine config dump: %s', diag_cfg)
    assert diag_cfg['p_enrich'] == 0.0, 'eval 経路は p_enrich=0 必須'
    assert diag_cfg['call_bonus_b'] == 0.0, 'eval 経路は call_bonus_b=0 必須'
    assert diag_cfg['eval_mode'] is True, 'eval 経路は argmax (eval_mode=True) 必須'
    assert diag_cfg['enable_rule_based_agari_guard'] is True, 'eval 経路は guard ON 必須'
    del _diag_engine

    tp = TestPlayer()
    log.info('arena 起動: %d hanchans (%d局ずつ進捗ログ)', total_hanchans, batch_size)

    for offset in range(0, total_hanchans, batch_size):
        n = min(batch_size, total_hanchans - offset)
        seed_start = (10000 + offset, 0x2000)
        log.info('対局バッチ開始: seed [%d, %d)', seed_start[0], seed_start[0] + n)
        tp.test_play_ppo(
            n,
            mortal,
            ac,
            device,
            seed_start=seed_start,
            clear_log_dir=(offset == 0),
        )
        done = min(offset + n, total_hanchans)
        n_logs = count_logs(log_dir)
        log.info('対局 %d/%d 完了 (json.gz=%d)', done, total_hanchans, n_logs)

    stat = Stat.from_dir(str(log_dir), 'mortal')

    log.info('avg_rank=%.4f', stat.avg_rank)
    log.info('houjuu_rate=%.2f%%', stat.houjuu_rate * 100)
    log.info('agari_rate=%.2f%%', stat.agari_rate * 100)
    log.info('fuuro_rate=%.2f%%', stat.fuuro_rate * 100)
    log.info('riichi_rate=%.2f%%', stat.riichi_rate * 100)
    log.info('ryukyoku_rate=%.2f%%', stat.ryukyoku_rate * 100)
    log.info('eval_sanity: 完了')

    print(f'label={eval_label}')
    print(f'checkpoint={state_file}')
    print(f'avg_rank={stat.avg_rank:.4f}')
    print(f'houjuu_rate={stat.houjuu_rate * 100:.2f}%')
    print(f'agari_rate={stat.agari_rate * 100:.2f}%')
    print(f'fuuro_rate={stat.fuuro_rate * 100:.2f}%')
    print(f'riichi_rate={stat.riichi_rate * 100:.2f}%')
    print(f'ryukyoku_rate={stat.ryukyoku_rate * 100:.2f}%')


if __name__ == '__main__':
    main()
