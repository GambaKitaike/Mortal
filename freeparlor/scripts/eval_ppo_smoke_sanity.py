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
from model import ActorCritic, Brain
from player import TestPlayer


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


def main():
    run_dir = Path(config['control']['state_file']).parent
    log_path = run_dir / 'logs' / 'eval_sanity.log'
    log = setup_logging(log_path)

    log.info('eval_sanity: 起動')
    log.info('config 読了 (MORTAL_CFG=%s)', os.environ.get('MORTAL_CFG', 'config.toml'))
    log.info('device=%s games=%s log_dir=%s',
             config['control']['device'],
             config['test_play']['games'],
             config['test_play']['log_dir'])

    device = torch.device(config['control']['device'])
    state_file = config['control']['state_file']
    total_hanchans = config['test_play']['games'] // 4
    batch_size = 10
    log_dir = Path(config['test_play']['log_dir'])

    log.info('checkpoint ロード開始: %s', state_file)
    state = torch.load(state_file, weights_only=True, map_location=device)
    version = config['control']['version']
    mortal = Brain(version=version, **config['resnet']).to(device).eval()
    ac = ActorCritic(version=version, tau=config['ppo']['tau_init']).to(device).eval()
    mortal.load_state_dict(state['mortal'])
    ac.load_state_dict(state['actor_critic'])
    log.info('checkpoint ロード完了: steps=%s', state.get('steps', '?'))

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
    log.info('eval_sanity: 完了')

    print(f'avg_rank={stat.avg_rank:.4f}')
    print(f'houjuu_rate={stat.houjuu_rate * 100:.2f}%')
    print(f'agari_rate={stat.agari_rate * 100:.2f}%')
    print(f'fuuro_rate={stat.fuuro_rate * 100:.2f}%')
    print(f'riichi_rate={stat.riichi_rate * 100:.2f}%')


if __name__ == '__main__':
    main()
