#!/usr/bin/env python3
"""メタ対決 probe: Stage2 checkpoint (challenger) vs Stage1 checkpoint (baseline) の 1v3。

stage2_design.md §5 レンズ3。濃縮 (p_enrich) で Stage2 が立てた鳴き戦術（立ったなら）が、
Stage1 の立直マキシマリズム方策 3 席が占める卓で生存する（搾取されない）かの
搾取可能性テスト第一歩。ハーネスは eval_grp_baseline_1v3.py の座席ローテ仕様
(OneVsThree.py_vs_py, 1 seed = 4 半荘・座席均等ローテ) をそのまま流用し、
baseline を DQN から Stage1 の PPOEngine に差し替えたのみ。

eval 経路のため challenger/baseline とも p_enrich は PPOEngine 既定の 0.0
（構成 dump で確認・assert）。
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

import torch

from config import config
from libriichi.arena import OneVsThree
from libriichi.stat import Stat
from ppo_engine import dump_engine_config

from eval_grp_baseline_1v3 import build_challenger_engine


class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('eval_meta_stage1_vs_stage2')
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


def main():
    run_dir = Path(config['control']['state_file']).parent
    eval_label = os.environ.get('EVAL_LABEL', 'meta')
    challenger_ckpt = Path(os.environ['EVAL_CHALLENGER_CHECKPOINT'])
    baseline_ckpt = Path(os.environ['EVAL_BASELINE_CHECKPOINT'])

    seed_base = int(os.environ.get('EVAL_SEED_BASE', 10000))
    seed_count = int(os.environ.get('EVAL_SEED_COUNT', 100))
    seed_key = int(os.environ.get('EVAL_SEED_KEY', 0x2000))

    log_path = run_dir / 'logs' / 'eval_meta' / f'eval_{eval_label}.log'
    log = setup_logging(log_path)

    log.info('eval_meta_stage1_vs_stage2: 起動 label=%s', eval_label)
    log.info('config 読了 (MORTAL_CFG=%s)', os.environ.get('MORTAL_CFG', 'config.toml'))
    log.info('challenger(Stage2) checkpoint=%s', challenger_ckpt)
    log.info('baseline(Stage1) checkpoint=%s', baseline_ckpt)

    device = torch.device(config['control']['device'])

    challenger_engine, challenger_steps = build_challenger_engine(
        challenger_ckpt, device, name='challenger',
    )
    baseline_engine, baseline_steps = build_challenger_engine(
        baseline_ckpt, device, name='baseline',
    )

    challenger_dump = dump_engine_config(challenger_engine)
    baseline_dump = dump_engine_config(baseline_engine)
    log.info('challenger engine config: %s', challenger_dump)
    log.info('baseline engine config: %s', baseline_dump)
    assert challenger_dump['eval_mode'] is True, 'challenger must be argmax (eval_mode=True)'
    assert baseline_dump['eval_mode'] is True, 'baseline must be argmax (eval_mode=True)'
    assert challenger_dump['enable_rule_based_agari_guard'] is True, 'challenger guard must be ON'
    assert baseline_dump['enable_rule_based_agari_guard'] is True, 'baseline guard must be ON'
    assert challenger_dump['p_enrich'] == 0.0, 'eval 経路の challenger は p_enrich=0 必須'
    assert baseline_dump['p_enrich'] == 0.0, 'eval 経路の baseline は p_enrich=0 必須'
    assert challenger_dump['call_bonus_b'] == 0.0, 'eval 経路の challenger は call_bonus_b=0 必須'
    assert baseline_dump['call_bonus_b'] == 0.0, 'eval 経路の baseline は call_bonus_b=0 必須'
    log.info('challenger steps=%s baseline steps=%s', challenger_steps, baseline_steps)

    # 座席ローテ仕様は eval_grp_baseline_1v3.py と同一
    # (1 seed => 4 半荘、challenger 座席 0->1->2->3、baseline が残り3席)。
    game_log_dir = run_dir / 'logs' / 'eval_meta' / f'game_logs_{eval_label}'
    if game_log_dir.exists():
        shutil.rmtree(game_log_dir)
    game_log_dir.mkdir(parents=True)

    total_hanchans = seed_count * 4
    log.info(
        '対戦開始: seed [%d, %d) key=0x%x, %d hanchans (4/seed, 座席均等ローテーション)',
        seed_base, seed_base + seed_count, seed_key, total_hanchans,
    )

    env = OneVsThree(disable_progress_bar=True, log_dir=str(game_log_dir))
    rankings = env.py_vs_py(
        challenger=challenger_engine,
        champion=baseline_engine,
        seed_start=(seed_base, seed_key),
        seed_count=seed_count,
    )
    rankings = list(rankings)
    log.info('対戦完了: challenger rankings(1着,2着,3着,4着 度数)=%s', rankings)

    n_logs = sum(1 for _ in game_log_dir.glob('*.json.gz'))
    assert n_logs == total_hanchans, f'json.gz count {n_logs} != expected {total_hanchans}'

    stat = Stat.from_dir(str(game_log_dir), 'challenger')

    log.info('avg_rank=%.4f', stat.avg_rank)
    log.info('houjuu_rate=%.2f%%', stat.houjuu_rate * 100)
    log.info('agari_rate=%.2f%%', stat.agari_rate * 100)
    log.info('fuuro_rate=%.2f%%', stat.fuuro_rate * 100)
    log.info('riichi_rate=%.2f%%', stat.riichi_rate * 100)
    log.info('ryukyoku_rate=%.2f%%', stat.ryukyoku_rate * 100)
    log.info('eval_meta_stage1_vs_stage2: 完了')

    print(f'label={eval_label}')
    print(f'challenger_checkpoint={challenger_ckpt}')
    print(f'challenger_steps={challenger_steps}')
    print(f'baseline_checkpoint={baseline_ckpt}')
    print(f'baseline_steps={baseline_steps}')
    print(f'seed_range=[{seed_base}, {seed_base + seed_count})')
    print('hanchans_per_seed=4')
    print(f'total_hanchans={total_hanchans}')
    print(f'avg_rank={stat.avg_rank:.4f}')
    print(f'houjuu_rate={stat.houjuu_rate * 100:.2f}%')
    print(f'agari_rate={stat.agari_rate * 100:.2f}%')
    print(f'fuuro_rate={stat.fuuro_rate * 100:.2f}%')
    print(f'riichi_rate={stat.riichi_rate * 100:.2f}%')
    print(f'ryukyoku_rate={stat.ryukyoku_rate * 100:.2f}%')
    print(f'rankings_1st_2nd_3rd_4th={rankings}')


if __name__ == '__main__':
    main()
