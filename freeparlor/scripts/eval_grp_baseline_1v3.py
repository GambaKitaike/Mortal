#!/usr/bin/env python3
"""grp_baseline (DQN) 相手の 1v3 対戦で PPO checkpoint の強さを測る。

Stage1 の自己対戦 argmax eval バッテリー（ppo_p3_stage1_result.md §6）は
4人とも同一方策のため avg_rank が常に 2.5 に固定され、方策自体の強さの
変化を測れない。固定 baseline（grp_baseline.pth, DQN 時代のモデル）との
1v3 対戦により、challenger 視点の avg_rank で世代跨ぎの強さを測る
（stage2_design.md §5 レンズ2: grp_baseline 対戦）。

本スクリプトは eval 経路であり常時自然分布。Stage2 の board.rs
rejection sampling（p_enrich）は訓練 client 専用の介入であり、
本スクリプトが使う OneVsThree self-play 経路には一切関与しない
（p_enrich が実装された後も、challenger 側の配牌は常に自然分布のまま）。
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
from model import ActorCritic, Brain, DQN
from engine import MortalEngine
from ppo_engine import PPOEngine, dump_engine_config

# eval_ppo_smoke_sanity.py と同一の checkpoint 読込経路:
# state['mortal'] + state['actor_critic'] があればそれを使い、
# actor_critic が無ければ load_ppo_from_mortal_checkpoint で
# DQN 由来 checkpoint から変換する。
from eval_ppo_smoke_sanity import load_checkpoint

# baseline (DQN) 固定パス。verify_ppo_p1.py check(9)
# check_dqn_one_vs_three (同ファイル L311-355 付近) と同一の
# 構築手順を使う。verify_ppo_p1.py 本体は変更しない。
DEFAULT_BASELINE_CKPT = '/home/gamba/mahjong/runs/grp_baseline.pth'


class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('eval_grp_baseline_1v3')
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


def build_challenger_engine(
    state_file: Path, device: torch.device, name: str = 'challenger',
) -> tuple[PPOEngine, int]:
    """PPO checkpoint から eval 用 PPOEngine を構築する（名称は name で指定）。

    エンジン構成は eval_ppo_smoke_sanity.py の eval 経路
    (player.py:TestPlayer._make_ppo_eval_engine と同一 kwargs) に揃える:
    enable_amp=False, enable_quick_eval=False,
    enable_rule_based_agari_guard=True (guard ON),
    eval_mode=True (argmax), record_trajectory=False。
    p_enrich は未指定=PPOEngine 既定の 0.0（eval 経路は常時自然分布）。
    """
    version = config['control']['version']
    mortal = Brain(version=version, **config['resnet']).to(device).eval()
    ac = ActorCritic(version=version, tau=config['ppo']['tau_init']).to(device).eval()
    ckpt_steps = load_checkpoint(state_file, mortal, ac, device)

    engine = PPOEngine(
        mortal,
        ac,
        is_oracle=False,
        version=version,
        device=device,
        enable_amp=False,
        enable_quick_eval=False,
        enable_rule_based_agari_guard=True,
        name=name,
        eval_mode=True,
        record_trajectory=False,
    )
    return engine, ckpt_steps


def build_baseline_engine(baseline_ckpt: Path, device: torch.device) -> MortalEngine:
    """baseline = grp_baseline.pth の DQN MortalEngine.

    verify_ppo_p1.py check(9) check_dqn_one_vs_three
    (verify_ppo_p1.py L326-355 付近) と同一の構築手順:
    Brain/DQN を state['config']['resnet'] から構築し、
    strict=False で current_dqn を読み込み、
    enable_amp=True / enable_rule_based_agari_guard=True で
    MortalEngine を作る。
    """
    baseline_state = torch.load(baseline_ckpt, weights_only=True, map_location='cpu')
    bcfg = baseline_state['config']
    bversion = bcfg['control'].get('version', 1)
    b_mortal = Brain(
        version=bversion,
        conv_channels=bcfg['resnet']['conv_channels'],
        num_blocks=bcfg['resnet']['num_blocks'],
    ).eval()
    b_dqn = DQN(version=bversion).eval()
    b_mortal.load_state_dict(baseline_state['mortal'])
    b_dqn.load_state_dict(baseline_state['current_dqn'], strict=False)

    return MortalEngine(
        b_mortal,
        b_dqn,
        is_oracle=False,
        version=bversion,
        device=device,
        enable_amp=True,
        enable_rule_based_agari_guard=True,
        name='baseline',
    )


def main():
    run_dir = Path(config['control']['state_file']).parent
    eval_label = os.environ.get('EVAL_LABEL', 'step0')
    state_file = Path(os.environ.get('EVAL_CHECKPOINT', config['control']['state_file']))
    baseline_ckpt = Path(os.environ.get('GRP_BASELINE_CKPT', DEFAULT_BASELINE_CKPT))

    seed_base = int(os.environ.get('EVAL_SEED_BASE', 10000))
    seed_count = int(os.environ.get('EVAL_SEED_COUNT', 100))
    seed_key = int(os.environ.get('EVAL_SEED_KEY', 0x2000))

    log_path = run_dir / 'logs' / 'eval_grp_baseline' / f'eval_{eval_label}.log'
    log = setup_logging(log_path)

    log.info('eval_grp_baseline_1v3: 起動 label=%s', eval_label)
    log.info('config 読了 (MORTAL_CFG=%s)', os.environ.get('MORTAL_CFG', 'config.toml'))
    log.info('challenger checkpoint=%s', state_file)
    log.info('baseline checkpoint=%s', baseline_ckpt)

    device = torch.device(config['control']['device'])

    challenger_engine, ckpt_steps = build_challenger_engine(state_file, device)
    baseline_engine = build_baseline_engine(baseline_ckpt, device)

    # 構成 dump をログ先頭に出力する（本番/eval 構成同一の規律。
    # 別 run・別 checkpoint との diff 確認に使う）。
    challenger_dump = dump_engine_config(challenger_engine)
    baseline_dump = dump_engine_config(baseline_engine)
    log.info('challenger engine config: %s', challenger_dump)
    log.info('baseline engine config: %s', baseline_dump)
    assert challenger_dump['eval_mode'] is True, 'challenger must be argmax (eval_mode=True)'
    assert challenger_dump['enable_rule_based_agari_guard'] is True, 'challenger guard must be ON'
    assert baseline_dump['enable_rule_based_agari_guard'] is True, 'baseline guard must be ON'
    log.info('challenger steps=%s', ckpt_steps)

    # --- OneVsThree 席替え仕様（libriichi/src/arena/one_vs_three.rs で確認済み） ---
    # 1 seed (u64, u64) につき 4 半荘 (split a/b/c/d) を実行する。
    # challenger の座席は split ごとに 0 → 1 → 2 → 3 と1回ずつ回り、
    # champion (baseline) は各 split で残り3座席
    # ([1,2,3] / [0,2,3] / [0,1,3] / [0,1,2]) を占有する
    # （バッチエージェントなので baseline は3人分の player_id を
    # 1インスタンスで同時に担当する。別インスタンス生成は不要）。
    # py_vs_py が返す rankings は「challenger の」着順分布
    # (0=1着 .. 3=4着 の度数、python 側で見るときは +1 して解釈)。
    # よって seed_count 個の seed => seed_count*4 半荘、
    # challenger は全4座席を均等回数だけ経験する。
    game_log_dir = run_dir / 'logs' / 'eval_grp_baseline' / f'game_logs_{eval_label}'
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

    # Stat.from_dir は各対局の names[] のうち player_name と一致する
    # 座席だけを集計するため、challenger の4座席分すべてのスタッツが
    # name='challenger' 一括指定で拾える（champion 側の3座席は
    # name='baseline' で除外される）。
    stat = Stat.from_dir(str(game_log_dir), 'challenger')

    log.info('avg_rank=%.4f', stat.avg_rank)
    log.info('houjuu_rate=%.2f%%', stat.houjuu_rate * 100)
    log.info('agari_rate=%.2f%%', stat.agari_rate * 100)
    log.info('fuuro_rate=%.2f%%', stat.fuuro_rate * 100)
    log.info('riichi_rate=%.2f%%', stat.riichi_rate * 100)
    log.info('ryukyoku_rate=%.2f%%', stat.ryukyoku_rate * 100)
    log.info('eval_grp_baseline_1v3: 完了')

    print(f'label={eval_label}')
    print(f'challenger_checkpoint={state_file}')
    print(f'challenger_steps={ckpt_steps}')
    print(f'baseline_checkpoint={baseline_ckpt}')
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
