"""DRCA プローブ共有ヘルパ (drca_probe_design.md §2/§5a).

drca_collect_branchpoints.py / drca_run_probe.py / drca_aggregate.py が
共通で使う: checkpoint ロード・本番 client と同一構成での engine 構築
(p_enrich=0.0 / call_bonus_b=0.0 assert 込み)・ログからの全4席分解デコード
(既存 GameplayLoader を再利用、libriichi 変更なし)・sel 定義
(鳴き可能∧赤保持) の再利用・局単位の正典3ストリーム報酬計算。

学習コード (train_ppo.py / client.py / ppo.py) は一切 import 変更しない。
本ファイルは診断専用で、生成データを学習に流用する経路は存在しない。
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MORTAL_DIR = ROOT / 'mortal'
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

DEFAULT_DRCA_CFG = ROOT / 'freeparlor' / 'configs' / 'ppo_drca_probe.toml'

# MORTAL_CFG must be set before `from config import config` (mortal/config.py
# reads the env var at import time). Default to the static DRCA support
# config (read-only, no run dir) unless the caller already set one.
os.environ.setdefault('MORTAL_CFG', str(DEFAULT_DRCA_CFG))

from config import config  # noqa: E402
from libriichi.arena import OneVsThree  # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402
from chip_from_log import load_kyoku_chip_deltas_from_log  # noqa: E402
from model import ActorCritic, Brain, GRP, load_ppo_from_mortal_checkpoint  # noqa: E402
from ppo import AKA_OBS_ROWS, CALL_ACTION_MAX, CALL_ACTION_MIN  # noqa: E402
from ppo_engine import build_production_trainee_engine, dump_engine_config  # noqa: E402
from reward_calculator import RewardCalculator  # noqa: E402

# `Some(45)` in libriichi/src/dataset/gameplay.rs: the label emitted when a
# player had a chi/pon/kan/ron option and actively declined it. This is the
# forced action for the DRCA "no-call" arm.
DECLINE_ACTION = 45


def load_checkpoint_into(state_file, mortal: Brain, ac: ActorCritic, device) -> int:
    """Same loader as freeparlor/scripts/eval_ppo_smoke_sanity.py:load_checkpoint."""
    state = torch.load(state_file, weights_only=True, map_location=device)
    mortal.load_state_dict(state['mortal'])
    if 'actor_critic' in state:
        ac.load_state_dict(state['actor_critic'])
    else:
        load_ppo_from_mortal_checkpoint(ac, str(state_file), map_location=device)
    return int(state.get('steps', 0))


def build_policy_engine(state_file, device, *, name: str) -> tuple:
    """Build a PPOEngine via the exact production-trainee-client builder
    (mortal/ppo_engine.py:build_production_trainee_engine, the same one
    mortal/client.py's train rollout uses), loaded from `state_file`.

    p_enrich is never passed (stays at the builder's 0.0 default: DRCA is a
    natural-distribution diagnostic, stage2_design.md §2 discipline).
    Returns (engine, steps, config_dump). Caller must assert
    config_dump['p_enrich'] == 0.0 and config_dump['call_bonus_b'] == 0.0
    (both are already guaranteed by construction, but re-asserted at every
    call site per the "no silent fallback" discipline).
    """
    version = config['control']['version']
    mortal = Brain(version=version, **config['resnet']).to(device).eval()
    ac = ActorCritic(version=version, tau=config['ppo']['tau_init']).to(device).eval()
    steps = load_checkpoint_into(state_file, mortal, ac, device)
    engine = build_production_trainee_engine(mortal, ac, version=version, device=device, name=name)
    dump = dump_engine_config(engine)
    assert dump['p_enrich'] == 0.0, f'DRCA engine {name}: p_enrich must be 0.0, got {dump["p_enrich"]}'
    assert dump['call_bonus_b'] == 0.0, f'DRCA engine {name}: call_bonus_b must be 0.0, got {dump["call_bonus_b"]}'
    return engine, steps, dump


def assert_construction_diff_empty(dump_a: dict, dump_b: dict, *, ignore=('name',)) -> None:
    """検定・診断の self-play client は本番 client と同一構成 (CLAUDE.md).

    Compares two dump_engine_config() outputs, ignoring cosmetic fields
    (engine name). Raises loudly (no silent skip) on any real divergence.
    """
    a = {k: v for k, v in dump_a.items() if k not in ignore}
    b = {k: v for k, v in dump_b.items() if k not in ignore}
    if a != b:
        raise RuntimeError(f'engine construction drift: {a} != {b}')


def call_possible_aka_held(obs: np.ndarray, mask: np.ndarray) -> bool:
    """The existing sel-family predicate (mortal/ppo.py:apply_call_bonus),
    minus the call_taken term: a branch point is any decision where a call
    is legally available AND the seat holds a red five, regardless of what
    was actually chosen. Reused verbatim, not redefined
    (drca_probe_design.md §1).
    """
    mask = np.asarray(mask)
    call_possible = bool(mask[CALL_ACTION_MIN:CALL_ACTION_MAX + 1].any())
    if not call_possible:
        return False
    obs = np.asarray(obs)
    aka_held = bool(np.abs(obs[AKA_OBS_ROWS, :]).sum() > 0)
    return aka_held


def legal_call_action_ids(mask: np.ndarray) -> list[int]:
    mask = np.asarray(mask)
    return [i for i in range(CALL_ACTION_MIN, CALL_ACTION_MAX + 1) if bool(mask[i])]


@dataclass
class SeatGameplay:
    player_id: int
    obs: list  # list[np.ndarray], per-decision feature planes
    masks: list  # list[np.ndarray[bool]]
    actions: list  # list[int]
    at_kyoku: list  # list[int]
    at_turns: list
    shantens: list


def reconstruct_all_seats(log_path: str, version: int) -> dict[int, SeatGameplay]:
    """Reconstruct all 4 seats' per-decision (obs, mask, action) streams
    from a json.gz mjai log via the existing GameplayLoader (used
    unmodified, same tool mortal/client.py:_finalize_ppo_trajectories uses
    for reward assignment). No engine/pending-trajectory bookkeeping
    involved -- this is a pure re-derivation from the log, deterministic
    given libriichi's PlayerState replay.
    """
    loader = GameplayLoader(
        version=version, oracle=False, player_names=[],
        always_include_kan_select=True,
    )
    games = loader.load_gz_log_files([str(log_path)])[0]
    out: dict[int, SeatGameplay] = {}
    for gp in games:
        pid = gp.take_player_id()
        out[pid] = SeatGameplay(
            player_id=pid,
            obs=[np.asarray(a) for a in gp.take_obs()],
            masks=[np.asarray(a) for a in gp.take_masks()],
            actions=gp.take_actions(),
            at_kyoku=gp.take_at_kyoku(),
            at_turns=gp.take_at_turns(),
            shantens=gp.take_shantens(),
        )
    assert len(out) == 4, f'expected 4 seats reconstructed from {log_path}, got {sorted(out)}'
    return out


def mask_obs_digest(obs: np.ndarray, mask: np.ndarray) -> str:
    """Short digest identifying an exact (obs, mask) pair, used to
    cross-check a stored branch point still matches on re-derivation
    (part of the r2 seed-determinism guard -- see drca_run_probe.py).
    """
    h = hashlib.sha256()
    h.update(np.asarray(mask).astype(np.uint8).tobytes())
    h.update(np.asarray(obs).astype(np.float32).tobytes())
    return h.hexdigest()[:16]


def split_letter_for_seat(seat: int) -> str:
    """OneVsThree::run_batch (libriichi/src/arena/one_vs_three.rs) fixes
    challenger_player_id = seat index for split a/b/c/d in that exact
    order (champion_player_ids_per_seed = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]
    for splits a..d respectively). Deterministic, not something we choose.
    """
    return ['a', 'b', 'c', 'd'][seat]


def parse_game_key(game_key: str) -> tuple[int, int, str]:
    seed_s, key_s, split = game_key.rsplit('_', 2)
    return int(seed_s), int(key_s), split


@dataclass
class KyokuRewards:
    reward_sotensu: list
    reward_grp: list
    reward_chip: list
    rank_by_player: list
    final_scores: list
    chip_deltas: list


_grp_model_cache = {}


def _load_grp_model(device):
    key = str(device)
    if key not in _grp_model_cache:
        grp = GRP(**config['grp']['network'])
        state = torch.load(config['grp']['state_file'], weights_only=True, map_location='cpu')
        grp.load_state_dict(state['model'])
        _grp_model_cache[key] = grp
    return _grp_model_cache[key]


def compute_kyoku_rewards(log_path: str, player_id: int, *, device=None) -> KyokuRewards:
    """Canonical 3-stream composite reward (alpha=beta=gamma=1), per kyoku,
    for one seat of one full-hanchan log. Same computation as
    mortal/client.py:_finalize_ppo_trajectories, generalized to an
    arbitrary player_id (that function is hardwired to the 'trainee'-named
    seat only).
    """
    grp = _load_grp_model(device or torch.device('cpu'))
    reward_calc = RewardCalculator(
        grp, config['env']['pts'],
        alpha=config['env'].get('alpha', 1.0),
        gamma_pt=config['env'].get('gamma_pt', 1.0),
    )
    beta = config['env'].get('beta', 1.0)
    chip_value = config['env'].get('chip_value', 5.0)

    loader = GameplayLoader(
        version=config['control']['version'], oracle=False, player_names=[],
        always_include_kan_select=True,
    )
    games = loader.load_gz_log_files([str(log_path)])[0]
    gp = next(g for g in games if g.take_player_id() == player_id)
    grp_obj = gp.take_grp()
    grp_feature = grp_obj.take_feature()
    rank_by_player = grp_obj.take_rank_by_player()
    final_scores = grp_obj.take_final_scores()

    chip_deltas = load_kyoku_chip_deltas_from_log(str(log_path), player_id, len(grp_feature))

    sotensu = reward_calc.calc_delta_points(player_id, grp_feature, final_scores) / 1000.0
    juni = reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
    reward_sotensu = reward_calc.alpha * sotensu
    reward_grp = reward_calc.gamma_pt * juni
    reward_chip = beta * chip_deltas * chip_value

    return KyokuRewards(
        reward_sotensu=reward_sotensu.tolist(),
        reward_grp=reward_grp.tolist(),
        reward_chip=reward_chip.tolist(),
        rank_by_player=list(map(int, rank_by_player)),
        final_scores=list(map(int, final_scores)),
        chip_deltas=chip_deltas.tolist() if hasattr(chip_deltas, 'tolist') else list(chip_deltas),
    )


def make_arena(log_dir: str):
    return OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
