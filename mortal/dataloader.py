import gzip
import json
import random
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from torch.utils.data import IterableDataset
from model import GRP
from reward_calculator import RewardCalculator
from libriichi.dataset import GameplayLoader
from config import config


def open_log(path):
    path = Path(path)
    with open(path, 'rb') as f:
        magic = f.read(2)
    if magic == b'\x1f\x8b':
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'rt', encoding='utf-8')


def get_hora_chip_delta(ev):
    meta = ev.get('meta') or {}
    if 'chip_delta' in meta:
        return meta['chip_delta']
    return ev.get('chip_delta')


def load_kyoku_hora_r_chip(file_path, player_id):
    """Sum hora meta.chip_delta[player_id] per kyoku (transport only; see assign_r_chip)."""
    per_kyoku = defaultdict(float)
    try:
        with open_log(file_path) as f:
            kyoku_idx = -1
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get('type') == 'start_kyoku':
                    kyoku_idx += 1
                elif ev.get('type') == 'hora' and kyoku_idx >= 0:
                    chip_delta = get_hora_chip_delta(ev)
                    if chip_delta is not None:
                        per_kyoku[kyoku_idx] += chip_delta[player_id]
    except OSError:
        pass
    return dict(per_kyoku)


def assign_r_chip_to_trainee_final_moves(game_size, at_kyoku, kyoku_hora_r_chip):
    """Attribute each kyoku's summed hora chip_delta to the trainee's last move in that kyoku."""
    r_chip = np.zeros(game_size, dtype=np.float32)
    last_idx_by_kyoku = {}
    for i in range(game_size):
        last_idx_by_kyoku[at_kyoku[i]] = i
    for kyoku, idx in last_idx_by_kyoku.items():
        r_chip[idx] = kyoku_hora_r_chip.get(kyoku, 0.0)
    return r_chip


def build_td_transitions(obs, masks, at_kyoku, dones):
    """Adjacent (s,a,r,s',done) pairing within kyoku. n-step (n=3) aggregation is layer 3."""
    game_size = len(obs)
    next_obs = []
    next_masks = []
    done_chip = []
    for i in range(game_size):
        has_next = (
            i + 1 < game_size
            and at_kyoku[i + 1] == at_kyoku[i]
            and not dones[i]
        )
        if has_next:
            next_obs.append(obs[i + 1])
            next_masks.append(masks[i + 1])
            done_chip.append(0)
        else:
            next_obs.append(np.zeros_like(obs[i]))
            next_masks.append(np.zeros_like(masks[i], dtype=bool))
            done_chip.append(1)
    return next_obs, next_masks, done_chip


class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle = False,
        file_batch_size = 20, # hint: around 660 instances per file
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.iterator = None

    def build_iter(self):
        # do not put it in __init__, it won't work on Windows
        self.grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        self.grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(
            self.grp, self.pts,
            alpha=config['env'].get('alpha', 1.0),
            gamma_pt=config['env'].get('gamma_pt', 1.0),
        )
        self.beta = config['env'].get('beta', 0.0)
        self.chip_value = config['env'].get('chip_value', 5.0)
        self.lambda_opp = config['env'].get('lambda_opp', 0.0)
        self.noten_factor = config['env'].get('noten_factor', 0.0)
        chip_dir = config['env'].get('chip_dir', '/home/gamba/mahjong/data/tenhou/chips')
        self.chip_dir = Path(chip_dir)

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        # shuffle the file list for each epoch
        random.shuffle(self.file_list)

        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def load_chip_deltas(self, file_path, player_id, n_kyoku):
        chip_path = self.chip_dir / f"{Path(file_path).name}.npz"
        if not chip_path.exists():
            return np.zeros(n_kyoku, dtype=np.float64)
        chips = np.load(chip_path)['chips']
        if chips.shape[0] < n_kyoku:
            padded = np.zeros((n_kyoku, 4), dtype=np.float64)
            padded[:chips.shape[0]] = chips
            chips = padded
        return chips[:n_kyoku, player_id].astype(np.float64)

    def load_kyoku_probe_arrays(self, file_path, player_id, n_kyoku):
        chip_path = self.chip_dir / f"{Path(file_path).name}.npz"
        keys = ('aka_held', 'tenpai_end', 'won', 'dealt_in')
        defaults = {
            'aka_held': np.int16,
            'tenpai_end': np.int8,
            'won': np.int8,
            'dealt_in': np.int8,
        }
        out = {k: np.zeros(n_kyoku, dtype=defaults[k]) for k in keys}
        if not chip_path.exists():
            return out
        with np.load(chip_path) as data:
            for k in keys:
                if k not in data:
                    continue
                arr = data[k]
                if arr.shape[0] < n_kyoku:
                    padded = np.zeros((n_kyoku, 4), dtype=arr.dtype)
                    padded[:arr.shape[0]] = arr
                    arr = padded
                out[k] = arr[:n_kyoku, player_id]
        return out

    def populate_buffer(self, file_list):
        data = self.loader.load_gz_log_files(file_list)
        for file_path, file in zip(file_list, data):
            for game in file:
                # per move
                obs = game.take_obs()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()

                # per game
                grp = game.take_grp()
                player_id = game.take_player_id()

                game_size = len(obs)

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                final_scores = grp.take_final_scores()
                chip_deltas = self.load_chip_deltas(file_path, player_id, len(grp_feature))
                probe = self.load_kyoku_probe_arrays(file_path, player_id, len(grp_feature))
                kyoku_rewards = self.reward_calc.calc_delta_blend(
                    player_id, grp_feature, rank_by_player, final_scores,
                    alpha=self.reward_calc.alpha, gamma_pt=self.reward_calc.gamma_pt,
                    chip_deltas=chip_deltas, beta=self.beta, chip_value=self.chip_value,
                    aka_held=probe['aka_held'], tenpai_end=probe['tenpai_end'],
                    won=probe['won'], dealt_in=probe['dealt_in'],
                    lambda_opp=self.lambda_opp, noten_factor=self.noten_factor,
                )
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1 # usually they are equal, unless there is no action in the last kyoku
                scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                kyoku_hora_r_chip = load_kyoku_hora_r_chip(file_path, player_id)
                r_chip = assign_r_chip_to_trainee_final_moves(
                    game_size, at_kyoku, kyoku_hora_r_chip,
                )
                next_obs, next_masks, done_chip = build_td_transitions(
                    obs, masks, at_kyoku, dones,
                )

                for i in range(game_size):
                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        steps_to_done[i],
                        kyoku_rewards[at_kyoku[i]],
                        player_ranks[at_kyoku[i] + 1],
                    ]
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    entry.extend([
                        next_obs[i],
                        next_masks[i],
                        done_chip[i],
                        r_chip[i],
                    ])
                    self.buffer.append(entry)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
