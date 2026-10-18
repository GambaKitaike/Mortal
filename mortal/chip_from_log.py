"""Per-kyoku chip deltas from self-play json.gz logs (online PPO path)."""

from __future__ import annotations

import gzip
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def _import_hora_chip_deltas():
    scripts = Path(__file__).resolve().parents[1] / 'freeparlor' / 'scripts'
    scripts_str = str(scripts)
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
    from preprocess_chips import hora_chip_deltas
    return hora_chip_deltas


def chip_delta_at_hora(events, hora_idx):
    """Arena hora chip_delta from log meta, or replay via preprocess_chips.hora_chip_deltas."""
    ev = events[hora_idx]
    chip_delta = get_hora_chip_delta(ev)
    if chip_delta is not None:
        return chip_delta

    from libriichi.state import PlayerState
    hora_chip_deltas = _import_hora_chip_deltas()
    states = [PlayerState(i) for i in range(4)]
    for e in events[:hora_idx]:
        if e.get('type') == 'hora':
            continue
        payload = json.dumps(e, separators=(',', ':'))
        for s in states:
            s.update(payload)
    actor = ev['actor']
    is_ron = actor != ev['target']
    ura = ev.get('ura_markers') or []
    detail = states[actor].agari_detail(is_ron, ura)
    return hora_chip_deltas(ev, detail)


def load_kyoku_chip_deltas_from_log(file_path, player_id, n_kyoku):
    """Per-kyoku chip deltas for one player from self-play json.gz."""
    with open_log(file_path) as f:
        events = [json.loads(line) for line in f if line.strip()]

    per_kyoku = defaultdict(float)
    unresolved = 0
    kyoku_idx = -1
    for i, ev in enumerate(events):
        if ev.get('type') == 'start_kyoku':
            kyoku_idx += 1
        elif ev.get('type') == 'hora' and kyoku_idx >= 0:
            try:
                chip_delta = chip_delta_at_hora(events, i)
            except Exception as exc:
                unresolved += 1
                logging.error(
                    'chip_delta unresolved at hora idx=%s file=%s: %s',
                    i, file_path, exc,
                )
            else:
                per_kyoku[kyoku_idx] += chip_delta[player_id]

    if unresolved:
        raise RuntimeError(
            f'chip_delta unresolved for {unresolved} hora event(s) in {file_path}',
        )

    chip_deltas = np.zeros(n_kyoku, dtype=np.float64)
    for k, v in per_kyoku.items():
        if k < n_kyoku:
            chip_deltas[k] = v
        elif v != 0:
            logging.error(
                'chip kyoku index %s out of range n_kyoku=%s file=%s delta=%s',
                k, n_kyoku, file_path, v,
            )
            raise RuntimeError(
                f'chip kyoku index {k} >= n_kyoku={n_kyoku} in {file_path}',
            )
    return chip_deltas
