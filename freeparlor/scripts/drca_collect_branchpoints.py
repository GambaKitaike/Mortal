#!/usr/bin/env python3
"""DRCA プローブ: 分岐点採取 (drca_probe_design.md §5a-4)。

被測定 checkpoint による自己対戦 (全4席同一重み、pure π サンプリング、
p_enrich=0.0=自然分布、本番 client と同一 engine 構成) を走らせ、鳴き可能
∧赤保持 (mortal/ppo.py:apply_call_bonus の sel 定義から call_taken 項を
除いたもの -- 既存計装と同一、新規定義は作らない) の decision step を
分岐点候補として記録する。

抽出プロトコル (§5a-4 に固定、変更禁止):
  - 1局あたり最大1分岐点 (同一局内に複数候補があれば一様乱択)
  - per-game seed 基点 = 20260713 (連番)、局内乱択の RNG seed = 713
  - ゲーム(=半荘) -> 局(=kyoku) の走査順で規定 N に達するまで採取

差し戻し修正 (監督側の独立再実行で判明): GameplayLoader の事後再構築による
(obs,mask) 席解決は実 react_batch クエリ列と 1:1 対応しない (at_kyoku>=1 で
既知の loader size delta と同種の divergence)。本スクリプトは両 engine に
記録専用パススルーラッパー (RecordingPassthroughEngine) を噛ませ、実クエリ
列そのものを game_key 別 (mask_obs_digest, action) の順序付きリストとして
記録する。これが drca_run_probe.py の台本再生で実際に消費される「真の」
クエリ列になる。GameplayLoader は分岐点のメタデータ (at_kyoku/shanten/
at_turn/点況順位) 抽出にのみ使う (候補判定・報酬計算も同様、既存どおり)。

出力は jsonl (分岐点メタデータ) + 選択された game_key ごとの台本 sidecar
jsonl (`<game_key>.script.jsonl`、各行 {"digest":..., "action":...})。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drca_common import (  # noqa: E402
    RecordingPassthroughEngine,
    assert_construction_diff_empty,
    build_policy_engine,
    call_possible_aka_held,
    kyoku_start_scores,
    legal_call_action_ids,
    make_arena,
    mask_obs_digest,
    reconstruct_all_seats,
    score_rank,
    write_script_sidecar,
)
from config import config  # noqa: E402

# Not part of the pre-registered protocol (only the `seed` component of
# seed_start is pre-registered in §5a-4); a fixed constant so runs stay
# reproducible. Distinct from other eval batteries' 0x2000 to avoid any
# accidental wall reuse across scripts (harmless either way, since (seed,
# key) as a whole determines the wall, but kept distinct for hygiene).
DEFAULT_SEED_KEY = 0x44524341  # ASCII 'DRCA'

SPLITS = ['a', 'b', 'c', 'd']


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stdout,
    )
    return logging.getLogger('drca_collect')


def collect_kyoku_candidates(seats, kyoku: int, *, challenger_seat_only: int | None = None):
    """All (seat, local_idx) branch-point candidates within one kyoku,
    across all 4 seats, in seat-ascending / local-idx-ascending order.

    If `challenger_seat_only` is given (a seat index), candidates are
    restricted to that seat -- used for set(b) collection where mode=b
    replay requires the branch point's physical seat to be the fixed
    challenger seat for that split (one_vs_three.rs, drca_run_probe.py
    mode-b assert).
    """
    out = []
    for seat in range(4):
        if challenger_seat_only is not None and seat != challenger_seat_only:
            continue
        sg = seats[seat]
        for local_idx, (obs, mask, ak) in enumerate(zip(sg.obs, sg.masks, sg.at_kyoku)):
            if ak != kyoku:
                continue
            if call_possible_aka_held(obs, mask):
                out.append((seat, local_idx))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, help='被測定 checkpoint (state_file)')
    ap.add_argument('--n', type=int, required=True, help='採取する分岐点の数')
    ap.add_argument('--out', required=True, help='出力 jsonl パス')
    ap.add_argument('--log-dir', default=None, help='json.gz ログの保存先 (default: <out>と同階層の<out>.logs/)')
    ap.add_argument('--seed-base', type=int, default=20260713, help='per-game seed 基点 (§5a-4)')
    ap.add_argument('--seed-key', type=int, default=DEFAULT_SEED_KEY)
    ap.add_argument('--extract-seed', type=int, default=713, help='局内乱択 RNG seed (§5a-4)')
    ap.add_argument('--max-seeds', type=int, default=5000, help='安全弁: この数の seed を使い切っても N 未達なら FATAL')
    ap.add_argument('--torch-seed', type=int, default=20260713,
                     help='採取 π サンプリングの再現性 seed (差し戻し要件4)。'
                          '同一引数なら同一分岐点セットが出ることの根拠')
    ap.add_argument('--challenger-seat-only', action='store_true',
                     help='候補をこの split の challenger 席 (SPLITS.index(split)) のみに'
                          '限定する (セット(b) 採取用、差し戻し要件2)')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    log = setup_logging()
    torch.manual_seed(args.torch_seed)
    device = torch.device(args.device or config['control']['device'])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else out_path.parent / f'{out_path.stem}.logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log.info('checkpoint=%s n=%s seed_base=%s seed_key=%#x extract_seed=%s torch_seed=%s '
              'challenger_seat_only=%s',
              args.checkpoint, args.n, args.seed_base, args.seed_key, args.extract_seed,
              args.torch_seed, args.challenger_seat_only)

    engine_chal, steps_chal, dump_chal = build_policy_engine(args.checkpoint, device, name='drca_collect_a')
    engine_champ, steps_champ, dump_champ = build_policy_engine(args.checkpoint, device, name='drca_collect_b')
    assert_construction_diff_empty(dump_chal, dump_champ)
    log.info('engine config dump (challenger): %s', dump_chal)
    log.info('engine config dump (champion):   %s', dump_champ)
    assert steps_chal == steps_champ, 'both self-play seats must load the same checkpoint step count'

    version = config['control']['version']
    rng = random.Random(args.extract_seed)
    arena = make_arena(log_dir)

    n_collected = 0
    n_kyoku_scanned = 0
    n_kyoku_with_candidate = 0
    n_at_kyoku_ge1 = 0

    with out_path.open('w', encoding='utf-8') as fout:
        for offset in range(args.max_seeds):
            if n_collected >= args.n:
                break
            seed = args.seed_base + offset
            # Fresh per-seed store: both recording wrappers append into the
            # SAME shared dict, so entries for a given game_key preserve the
            # true cross-seat dispatch order regardless of which of the two
            # Rust-level agent objects (challenger/champion) is queried.
            script_store: dict[str, list[tuple[str, int]]] = {}
            wrapped_chal = RecordingPassthroughEngine(engine_chal, script_store, name='drca_collect_a_rec')
            wrapped_champ = RecordingPassthroughEngine(engine_champ, script_store, name='drca_collect_b_rec')
            arena.py_vs_py(
                challenger=wrapped_chal,
                champion=wrapped_champ,
                seed_start=(seed, args.seed_key),
                seed_count=1,
            )
            for split in SPLITS:
                if n_collected >= args.n:
                    break
                game_key = f'{seed}_{args.seed_key}_{split}'
                log_path = log_dir / f'{game_key}.json.gz'
                if not log_path.is_file():
                    raise RuntimeError(f'expected log missing: {log_path}')
                queue = script_store.get(game_key, [])
                if not queue:
                    raise RuntimeError(
                        f'no recorded react_batch queries for game_key={game_key} '
                        '-- recording wrapper / step_meta wiring is broken'
                    )
                seats = reconstruct_all_seats(str(log_path), version)
                kyoku_values = sorted({ak for sg in seats.values() for ak in sg.at_kyoku})
                chal_seat = SPLITS.index(split) if args.challenger_seat_only else None
                scores_cache = None
                script_written = False
                for kyoku in kyoku_values:
                    if n_collected >= args.n:
                        break
                    candidates = collect_kyoku_candidates(seats, kyoku, challenger_seat_only=chal_seat)
                    n_kyoku_scanned += 1
                    if not candidates:
                        continue
                    n_kyoku_with_candidate += 1
                    seat, local_idx = rng.choice(candidates)
                    sg = seats[seat]
                    obs = sg.obs[local_idx]
                    mask = sg.masks[local_idx]
                    digest = mask_obs_digest(obs, mask)

                    # Bridge GameplayLoader-selected candidate (per-seat
                    # semantics, used for metadata only) to its position in
                    # the true recorded query stream, by digest. Exactly one
                    # match is required (drca_probe_design.md differential
                    # fix item 1) -- 0 or 2+ is a loud FAIL, not silently
                    # resolved.
                    matches = [i for i, (d, _a) in enumerate(queue) if d == digest]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f'branch point digest matched {len(matches)} script entries '
                            f'(expected exactly 1) for game_key={game_key} seat={seat} '
                            f'local_idx={local_idx} at_kyoku={kyoku}: matches={matches}'
                        )
                    script_index = matches[0]

                    script_path = log_dir / f'{game_key}.script.jsonl'
                    if not script_written:
                        write_script_sidecar(script_path, queue)
                        script_written = True

                    if scores_cache is None:
                        scores_cache = kyoku_start_scores(str(log_path), version)
                    if kyoku >= scores_cache.shape[0]:
                        raise RuntimeError(
                            f'kyoku {kyoku} beyond scores_cache length {scores_cache.shape[0]} '
                            f'for game_key={game_key}'
                        )
                    rank = score_rank(scores_cache[kyoku], seat)
                    if kyoku >= 1:
                        n_at_kyoku_ge1 += 1

                    record = {
                        'collection_checkpoint': str(args.checkpoint),
                        'collection_steps': steps_chal,
                        'version': version,
                        'seed': seed,
                        'key': args.seed_key,
                        'split': split,
                        'game_key': game_key,
                        'game_log_path': str(log_path.resolve()),
                        'script_path': str(script_path.resolve()),
                        'script_index': script_index,
                        'script_len': len(queue),
                        'seat': seat,
                        'seat_local_index': local_idx,
                        'at_kyoku': int(sg.at_kyoku[local_idx]),
                        'at_turn': int(sg.at_turns[local_idx]),
                        'shanten': int(sg.shantens[local_idx]),
                        'action_taken_originally': int(sg.actions[local_idx]),
                        'call_types_available': legal_call_action_ids(mask),
                        'n_candidates_in_kyoku': len(candidates),
                        'mask_obs_digest': digest,
                        'score_rank_at_branch': rank,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                    fout.flush()
                    n_collected += 1
                    log.info(
                        'branch %d/%d: game_key=%s seat=%d local_idx=%d at_kyoku=%d '
                        'shanten=%d n_candidates=%d script_index=%d/%d score_rank=%d',
                        n_collected, args.n, game_key, seat, local_idx,
                        record['at_kyoku'], record['shanten'], len(candidates),
                        script_index, len(queue), rank,
                    )

        if n_collected < args.n:
            raise RuntimeError(
                f'exhausted max_seeds={args.max_seeds} with only '
                f'{n_collected}/{args.n} branch points collected'
            )

    log.info(
        'DONE: collected=%d/%d kyoku_scanned=%d kyoku_with_candidate=%d (%.2f%%) '
        'at_kyoku>=1=%d out=%s log_dir=%s',
        n_collected, args.n, n_kyoku_scanned, n_kyoku_with_candidate,
        100.0 * n_kyoku_with_candidate / max(n_kyoku_scanned, 1),
        n_at_kyoku_ge1, out_path, log_dir,
    )


if __name__ == '__main__':
    main()
