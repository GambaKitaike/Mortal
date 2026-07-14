#!/usr/bin/env python3
"""DRCA プローブ: fork-by-replay 並走ロールアウト (drca_probe_design.md §2/§3/§5a)。

分岐点 jsonl (drca_collect_branchpoints.py の出力) の各行について、同一
seed (= 同一山、libriichi の wall RNG は shuffle 直後に破棄される決定論的
挙動 -- board.rs 確認済み・drca_probe_design.md §2-1) を再走し、採取時に
記録された「真の」react_batch クエリ列 (`<game_key>.script.jsonl`
sidecar、drca_collect_branchpoints.py が RecordingPassthroughEngine で
記録したもの) を分岐点まで台本として再生する (推論ゼロ)。分岐点で:
  - call 腕: legal な鳴き選択肢に π を制限して再正規化サンプリング
  - no-call 腕: legal な非鳴き選択肢 (合法 mask ∧ ¬[38..42]) に π を制限して
    再正規化サンプリング (call 腕と対称。2026-07-13 amendment、
    drca_probe_design.md §1)
以降は全席、実チェックポイントによる素の π サンプリングで局終了まで進む。

差し戻し修正1 (監督側の独立再実行で判明): GameplayLoader 事後再構築 + 席解決
(obs,mask 全席探索) は実クエリ列と 1:1 対応しなかった (at_kyoku>=1 で
divergence)。本スクリプトはもう GameplayLoader を台本の情報源として使わず、
sidecar に記録された (role, seq, digest, action) を消費する。GameplayLoader
は分岐点のメタデータ (at_kyoku/shanten/at_turn) と最終報酬計算にのみ使う
(drca_collect_branchpoints.py と同じ用途限定)。

差し戻し修正2 (2度目、監督側の独立再実行で判明): sidecar を単一 FIFO として
到着順に消費していたが、react_batch へのバッチ到着順は rayon::spawn の完了順
に左右され実行ごとに変動する (libriichi/src/agent/mortal.rs) ため、到着順
基準の FIFO index は再現性を持たない。台本再生は到着順に依存しない
(role, seq) バケット + digest 照合に置き換えた: 各クエリはそれを発した
ScriptedForkEngine の role ('challenger'/'champion') と step_meta の seq
(スロット単位の決定論的連番) でバケットを引き、そのバケット内で digest が
一致するエントリが厳密1件であることを assert して消費する (0件・複数件は
loud FAIL、r2 guard の新形態)。challenger は 1 game に 1 slot のみなので
(role, seq) だけで一意に定まるが、champion は 1 game に 3 slot 持ち各 slot
の seq が独立に 0 起算のため同一 (role, seq) に最大3エントリが集まり得る --
digest がその中から実際に問い合わせられた1件を特定する。

新規 Rust 表面はゼロ。台本再生は純 Python ラッパー engine
(ScriptedForkEngine) が react_batch を横取りして実現する。

set(a): 全4席とも被測定 checkpoint で続行。
set(b): 分岐点席のみ被測定 checkpoint、他3席は --reference-checkpoint
  (drca_probe_design.md §5a-3 では Stage1 step_016000 固定)。分岐点の
  物理席は必ず split の challenger 席 (one_vs_three.rs 固定写像) でなければ
  ならない -- 起動時に全分岐点を assert する (差し戻し要件2)。

各ロールアウトは実際には4スプリット (a/b/c/d) 全てが同時に再生されるが
(OneVsThree の固定 API 制約)、分岐点の game_key と一致するスプリットのみを使う。

決定論性の壊れ (r2) は「台本再生中に問い合わせが来た (role, seq, digest) が
対応するバケット内のエントリと厳密1件で一致しない」場合に即座に loud FAIL
する (--inject-fault-for-test でこの経路自体を実演できる)。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drca_common import (  # noqa: E402
    CALL_ACTION_MAX,
    CALL_ACTION_MIN,
    assert_construction_diff_empty,
    build_policy_engine,
    call_possible_aka_held,
    compute_kyoku_rewards,
    find_unique_role_seq_match,
    legal_call_action_ids,
    load_script_sidecar,
    make_arena,
    mask_obs_digest,
    parse_game_key,
    reconstruct_all_seats,
)
from config import config  # noqa: E402

SPLITS = ['a', 'b', 'c', 'd']


def branch_identity(branch: dict) -> tuple:
    return (branch['game_key'], branch['branch_role'], branch['branch_seq'])


def branch_identity_from_record(record: dict) -> tuple:
    return (record['game_key'], record['branch_role'], record['branch_seq'])


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stdout,
    )
    return logging.getLogger('drca_probe')


class ForkState:
    """Shared mutable replay state for one (branch point, rollout, arm)
    fork. Both the challenger-role and champion-role ScriptedForkEngine
    instances hold a reference to the same ForkState, so decisions from
    any of the 4 physical seats of the *target* game_key are resolved
    against one shared script, regardless of which of the two Rust-level
    agent objects (challenger / champion) happens to be asked.

    The script is `script_buckets: dict[role] -> dict[seq] -> list[(digest,
    action)]`, recorded live during collection
    (drca_collect_branchpoints.py's RecordingPassthroughEngine), keyed by
    (role, seq) rather than arrival order -- react_batch batch-arrival
    order is not reproducible across runs (rayon::spawn completion order
    in libriichi/src/agent/mortal.rs), but (role, seq) is deterministic
    per physical slot regardless of scheduling. Each query is matched to
    its recorded entry by (role, seq) bucket + exact digest equality
    within that bucket (challenger has 1 slot/game so (role, seq) alone is
    unique; champion has 3 slots/game with independently 0-based seq
    counters, so a bucket can hold up to 3 entries, disambiguated by
    digest).
    """

    def __init__(self, *, target_game_key, script_buckets, branch_role, branch_seq,
                 branch_digest, arm, inject_fault=False):
        self.target_game_key = target_game_key
        self.script_buckets = script_buckets  # dict[role] -> dict[seq] -> list[(digest, action)]
        self.branch_role = branch_role
        self.branch_seq = branch_seq
        self.branch_digest = branch_digest
        self.arm = arm
        self.crossed_branch = False
        self.branch_served = False
        self.inject_fault = inject_fault
        self._fault_injected = False
        self.n_scripted_served = 0
        self.n_live_served = 0
        self.branch_forced_action = None
        self.branch_call_options = None

    def next_action_for(self, role: str, seq: int, digest: str):
        """Resolve one query against the recorded script.

        Returns the recorded action, or None if this decision *is* the
        flagged branch point (caller computes the forced/live branch
        action). Raises loudly (no silent fallback) if the (role, seq)
        bucket is missing or empty, or if digest doesn't match exactly one
        entry in that bucket -- the r2 seed-determinism /
        prefix-replay-integrity guard, now order-independent.
        """
        is_branch = (
            not self.branch_served
            and role == self.branch_role
            and seq == self.branch_seq
            and digest == self.branch_digest
        )
        if is_branch:
            self.branch_served = True
            self.crossed_branch = True
            return None

        bucket = self.script_buckets.get(role, {}).get(seq)
        if not bucket:
            raise RuntimeError(
                f'DRCA replay divergence in game_key={self.target_game_key}: '
                f'no recorded script entries for role={role!r} seq={seq} -- '
                'prefix determinism broken (bucket missing/empty)'
            )
        matches = [i for i, (d, _a) in enumerate(bucket) if d == digest]
        if len(matches) != 1:
            raise RuntimeError(
                f'DRCA replay divergence in game_key={self.target_game_key}: '
                f'query digest {digest} matched {len(matches)} entries in '
                f'role={role!r} seq={seq} bucket (expected exactly 1) -- this '
                'means the replayed wall/action prefix no longer matches the '
                'recorded script (seed-determinism assert failed)'
            )
        idx = matches[0]
        exp_digest, exp_action = bucket.pop(idx)
        assert exp_digest == digest

        if self.inject_fault and not self._fault_injected:
            self._fault_injected = True
            faulted = 0 if exp_action != 0 else 1
            logging.getLogger('drca_probe').warning(
                'FAULT INJECTED (--inject-fault-for-test): role=%s seq=%d '
                'recorded_action=%d -> returning %d instead',
                role, seq, exp_action, faulted,
            )
            return faulted
        self.n_scripted_served += 1
        return exp_action


class ScriptedForkEngine:
    """react_batch wrapper: scripted prefix -> forced branch decision ->
    live delegation, shared via `fork_state` across the challenger-role
    and champion-role instances of one arena call.
    """

    engine_type = 'mortal'

    def __init__(self, *, role: str, fork_state: ForkState, live_engine, brain, actor_critic,
                 device, engine_lock: threading.Lock):
        self.role = role
        self.fork_state = fork_state
        self.live_engine = live_engine
        self.brain = brain
        self.actor_critic = actor_critic
        self.device = device
        self.engine_lock = engine_lock
        self.name = f'drca_fork_{role}'
        self.is_oracle = live_engine.is_oracle
        self.version = live_engine.version
        self.enable_quick_eval = live_engine.enable_quick_eval
        self.enable_rule_based_agari_guard = live_engine.enable_rule_based_agari_guard
        # p_enrich intentionally left undefined -- OneVsThree::py_vs_py
        # falls back to 0.0 via getattr (one_vs_three.rs), same as every
        # other eval/opponent engine in this codebase.

    def _forward_logits(self, obs_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
        with self.engine_lock:
            with torch.inference_mode():
                match self.version:
                    case 1:
                        mu, _logsig = self.brain(obs_t, None)
                        phi = mu
                    case 2 | 3 | 4:
                        phi = self.brain(obs_t)
                logits, _values = self.actor_critic(phi, mask_t)
        return logits

    def _sample_branch_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """Live decision for the forced branch step (drca_probe_design.md
        §1, 2026-07-13 amendment). call arm: restrict pi to the legal call
        subset [38,42] and renormalize-sample (preserves the policy's
        relative preference among call options). no-call arm: restrict pi
        to the legal non-call subset (legal mask & ~[38,42]) and
        renormalize-sample -- symmetric to the call arm. This reduces to
        forcing DECLINE_ACTION(45) in the ordinary reaction case (pass is
        the only legal non-call action there), but also covers own-turn
        ankan/kakan opportunities (no pass action exists on the mask; the
        original definition was inapplicable there per the amendment) and
        ron-eligible reactions (ron is legal and non-call, so it is not
        excluded -- ron is not counted against the call's opportunity cost).
        """
        mask_arr = np.asarray(mask)
        arm = self.fork_state.arm
        obs_t = torch.as_tensor(np.asarray(obs)[None], device=self.device)
        mask_t = torch.as_tensor(mask_arr[None], device=self.device, dtype=torch.bool)
        logits = self._forward_logits(obs_t, mask_t)[0]

        if arm == 'no_call':
            restricted_mask = mask_t[0].clone()
            restricted_mask[CALL_ACTION_MIN:CALL_ACTION_MAX + 1] = False
            assert bool(restricted_mask.any()), (
                'no-call arm requires at least one legal non-call action in the mask, '
                'but the mask has none outside the call action range '
                f'[{CALL_ACTION_MIN},{CALL_ACTION_MAX}] at the recorded branch point'
            )
            restricted = logits.float().masked_fill(~restricted_mask, -1e9)
            action = int(Categorical(logits=restricted).sample().item())
            assert bool(restricted_mask[action]), (
                f'sampled no-call action {action} not in legal non-call subset'
            )
            return action

        assert arm == 'call', f'unknown arm {arm!r}'
        call_ids = legal_call_action_ids(mask_arr)
        assert call_ids, 'call arm requires at least one legal call action in the mask'
        call_mask = torch.zeros_like(mask_t[0])
        for cid in call_ids:
            call_mask[cid] = True
        restricted = logits.float().masked_fill(~call_mask, -1e9)
        action = int(Categorical(logits=restricted).sample().item())
        assert action in call_ids, f'sampled action {action} not in legal call subset {call_ids}'
        return action

    def react_batch(self, obs, masks, invisible_obs, step_meta=None):
        obs_np = [np.asarray(o) for o in obs]
        masks_np = [np.asarray(m) for m in masks]
        batch_size = len(obs_np)

        actions = [None] * batch_size
        live_idx = []

        for i in range(batch_size):
            meta = step_meta[i] if step_meta is not None and len(step_meta) == batch_size else None
            game_key = meta[0] if meta else None
            if game_key != self.fork_state.target_game_key or self.fork_state.crossed_branch:
                live_idx.append(i)
                continue
            seq = int(meta[1])
            digest = mask_obs_digest(obs_np[i], masks_np[i])
            action = self.fork_state.next_action_for(self.role, seq, digest)
            if action is None:
                # this is the flagged branch decision itself
                action = self._sample_branch_action(obs_np[i], masks_np[i])
                self.fork_state.branch_forced_action = action
                self.fork_state.branch_call_options = legal_call_action_ids(masks_np[i])
                assert bool(masks_np[i][action]), 'forced branch action must be legal per mask'
            actions[i] = action

        if live_idx:
            live_obs = [obs[i] for i in live_idx]
            live_masks = [masks[i] for i in live_idx]
            live_invisible = None
            if invisible_obs is not None:
                live_invisible = [invisible_obs[i] for i in live_idx]
            live_step_meta = None
            if step_meta is not None and len(step_meta) == batch_size:
                live_step_meta = [step_meta[i] for i in live_idx]
            with self.engine_lock:
                live_actions, _q, _m, _g = self.live_engine.react_batch(
                    live_obs, live_masks, live_invisible, live_step_meta,
                )
            for j, i in enumerate(live_idx):
                actions[i] = live_actions[j]
            self.fork_state.n_live_served += len(live_idx)

        q_values = [[0.0] * len(m) for m in masks_np]
        is_greedy = [False] * batch_size
        return actions, q_values, [m.tolist() for m in masks_np], is_greedy


def run_one_rollout(branch, arm, chal_engine, champ_engine, device, tmp_root, *,
                     engine_lock, inject_fault=False, log=None):
    """chal_engine / champ_engine are pre-built (main() constructs them once
    and reuses them across all rollouts, differential-fix item 3 -- only the
    ForkState/wrapper objects below are created fresh per rollout, since
    those carry per-rollout mutable replay-consumption state).
    """
    script_buckets = load_script_sidecar(branch['script_path'])
    branch_role = branch['branch_role']
    branch_seq = branch['branch_seq']
    branch_digest = branch['mask_obs_digest']

    # Re-derive the (role, seq) match independently from the full sidecar
    # (not just the recorded branch_role/branch_seq) and assert it agrees --
    # catches sidecar corruption / index drift the same way
    # drca_collect_branchpoints.py's find_unique_role_seq_match does at
    # collection time.
    found_role, found_seq = find_unique_role_seq_match(script_buckets, branch_digest)
    if (found_role, found_seq) != (branch_role, branch_seq):
        raise RuntimeError(
            f'branch point script sidecar mismatch for game_key={branch["game_key"]}: '
            f'recorded branch_role/branch_seq=({branch_role!r}, {branch_seq}) but '
            f're-derivation from sidecar found ({found_role!r}, {found_seq}) '
            '(sidecar corrupted or branch metadata drift)'
        )

    # GameplayLoader used only for metadata / sel-definition-drift check
    # (drca_probe_design.md differential fix item 1) -- NOT as the replay
    # script source (that's script_buckets above, recorded live at collection).
    target_seat = branch['seat']
    target_local_index = branch['seat_local_index']
    seats = reconstruct_all_seats(branch['game_log_path'], branch['version'])
    sg = seats[target_seat]
    obs0, mask0 = sg.obs[target_local_index], sg.masks[target_local_index]
    if not call_possible_aka_held(obs0, mask0):
        raise RuntimeError(
            f'branch point re-derivation for {branch["game_key"]} no longer satisfies '
            'call_possible & aka_held -- sel-definition drift or corrupted branch point'
        )

    fork_state = ForkState(
        target_game_key=branch['game_key'],
        script_buckets=script_buckets,
        branch_role=branch_role,
        branch_seq=branch_seq,
        branch_digest=branch_digest,
        arm=arm,
        inject_fault=inject_fault,
    )

    chal_wrapper = ScriptedForkEngine(
        role='challenger', fork_state=fork_state, live_engine=chal_engine,
        brain=chal_engine.brain, actor_critic=chal_engine.actor_critic, device=device,
        engine_lock=engine_lock,
    )
    champ_wrapper = ScriptedForkEngine(
        role='champion', fork_state=fork_state, live_engine=champ_engine,
        brain=champ_engine.brain, actor_critic=champ_engine.actor_critic, device=device,
        engine_lock=engine_lock,
    )

    with tempfile.TemporaryDirectory(prefix='drca_probe_', dir=str(tmp_root)) as tmp:
        arena = make_arena(tmp)
        arena.py_vs_py(
            challenger=chal_wrapper,
            champion=champ_wrapper,
            seed_start=(branch['seed'], branch['key']),
            seed_count=1,
        )
        if not fork_state.branch_served:
            raise RuntimeError(
                f'branch point for game_key={branch["game_key"]} was never reached during '
                'replay (prefix determinism broken -- r2)'
            )
        result_log = Path(tmp) / f'{branch["game_key"]}.json.gz'
        if not result_log.is_file():
            raise RuntimeError(f'expected replay log missing: {result_log}')

        rewards = compute_kyoku_rewards(str(result_log), target_seat, device=device)
        at_kyoku = branch['at_kyoku']
        n_kyoku = len(rewards.reward_sotensu)
        if at_kyoku >= n_kyoku:
            raise RuntimeError(
                f'branch kyoku {at_kyoku} beyond replayed hanchan length {n_kyoku} '
                f'for game_key={branch["game_key"]}'
            )
        reward_primary = (
            rewards.reward_sotensu[at_kyoku]
            + rewards.reward_grp[at_kyoku]
            + rewards.reward_chip[at_kyoku]
        )

        out = {
            'arm': arm,
            'forced_action': fork_state.branch_forced_action,
            'branch_call_options': fork_state.branch_call_options,
            'n_scripted_served': fork_state.n_scripted_served,
            'n_live_served': fork_state.n_live_served,
            'reward_primary': reward_primary,
            'reward_sotensu_kyoku': rewards.reward_sotensu[at_kyoku],
            'reward_grp_kyoku': rewards.reward_grp[at_kyoku],
            'reward_chip_kyoku': rewards.reward_chip[at_kyoku],
        }
        if log:
            log.debug('rollout arm=%s forced_action=%s reward_primary=%.4f',
                       arm, fork_state.branch_forced_action, reward_primary)
        return out, rewards


def maybe_full_hanchan_fields(rewards) -> dict:
    target_rank = rewards.rank_by_player
    return {
        'final_rank_by_player': target_rank,
        'final_scores': rewards.final_scores,
        'total_chip_delta': float(sum(rewards.chip_deltas)),
    }


def load_existing_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_complete_branch_records(records: list[dict], k: int) -> tuple[list[dict], int, int]:
    """Keep only branch points with exactly K call + K no_call rollouts."""
    grouped: dict[tuple, dict[str, list[dict]]] = {}
    for r in records:
        key = branch_identity_from_record(r)
        grouped.setdefault(key, {'call': [], 'no_call': []})[r['arm']].append(r)

    kept = []
    n_complete = 0
    n_discarded = 0
    for _key, arms in grouped.items():
        n_call = len(arms.get('call', []))
        n_nocall = len(arms.get('no_call', []))
        if n_call == k and n_nocall == k:
            kept.extend(arms['call'])
            kept.extend(arms['no_call'])
            n_complete += 1
        else:
            n_discarded += 1
    return kept, n_complete, n_discarded


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--branchpoints', required=True)
    ap.add_argument('--checkpoint', required=True, help='被測定 checkpoint')
    ap.add_argument('--mode', choices=['a', 'b'], required=True)
    ap.add_argument('--reference-checkpoint', default=None,
                     help='mode=b で他3席に使う参照方策 (§5a-3: Stage1 step_016000)')
    ap.add_argument('--k', type=int, default=8, help='腕あたりロールアウト数')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None, help='処理する分岐点数の上限 (smoke test 用)')
    ap.add_argument('--parallel', type=int, default=1,
                     help='ロールアウト並列ワーカー数 (default 1 = 直列)')
    ap.add_argument('--resume', action='store_true',
                     help='--out が既存なら完走済み分岐点のみ保持し残りを処理')
    ap.add_argument('--full-hanchan', action='store_true')
    ap.add_argument('--inject-fault-for-test', action='store_true',
                     help='デバッグ専用: 台本再生中に1アクションを故意に破損させ、'
                          'r2 の assert が実際に発火することを実演する')
    ap.add_argument('--device', default=None)
    ap.add_argument('--tmp-root', default=None)
    args = ap.parse_args()

    if args.mode == 'b' and not args.reference_checkpoint:
        ap.error('--mode b requires --reference-checkpoint')
    if args.parallel < 1:
        ap.error('--parallel must be >= 1')

    log = setup_logging()
    device = torch.device(args.device or config['control']['device'])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(args.tmp_root) if args.tmp_root else out_path.parent / f'{out_path.stem}.tmp'
    tmp_root.mkdir(parents=True, exist_ok=True)

    branches = []
    with open(args.branchpoints, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                branches.append(json.loads(line))
    if args.limit:
        branches = branches[:args.limit]
    log.info('loaded %d branch points from %s (processing %d)',
              sum(1 for _ in open(args.branchpoints, encoding='utf-8')), args.branchpoints, len(branches))

    if args.mode == 'b':
        # set(b): branch point's physical seat must be the fixed challenger
        # seat for its split (one_vs_three.rs), since only that seat gets
        # replaced by the probed checkpoint -- differential-fix item 2.
        for branch in branches:
            _seed, _key, split = parse_game_key(branch['game_key'])
            expected_seat = SPLITS.index(split)
            if branch['seat'] != expected_seat:
                raise RuntimeError(
                    f'mode=b requires branch point seat to be the challenger seat '
                    f'for its split: game_key={branch["game_key"]} split={split} '
                    f'expected_seat={expected_seat} got seat={branch["seat"]} '
                    '(use --challenger-seat-only at collection time for set(b))'
                )
        log.info('mode=b challenger-seat assert passed for all %d branch points', len(branches))

    kept_records: list[dict] = []
    completed_keys: set[tuple] = set()
    if args.resume and out_path.is_file():
        existing = load_existing_records(out_path)
        kept_records, n_kept_branches, n_discarded_branches = filter_complete_branch_records(
            existing, args.k,
        )
        grouped = {}
        for r in kept_records:
            grouped.setdefault(branch_identity_from_record(r), []).append(r)
        completed_keys = set(grouped.keys())
        log.info(
            '--resume: loaded %d existing records from %s; kept %d complete branch points '
            '(%d records), discarded %d incomplete branch points',
            len(existing), out_path, len(completed_keys), len(kept_records),
            n_discarded_branches,
        )
        with out_path.open('w', encoding='utf-8') as fout:
            for record in kept_records:
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    elif args.resume:
        log.info('--resume: no existing output at %s, starting fresh', out_path)

    pending_branches = []
    for bi, branch in enumerate(branches):
        if branch_identity(branch) not in completed_keys:
            pending_branches.append((bi, branch))
    log.info(
        'branch points to process: %d pending / %d total (%d already complete)',
        len(pending_branches), len(branches), len(completed_keys),
    )

    # Engine construction, once up front (differential-fix item 3 -- reused
    # across all rollouts below; only ForkState/wrapper objects are built
    # per rollout, since those carry per-rollout mutable replay state).
    # `別インスタンス必須` (player.py convention): challenger and champion
    # must be distinct PyObjects even when they share weights (set a).
    chal_ckpt = args.checkpoint
    champ_ckpt = args.checkpoint if args.mode == 'a' else args.reference_checkpoint
    chal_engine, _s1, dump_chal = build_policy_engine(chal_ckpt, device, name='drca_probe_chal')
    champ_engine, _s2, dump_champ = build_policy_engine(champ_ckpt, device, name='drca_probe_champ')
    chal_engine.record_trajectory = False
    champ_engine.record_trajectory = False
    if args.mode == 'a':
        assert_construction_diff_empty(dump_chal, dump_champ)
    log.info('probed checkpoint chal engine config dump: %s', dump_chal)
    log.info('probed checkpoint champ engine config dump: %s', dump_champ)
    log.info(
        'record_trajectory=False set on both live engines '
        '(probe never drains pending_by_game; avoids unbounded RAM at measurement scale)'
    )
    assert dump_chal['p_enrich'] == 0.0 and dump_chal['call_bonus_b'] == 0.0
    assert dump_champ['p_enrich'] == 0.0 and dump_champ['call_bonus_b'] == 0.0

    engine_lock = threading.Lock()
    write_lock = threading.Lock()

    tasks = []
    for bi, branch in pending_branches:
        for arm in ('call', 'no_call'):
            for k in range(args.k):
                inject = args.inject_fault_for_test and bi == 0 and arm == 'call' and k == 0
                tasks.append((bi, branch, arm, k, inject))

    log.info(
        'scheduling %d rollouts (%d branch points × 2 arms × K=%d) with --parallel=%d',
        len(tasks), len(pending_branches), args.k, args.parallel,
    )

    def run_task(task):
        bi, branch, arm, k, inject = task
        result, rewards = run_one_rollout(
            branch, arm, chal_engine, champ_engine,
            device, tmp_root, engine_lock=engine_lock, inject_fault=inject, log=log,
        )
        record = {
            'branch_index': bi,
            'game_key': branch['game_key'],
            'branch_role': branch['branch_role'],
            'branch_seq': branch['branch_seq'],
            'seat': branch['seat'],
            'at_kyoku': branch['at_kyoku'],
            'at_turn': branch['at_turn'],
            'shanten': branch['shanten'],
            'call_types_available': branch['call_types_available'],
            'score_rank_at_branch': branch.get('score_rank_at_branch'),
            'rollout_index': k,
            'mode': args.mode,
            'probed_checkpoint': args.checkpoint,
            'reference_checkpoint': args.reference_checkpoint,
            **result,
        }
        if args.full_hanchan:
            record.update(maybe_full_hanchan_fields(rewards))
        return bi, branch, arm, k, result, record

    n_out = len(kept_records)
    t0 = time.perf_counter()
    n_done = 0

    open_mode = 'a' if args.resume and kept_records else 'w'
    with out_path.open(open_mode, encoding='utf-8') as fout:
        if args.parallel == 1:
            for task in tasks:
                bi, branch, arm, k, result, record = run_task(task)
                with write_lock:
                    fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                    fout.flush()
                n_out += 1
                n_done += 1
                log.info(
                    '  branch %d/%d arm=%s k=%d forced_action=%s reward_primary=%.4f '
                    'n_scripted=%d n_live=%d (%d/%d rollouts)',
                    bi + 1, len(branches), arm, k, result['forced_action'],
                    result['reward_primary'], result['n_scripted_served'],
                    result['n_live_served'], n_done, len(tasks),
                )
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {pool.submit(run_task, task): task for task in tasks}
                for fut in as_completed(futures):
                    bi, branch, arm, k, result, record = fut.result()
                    with write_lock:
                        fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                        fout.flush()
                    n_out += 1
                    n_done += 1
                    log.info(
                        '  branch %d/%d arm=%s k=%d forced_action=%s reward_primary=%.4f '
                        'n_scripted=%d n_live=%d (%d/%d rollouts)',
                        bi + 1, len(branches), arm, k, result['forced_action'],
                        result['reward_primary'], result['n_scripted_served'],
                        result['n_live_served'], n_done, len(tasks),
                    )

    elapsed = time.perf_counter() - t0
    new_rollouts = len(tasks)
    if new_rollouts > 0:
        rollouts_per_h = new_rollouts / elapsed * 3600.0
        log.info(
            'timing: %d new rollouts in %.1fs (%.2f rollouts/h)',
            new_rollouts, elapsed, rollouts_per_h,
        )

    assert chal_engine.illegal_action_fallback_count == 0, (
        f'chal_engine illegal_action_fallback_count={chal_engine.illegal_action_fallback_count}'
    )
    assert champ_engine.illegal_action_fallback_count == 0, (
        f'champ_engine illegal_action_fallback_count={champ_engine.illegal_action_fallback_count}'
    )
    log.info(
        'illegal_action_fallback_count=0 on both live engines (chal=%d champ=%d)',
        chal_engine.illegal_action_fallback_count, champ_engine.illegal_action_fallback_count,
    )
    log.info('DONE: %d total rollout records in %s', n_out, out_path)


if __name__ == '__main__':
    main()
