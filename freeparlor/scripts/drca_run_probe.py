#!/usr/bin/env python3
"""DRCA プローブ: fork-by-replay 並走ロールアウト (drca_probe_design.md §2/§3/§5a)。

分岐点 jsonl (drca_collect_branchpoints.py の出力) の各行について、同一
seed (= 同一山、libriichi の wall RNG は shuffle 直後に破棄される決定論的
挙動 -- board.rs 確認済み・drca_probe_design.md §2-1) を再走し、採取時に
記録された「真の」react_batch クエリ列 (`<game_key>.script.jsonl`
sidecar、drca_collect_branchpoints.py が RecordingPassthroughEngine で
記録したもの) を分岐点まで台本として再生する (推論ゼロ)。分岐点で:
  - call 腕: legal な鳴き選択肢に π を制限して再正規化サンプリング
  - no-call 腕: その鳴き機会だけをパス強制
以降は全席、実チェックポイントによる素の π サンプリングで局終了まで進む。

差し戻し修正 (監督側の独立再実行で判明): GameplayLoader 事後再構築 + 席解決
(obs,mask 全席探索) は実クエリ列と 1:1 対応しなかった (at_kyoku>=1 で
divergence)。本スクリプトはもう GameplayLoader を台本の情報源として使わず、
sidecar に記録された (digest, action) の単一 FIFO キューを game_key ごとに
消費する。物理席は問わない (どのクエリがどの席から来たかに関わらず、その
game_key への問い合わせである限り同じキューを順に消費する) -- 採取時に
実際に発生した合成クエリ順そのものが台本なので、席解決は不要かつ廃止。
GameplayLoader は分岐点のメタデータ (at_kyoku/shanten/at_turn) と最終
報酬計算にのみ使う (drca_collect_branchpoints.py と同じ用途限定)。

新規 Rust 表面はゼロ。台本再生は純 Python ラッパー engine
(ScriptedForkEngine) が react_batch を横取りして実現する。

set(a): 全4席とも被測定 checkpoint で続行。
set(b): 分岐点席のみ被測定 checkpoint、他3席は --reference-checkpoint
  (drca_probe_design.md §5a-3 では Stage1 step_016000 固定)。分岐点の
  物理席は必ず split の challenger 席 (one_vs_three.rs 固定写像) でなければ
  ならない -- 起動時に全分岐点を assert する (差し戻し要件2)。

各ロールアウトは実際には4スプリット (a/b/c/d) 全てが同時に再生されるが
(OneVsThree の固定 API 制約)、分岐点の game_key と一致するスプリットのみを使う。

決定論性の壊れ (r2) は「台本再生中に問い合わせが来た (obs, mask) の digest
が記録列の次に消費すべき項目の digest と一致しない」場合に即座に loud FAIL
する (--inject-fault-for-test でこの経路自体を実演できる)。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drca_common import (  # noqa: E402
    DECLINE_ACTION,
    assert_construction_diff_empty,
    build_policy_engine,
    call_possible_aka_held,
    compute_kyoku_rewards,
    legal_call_action_ids,
    load_script_sidecar,
    make_arena,
    mask_obs_digest,
    parse_game_key,
    reconstruct_all_seats,
)
from config import config  # noqa: E402

SPLITS = ['a', 'b', 'c', 'd']


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

    The script is a single FIFO queue of (digest, action) recorded live
    during collection (drca_collect_branchpoints.py's
    RecordingPassthroughEngine) -- not a per-seat reconstruction. Physical
    seat identity plays no role in matching; only queue order + digest
    equality does, since that queue *is* the literal sequence of queries
    that occurred for this game_key during collection.
    """

    def __init__(self, *, target_game_key, script_queue, branch_index, arm, inject_fault=False):
        self.target_game_key = target_game_key
        self.script_queue = script_queue  # list[(digest: str, action: int)]
        self.branch_index = branch_index
        self.consumed = 0
        self.arm = arm
        self.crossed_branch = False
        self.branch_served = False
        self.inject_fault = inject_fault
        self._fault_injected = False
        self.n_scripted_served = 0
        self.n_live_served = 0
        self.branch_forced_action = None
        self.branch_call_options = None

    def next_action_for(self, digest: str):
        """Consume the next scripted entry for the target game_key's queue.

        Returns the recorded action, or None if this decision *is* the
        flagged branch point (caller computes the forced/live branch
        action). Raises loudly (no silent fallback) if `digest` doesn't
        match the expected head-of-queue digest -- the r2
        seed-determinism / prefix-replay-integrity guard.
        """
        idx = self.consumed
        if idx >= len(self.script_queue):
            raise RuntimeError(
                f'DRCA replay divergence in game_key={self.target_game_key}: '
                f'script queue exhausted (len={len(self.script_queue)}) before '
                'branch point was reached -- prefix determinism broken'
            )
        exp_digest, exp_action = self.script_queue[idx]
        if digest != exp_digest:
            raise RuntimeError(
                f'DRCA replay divergence in game_key={self.target_game_key}: '
                f'query digest {digest} != expected {exp_digest} at '
                f'queue index={idx} -- this means the replayed wall/action '
                'prefix no longer matches the recorded script (seed-determinism '
                'assert failed)'
            )
        self.consumed = idx + 1
        is_branch = not self.branch_served and idx == self.branch_index
        if is_branch:
            self.branch_served = True
            self.crossed_branch = True
            return None
        if self.inject_fault and not self._fault_injected:
            self._fault_injected = True
            faulted = 0 if exp_action != 0 else 1
            logging.getLogger('drca_probe').warning(
                'FAULT INJECTED (--inject-fault-for-test): idx=%d recorded_action=%d '
                '-> returning %d instead', idx, exp_action, faulted,
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

    def __init__(self, *, role: str, fork_state: ForkState, live_engine, brain, actor_critic, device):
        self.role = role
        self.fork_state = fork_state
        self.live_engine = live_engine
        self.brain = brain
        self.actor_critic = actor_critic
        self.device = device
        self.name = f'drca_fork_{role}'
        self.is_oracle = live_engine.is_oracle
        self.version = live_engine.version
        self.enable_quick_eval = live_engine.enable_quick_eval
        self.enable_rule_based_agari_guard = live_engine.enable_rule_based_agari_guard
        # p_enrich intentionally left undefined -- OneVsThree::py_vs_py
        # falls back to 0.0 via getattr (one_vs_three.rs), same as every
        # other eval/opponent engine in this codebase.

    def _forward_logits(self, obs_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
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
        """Live decision for the forced branch step. call arm: restrict pi
        to the legal call subset [38,42] and renormalize-sample (preserves
        the policy's relative preference among call options,
        drca_probe_design.md §1). no-call arm: force DECLINE_ACTION.
        """
        mask_arr = np.asarray(mask)
        arm = self.fork_state.arm
        if arm == 'no_call':
            assert bool(mask_arr[DECLINE_ACTION]), (
                f'no-call arm requires action {DECLINE_ACTION} (decline) to be legal, '
                f'but mask says otherwise at the recorded branch point'
            )
            return DECLINE_ACTION

        assert arm == 'call', f'unknown arm {arm!r}'
        call_ids = legal_call_action_ids(mask_arr)
        assert call_ids, 'call arm requires at least one legal call action in the mask'
        obs_t = torch.as_tensor(np.asarray(obs)[None], device=self.device)
        mask_t = torch.as_tensor(mask_arr[None], device=self.device, dtype=torch.bool)
        logits = self._forward_logits(obs_t, mask_t)[0]
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
            digest = mask_obs_digest(obs_np[i], masks_np[i])
            action = self.fork_state.next_action_for(digest)
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
                     inject_fault=False, log=None):
    """chal_engine / champ_engine are pre-built (main() constructs them once
    and reuses them across all rollouts, differential-fix item 3 -- only the
    ForkState/wrapper objects below are created fresh per rollout, since
    those carry per-rollout mutable replay-consumption state).
    """
    script_queue = load_script_sidecar(branch['script_path'])
    branch_index = branch['script_index']
    if branch_index >= len(script_queue):
        raise RuntimeError(
            f'branch_index={branch_index} beyond script sidecar length '
            f'{len(script_queue)} for game_key={branch["game_key"]} '
            f'(script_path={branch["script_path"]})'
        )
    sidecar_digest, _sidecar_action = script_queue[branch_index]
    if sidecar_digest != branch['mask_obs_digest']:
        raise RuntimeError(
            f'branch point script sidecar mismatch for game_key={branch["game_key"]} '
            f'script_index={branch_index}: sidecar digest {sidecar_digest} != '
            f'recorded {branch["mask_obs_digest"]} (sidecar corrupted or index drift)'
        )

    # GameplayLoader used only for metadata / sel-definition-drift check
    # (drca_probe_design.md differential fix item 1) -- NOT as the replay
    # script source (that's script_queue above, recorded live at collection).
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
        script_queue=script_queue,
        branch_index=branch_index,
        arm=arm,
        inject_fault=inject_fault,
    )

    chal_wrapper = ScriptedForkEngine(
        role='challenger', fork_state=fork_state, live_engine=chal_engine,
        brain=chal_engine.brain, actor_critic=chal_engine.actor_critic, device=device,
    )
    champ_wrapper = ScriptedForkEngine(
        role='champion', fork_state=fork_state, live_engine=champ_engine,
        brain=champ_engine.brain, actor_critic=champ_engine.actor_critic, device=device,
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
    ap.add_argument('--full-hanchan', action='store_true')
    ap.add_argument('--inject-fault-for-test', action='store_true',
                     help='デバッグ専用: 台本再生中に1アクションを故意に破損させ、'
                          'r2 の assert が実際に発火することを実演する')
    ap.add_argument('--device', default=None)
    ap.add_argument('--tmp-root', default=None)
    args = ap.parse_args()

    if args.mode == 'b' and not args.reference_checkpoint:
        ap.error('--mode b requires --reference-checkpoint')

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

    # Engine construction, once up front (differential-fix item 3 -- reused
    # across all rollouts below; only ForkState/wrapper objects are built
    # per rollout, since those carry per-rollout mutable replay state).
    # `別インスタンス必須` (player.py convention): challenger and champion
    # must be distinct PyObjects even when they share weights (set a).
    chal_ckpt = args.checkpoint
    champ_ckpt = args.checkpoint if args.mode == 'a' else args.reference_checkpoint
    chal_engine, _s1, dump_chal = build_policy_engine(chal_ckpt, device, name='drca_probe_chal')
    champ_engine, _s2, dump_champ = build_policy_engine(champ_ckpt, device, name='drca_probe_champ')
    if args.mode == 'a':
        assert_construction_diff_empty(dump_chal, dump_champ)
    log.info('probed checkpoint chal engine config dump: %s', dump_chal)
    log.info('probed checkpoint champ engine config dump: %s', dump_champ)
    assert dump_chal['p_enrich'] == 0.0 and dump_chal['call_bonus_b'] == 0.0
    assert dump_champ['p_enrich'] == 0.0 and dump_champ['call_bonus_b'] == 0.0

    n_out = 0
    with out_path.open('w', encoding='utf-8') as fout:
        for bi, branch in enumerate(branches):
            log.info('branch %d/%d: game_key=%s seat=%d at_kyoku=%d',
                      bi + 1, len(branches), branch['game_key'], branch['seat'], branch['at_kyoku'])
            for arm in ('call', 'no_call'):
                for k in range(args.k):
                    inject = args.inject_fault_for_test and bi == 0 and arm == 'call' and k == 0
                    result, rewards = run_one_rollout(
                        branch, arm, chal_engine, champ_engine,
                        device, tmp_root, inject_fault=inject, log=log,
                    )
                    record = {
                        'branch_index': bi,
                        'game_key': branch['game_key'],
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
                    fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                    fout.flush()
                    n_out += 1
                    log.info(
                        '  arm=%s k=%d forced_action=%s reward_primary=%.4f '
                        'n_scripted=%d n_live=%d',
                        arm, k, result['forced_action'], result['reward_primary'],
                        result['n_scripted_served'], result['n_live_served'],
                    )

    log.info('DONE: wrote %d rollout records to %s', n_out, out_path)


if __name__ == '__main__':
    main()
