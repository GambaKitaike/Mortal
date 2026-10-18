#!/usr/bin/env python3
"""DRCA pilot の分岐点別 ΔQ̂ ドリルダウン + 極値分岐点の局面近似再構成（read-only）。

**exploratory・判定非関与**。本測定の判定は `drca_probe_design.md` §5a の凍結条件
（全体平均 + 事前列挙 contrast）のみで行う。本スクリプトは
`drca_pilot_qualitative_notes.md`（0b セッションの定性材料）の再現用。

- 分岐点別 ΔQ̂ = mean(call 腕 K=8) − mean(no_call 腕 K=8)。全体平均が commit 済みの
  −0.4575 と一致することを assert（整合検証）
- 局面再構成は牌譜 json.gz の手牌追跡による**近似**（分岐点の厳密な特定は engine
  再実行なしには不可能。at_turn と discarder 打牌数の近接で照合し、一致候補に
  マークを付けるのみ）

stdlib のみ・GPU/conda 不要。進行中の run には触れない。
"""
import gzip
import json
import statistics
from pathlib import Path

PILOT = Path('/home/gamba/mahjong/runs/drca/pilot_20260714_020133')
# |ΔQ̂| 上位 + 縮退例（ΔQ̂=0）。notes md §1 と対応
DRILLDOWN_BRANCHES = [0, 44, 39, 24, 2, 30, 15, 12]
COMMITTED_OVERALL = -0.4575


def load_branch_table():
    bps = [json.loads(l) for l in (PILOT / 'bp.jsonl').read_text().splitlines() if l.strip()]
    rows = [json.loads(l) for l in (PILOT / 'probe_a.jsonl').read_text().splitlines() if l.strip()]
    by_branch = {}
    for r in rows:
        by_branch.setdefault(r['branch_index'], {'call': [], 'no_call': []})[r['arm']].append(r)

    out = []
    for bi, arms in sorted(by_branch.items()):
        call, nocall = arms['call'], arms['no_call']
        assert len(call) == 8 and len(nocall) == 8, (bi, len(call), len(nocall))
        bp = bps[bi]
        assert bp['game_key'] == call[0]['game_key']

        def mean(recs, key):
            return statistics.fmean(rec[key] for rec in recs)

        d_primary = mean(call, 'reward_primary') - mean(nocall, 'reward_primary')
        se = (statistics.stdev(r['reward_primary'] for r in call) ** 2 / 8
              + statistics.stdev(r['reward_primary'] for r in nocall) ** 2 / 8) ** 0.5
        out.append({
            'branch_index': bi,
            'game_key': bp['game_key'],
            'seat': bp['seat'],
            'at_kyoku': bp['at_kyoku'],
            'at_turn': bp['at_turn'],
            'shanten': bp['shanten'],
            'call_types': bp['call_types_available'],
            'score_rank': bp['score_rank_at_branch'],
            'orig_action': bp['action_taken_originally'],
            'd_primary': round(d_primary, 3),
            'se_within': round(se, 3),
            'd_sotensu': round(mean(call, 'reward_sotensu_kyoku') - mean(nocall, 'reward_sotensu_kyoku'), 3),
            'd_grp': round(mean(call, 'reward_grp_kyoku') - mean(nocall, 'reward_grp_kyoku'), 3),
            'd_chip': round(mean(call, 'reward_chip_kyoku') - mean(nocall, 'reward_chip_kyoku'), 3),
            'call_mean': round(mean(call, 'reward_primary'), 3),
            'nocall_mean': round(mean(nocall, 'reward_primary'), 3),
            'log': bp['game_log_path'],
        })

    overall = statistics.fmean(o['d_primary'] for o in out)
    assert abs(overall - COMMITTED_OVERALL) < 5e-4, f'overall {overall} != commit値 {COMMITTED_OVERALL}'
    print(f'branches={len(out)} overall_mean_dQ={overall:.4f} (commit値と一致)')
    return bps, out


def norm(pai):
    if pai.endswith('r'):
        return pai[1], int(pai[0]), True
    if pai[0] == '0':
        return pai[1], 5, True
    if pai[0].isdigit():
        return pai[1], int(pai[0]), False
    return pai, 0, False  # 字牌


def hand_counts(hand):
    c = {}
    for p in hand:
        s, r, _ = norm(p)
        c[(s, r)] = c.get((s, r), 0) + 1
    return c


def can_chi(hand, pai):
    s, r, _ = norm(pai)
    if s not in 'mps':
        return False
    c = hand_counts(hand)
    return any(c.get((s, a), 0) and c.get((s, b), 0)
               for a, b in ((r - 2, r - 1), (r - 1, r + 1), (r + 1, r + 2)))


def remove_tiles(hand, tiles):
    for t in tiles:
        if t in hand:
            hand.remove(t)
        else:
            s, r, _ = norm(t)
            for h in list(hand):
                if norm(h)[:2] == (s, r):
                    hand.remove(h)
                    break


def print_situation(bp):
    seat, kyoku_idx, at_turn = bp['seat'], bp['at_kyoku'], bp['at_turn']
    with gzip.open(bp['game_log_path'], 'rt') as f:
        events = [json.loads(l) for l in f if l.strip()]

    k = -1
    start_i = end_i = None
    for i, e in enumerate(events):
        if e['type'] == 'start_kyoku':
            k += 1
            if k == kyoku_idx:
                start_i = i
        elif e['type'] == 'end_kyoku' and k == kyoku_idx and start_i is not None:
            end_i = i
            break
    ev = events[start_i:end_i + 1]
    sk = ev[0]

    hand = list(sk['tehais'][seat])
    haipai = sorted(hand)
    candidates = []
    dahai_count = {a: 0 for a in range(4)}
    endings = []

    for e in ev[1:]:
        t = e['type']
        if t == 'tsumo' and e['actor'] == seat:
            hand.append(e['pai'])
        elif t == 'dahai':
            actor = e['actor']
            dahai_count[actor] += 1
            if actor == seat:
                remove_tiles(hand, [e['pai']])
            else:
                pai = e['pai']
                s, r, _ = norm(pai)
                c = hand_counts(hand)
                types = []
                if actor == (seat + 3) % 4 and can_chi(hand, pai):
                    types.append('chi')
                if c.get((s, r), 0) >= 2:
                    types.append('pon')
                if c.get((s, r), 0) >= 3:
                    types.append('kan')
                if types and any(norm(p)[2] for p in hand):
                    candidates.append({
                        'discarder': actor,
                        'discarder_junme': dahai_count[actor],
                        'pai': pai,
                        'types': types,
                        'hand': sorted(hand),
                    })
        elif t in ('chi', 'pon', 'daiminkan', 'ankan') and e['actor'] == seat:
            remove_tiles(hand, e['consumed'])
        elif t == 'kakan' and e['actor'] == seat:
            remove_tiles(hand, [e['pai']])
        elif t in ('hora', 'ryukyoku'):
            endings.append({kk: vv for kk, vv in e.items()
                            if kk in ('type', 'actor', 'target', 'deltas', 'yakus')})

    print('=' * 70)
    print(f"branch {bp['branch_index']}: game={bp['game_key']} seat={seat} "
          f"kyoku_idx={kyoku_idx} at_turn={at_turn} shanten={bp['shanten']} "
          f"call_types={bp['call_types']} orig_action={bp['orig_action']} "
          f"score_rank={bp['score_rank']}")
    print(f"  start_kyoku: bakaze={sk['bakaze']} kyoku={sk['kyoku']} honba={sk['honba']} "
          f"oya={sk['oya']} dora_marker={sk['dora_marker']}")
    print(f"  scores(at kyoku start)={sk['scores']}  (seat {seat} = {sk['scores'][seat]})")
    print(f"  haipai[seat{seat}]={haipai}")
    print(f"  call-opportunity candidates w/ aka (approx, n={len(candidates)}):")
    for cd in candidates:
        mark = ' <-- at_turn?' if cd['discarder_junme'] == at_turn else ''
        print(f"    junme={cd['discarder_junme']} discarder=seat{cd['discarder']} "
              f"pai={cd['pai']} types={cd['types']}{mark}")
        if mark:
            print(f"      hand at moment: {cd['hand']}")
    print(f"  original kyoku ending: {json.dumps(endings, ensure_ascii=False)}")


def main():
    bps, table = load_branch_table()
    table.sort(key=lambda o: -abs(o['d_primary']))
    print()
    print('=== |ΔQ̂| 上位 10（個別値は腕内 SE 大 — anecdotal）===')
    for o in table[:10]:
        print(json.dumps(o, ensure_ascii=False))
    print()
    by_index = {o['branch_index']: o for o in table}
    for bi in DRILLDOWN_BRANCHES:
        o = dict(by_index[bi])
        o['game_log_path'] = o.pop('log')
        print_situation(o)


if __name__ == '__main__':
    main()
