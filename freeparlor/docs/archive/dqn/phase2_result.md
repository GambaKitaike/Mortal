# Phase 2 Result — Free-Parlor Reward (64×10 Connectivity, 2026-06-20)

## Reward Design

```
reward_t = alpha · (calc_delta_points / 1000) + gamma_pt · calc_delta_pt
```

| Parameter | Value | Note |
|---|---|---|
| `pts` | `[+35, +5, −15, −25]` | ウマオカ順位点（千点・ゼロ和） |
| `alpha` | 1.0 | 素点（千点正規化）の重み |
| `gamma_pt` | 1.0 | 順位点の重み |
| `gamma` | 1 | 割引率（既存・変更なし） |

Implementation: `calc_delta_blend` in `reward_calculator.py`; wired at `dataloader.py:102`.

## Step 0 — Array Length Check

```
sample_file: .../2009020103gm-00a9-0000-2453a04c.mjson
player_id: 0
grp_feature shape: (34, 7)
len(calc_delta_pt): 34
len(calc_delta_points): 34
match: True
```

## Training (64×10, 2009 data)

| Item | Value |
|---|---|
| ResNet | conv_channels=64, num_blocks=10 |
| GRP | Phase 1 `grp.pth` reused (no retrain) |
| Steps | 17,600 (1 offline epoch) |
| GPU | cuda:0 (RTX 5060) |
| dqn_loss (final) | 33.90 |
| cql_loss (final) | 1.02 |
| checkpoint | `runs/mortal.pth` |

Loss absolute values differ from Phase 1 rank-reward run (dqn 0.435 / cql 0.664) due to reward scale; no NaN/divergence.

## one_vs_three Self-Play (400 games, identical 64×10 model × 4)

| Metric | Value |
|---|---|
| avg_rank | 2.502 |
| avg_pt | −0.34 |

## Playstyle Stats vs Phase 1 Baseline

Phase 1 baseline: 192×40 rank-reward model (`phase1_stats_192x40.md`). Phase 2: 64×10 free-parlor reward model. Architecture differs; treat as directional signal, not controlled ablation.

| Metric | Phase 1 | Phase 2 | Δ |
|---|---:|---:|---:|
| Win rate (和了率) | 22.27% | 20.88% | −1.4pp |
| Deal-in rate (放銃率) | 14.31% | 14.53% | +0.2pp |
| Riichi rate (立直率) | 15.67% | 29.07% | +13.4pp |
| Call rate (副露率) | 30.39% | 11.74% | −18.7pp |
| Avg winning Δscore | 5505 | 7027 | +1522 |
| Ryukyoku rate | 11.26% | 17.89% | +6.6pp |

### Interpretation

- **副露↓ / 立直↑** — 素点報酬により鳴き依存が減り、立直・ダマ寄りの打牌にシフト。
- **平均和了打点↑ (+1522)** — 小打点より大きい手を狙う方向性が観測される（設計書の仮説と一致）。
- **放銃率** — 設計書仮説（低下）には未達（+0.2pp）。64×10 vs 192×40 の差も混在。
- **流局率↑** — 守備・テンパイ重視の副作用の可能性。

## Code Changes (vs upstream/main)

Only `mortal/reward_calculator.py` and `mortal/dataloader.py` (plus local `config.toml`, gitignored). See commit diff.

## Artifacts

```
runs/
  mortal.pth                 # Phase 2 64×10 free-parlor reward
  mortal_gen1.pth
  champion.pth
  step3_train_phase2.log
  step4_one_vs_three_phase2.log
  check_reward_array_lengths.py  # Step 0 verification (runs/, not committed)
```

---

# Phase 2 Production — 192×40 Run (2026-06-21)

## Training

| Item | Value |
|---|---|
| ResNet | conv_channels=192, num_blocks=40 |
| batch_size | 128 |
| GRP | Phase 1 `grp.pth` reused |
| Steps | 35,200 (1 offline epoch) |
| Duration | ~3h08m (07:16–10:24) |
| GPU | cuda:0 (RTX 5060) |
| dqn_loss (final) | 15.80 |
| cql_loss (final) | 0.95 |
| checkpoint | `runs/mortal.pth`, `runs/best.pth`, `runs/mortal_gen1.pth` |

No NaN/divergence. Loss scale remains higher than Phase 1 rank-reward (dqn 0.427 / cql 0.631) due to free-parlor reward.

## one_vs_three Self-Play (400 games, identical 192×40 model × 4)

| Metric | Value |
|---|---|
| avg_rank | 2.490 |
| avg_pt | +1.13 |

## Playstyle Stats vs Phase 1 (both 192×40)

Phase 1 baseline: rank-reward (`phase1_stats_192x40.md`). Phase 2: free-parlor reward. Same architecture — controlled comparison.

| Metric | Phase 1 | Phase 2 | Δ |
|---|---:|---:|---:|
| Win rate (和了率) | 22.27% | 21.38% | −0.9pp |
| Deal-in rate (放銃率) | 14.31% | 13.98% | −0.3pp |
| Riichi rate (立直率) | 15.67% | 26.23% | +10.6pp |
| Call rate (副露率) | 30.39% | 16.97% | −13.4pp |
| Avg winning Δscore | 5505 | 6827 | +1322 |
| Ryukyoku rate | 11.26% | 15.45% | +4.2pp |

### Interpretation (192×40)

- **副露↓ / 立直↑ / 平均打点↑** — 64×10 疎通と同方向。アーキテクチャを揃えても報酬差の効果は再現。
- **放銃率 −0.3pp** — 設計書仮説方向だが効果は小さい。
- 64×10 比 192×40 の方が副露率差（−13.4pp vs −18.7pp）は穏やか。大モデルは鳴き判断が残る可能性。

## Production Artifacts

```
runs/
  mortal.pth                      # Phase 2 192×40 free-parlor reward
  mortal_gen1.pth
  champion.pth
  best.pth
  mortal_phase2_64x10.pth         # archived 64×10 connectivity ckpt
  step3_train_phase2_production.log
  step4_one_vs_three_phase2_production.log
```
