# Phase 3 Result — α:γ Ratio Sweep (192×40, 2026-06-21)

## Sweep Points

| # | name | alpha | gamma_pt | pts | 学習 | model |
|---|---|---:|---:|---|---|---|
| 1 | rank_only | 0 | 1 | [6,4,2,0] | 既存(Phase1) | `runs/sweep/rank_only/model.pth` |
| 2 | rank_heavy | 1 | 2 | [35,5,-15,-25] | 新規 | `runs/sweep/rank_heavy/model.pth` |
| 3 | balanced | 1 | 1 | [35,5,-15,-25] | 既存(Phase2) | `runs/sweep/balanced/model.pth` |
| 4 | score_heavy | 2 | 1 | [35,5,-15,-25] | 新規 | `runs/sweep/score_heavy/model.pth` |

共通: ResNet 192×40, data=tenhou/2009, GRP=`runs/grp.pth`, 1 offline epoch (35,200 steps).

## Training (新規2本)

| name | alpha | gamma_pt | dqn_loss | cql_loss | duration |
|---|---:|---:|---:|---:|---|
| rank_heavy | 1 | 2 | 37.74 | 1.14 | ~3h07m |
| score_heavy | 2 | 1 | 34.99 | 1.09 | ~3h08m |

NaN/発散なし。既存2本の loss (参考): rank_only dqn=0.427/cql=0.631, balanced dqn=15.80/cql=0.95。

## Evaluation (同一条件)

| Item | Value |
|---|---|
| seed_key | 42 |
| games_per_iter | 40 |
| iters | 10 |
| total games | 400 / model |
| champion = challenger | 各モデル自身 (自己対戦) |

各 log_dir: `runs/sweep/<name>/1v3/`

## 4-Point Comparison Table

| 指標 | rank_only | rank_heavy | balanced | score_heavy |
|---|---:|---:|---:|---:|
| 和了率 | 22.27% | 18.92% | 21.38% | 19.44% |
| 放銃率 | 14.31% | 12.86% | 13.98% | 12.76% |
| 立直率 | 15.67% | 27.38% | 26.23% | 29.24% |
| 副露率 | 30.39% | 6.84% | 16.97% | 10.41% |
| 平均和了打点 | 5505 | 7264 | 6827 | 7301 |
| 流局率 | 11.26% | 25.13% | 15.45% | 22.61% |
| avg_rank (sanity) | 2.502 | 2.480 | 2.490 | 2.495 |

Tool: `libriichi.stat.Stat.from_dir(log_dir, 'mortal')`, 400 games each.

## 所見 — α:γ比 vs 打牌統計

**素点重視方向 (rank_only → score_heavy):**

- **平均和了打点↑** — rank_only 5505 → score_heavy 7301 (+33%)。単調増加に近い (balanced 6827 は中間)。
- **副露率↓** — 30.4% → 6.8% (rank_heavy) → 17.0% (balanced) → 10.4% (score_heavy)。順位点のみ (rank_only) 以外は大幅低下。γ↑ (rank_heavy) で最も低い。
- **立直率↑** — 15.7% → 29.2% (score_heavy)。素点重視ほど立直寄り。単調増加。
- **放銃率↓** — 14.3% → 12.8%。素点/順位点ブレンドモデルは rank_only より低い。
- **和了率↓** — 22.3% → 18.9〜19.4%。大打点・守備寄りのトレードオフ。
- **流局率↑** — 11.3% → 25.1% (rank_heavy)。守備・テンパイ重視の副作用。

**γ重視 (rank_heavy vs balanced, 同 pts):**

- rank_heavy (γ=2) は balanced (γ=1) より副露↓ (6.8% vs 17.0%)、流局↑ (25.1% vs 15.5%)、打点↑ (7264 vs 6827)。順位点を強くすると「鳴かない・流す」方向が増幅。

**注意:** rank_only は pts=[6,4,2,0] で他3点と pts も異なるため、純粋な α:γ アブレーションではない。pts 差 (rank_only の高副露) と α:γ 差が混在。同一 pts 内 (rank_heavy / balanced / score_heavy) の比較がよりクリーン。

## Artifacts

```
runs/sweep/
  rank_only/model.pth, 1v3/
  rank_heavy/model.pth, 1v3/, train.log, tb/
  balanced/model.pth, 1v3/
  score_heavy/model.pth, 1v3/, train.log, tb/
  step2_all_1v3.log
  step3_stats.log
```

Config templates: `freeparlor/configs/phase3_sweep_*.toml`
