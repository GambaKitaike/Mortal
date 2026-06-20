# Phase 1 Result — Reproducible 64×10 Run (2026-06-20)

## Data & Model

| Item | Value |
|---|---|
| Data | `tenhou/2009` (6,897 `.mjson` files) |
| ResNet | conv_channels=64, num_blocks=10 |
| Mortal params | 1,569,864 |
| Training steps | 17,600 (1 offline epoch) |

## GRP (Step 2)

| Metric | Value |
|---|---|
| val_loss @ step 1000 | 3.140 |
| checkpoint | `runs/grp.pth` |

## Offline CQL (Step 3)

| Metric | Value |
|---|---|
| dqn_loss (final) | 0.435 |
| cql_loss (final) | 0.664 |
| checkpoint | `runs/mortal.pth`, `runs/mortal_gen1.pth` |

## one_vs_three Self-Play Sanity (Step 4)

Champion = challenger (same 64×10 `mortal_gen1.pth` copied to `champion.pth`).

| Metric | Value |
|---|---|
| avg_rank | 2.5 |
| avg_pt | 0.0 |

All 10 iterations: `[10 10 10 10]` — expected for identical models.

## WSL2 Note

Direct `python train_grp.py` / `python train.py` fails with CUDA fork error.
Use spawn launchers in `runs/` (Mortal code unchanged):

```bash
export MORTAL_CFG=<HOME>/mahjong/Mortal/mortal/config.toml
python <HOME>/mahjong/runs/run_train_grp.py
python <HOME>/mahjong/runs/run_train.py
```

## Artifacts

```
runs/
  grp.pth
  mortal.pth
  mortal_gen1.pth
  champion.pth
  grp_baseline.pth  # 64×10 baseline for test_play init
```
