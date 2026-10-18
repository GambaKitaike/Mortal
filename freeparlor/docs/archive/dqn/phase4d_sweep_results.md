# Phase 4d — lambda_opp スイープ結果

## Config 確認（3 本共通 vs 差分）

| 項目 | lo=0.0 | lo=0.3 | lo=0.6 |
|---|---:|---:|---:|
| beta ✓ | 0.3 | 0.3 | 0.3 |
| noten_factor ✓ | 0.0 | 0.0 | 0.0 |
| alpha ✓ | 1.0 | 1.0 | 1.0 |
| gamma_pt ✓ | 1.0 | 1.0 | 1.0 |
| lambda_opp ✓ | 0.0 | 0.3 | 0.6 |
| file_index ✓ | /home/gamba/mahjong/runs/file_index.pth | /home/gamba/mahjong/runs/file_index.pth | /home/gamba/mahjong/runs/file_index.pth |
| num_epochs ✓ | 1 | 1 | 1 |
| seed_key (1v3) ✓ | 42 | 42 | 42 |
| batch_size ✓ | 128 | 128 | 128 |
| resnet ✓ | 192×40 | 192×40 | 192×40 |

## aka-conditional 母集団照合（Step 1）

集計スクリプト `analyze_aka_conditional.analyze_dir` の読み込み先 = 1v3 サマリと同一の `eval/<run>/1v3/`。
全体副露率が 1v3 サマリと一致することを run ごとに確認。

| run | ログ | rounds | Stat fuuro | analyze_dir fuuro | 1v3 期待 | 一致 |
|---|---|---:|---:|---:|---:|:---:|
| lo=0.0 | `/home/gamba/mahjong/runs/phase4d/eval/phase4d_lo00/1v3` | 4007 | 11.95% | 11.95% | 11.95% | ✓ |
| lo=0.3 | `/home/gamba/mahjong/runs/phase4d/eval/phase4d_lo03/1v3` | 3951 | 17.59% | 17.59% | 17.59% | ✓ |
| lo=0.6 | `/home/gamba/mahjong/runs/phase4d/eval/phase4d_lo06/1v3` | 4075 | 20.29% | 20.29% | 20.29% | ✓ |

## 赤条件別 Δ（赤あり − 赤なし, pp）

集計元: `eval/phase4d_lo{00,03,06}/1v3/`（上表照合済み・mortal 席のみ）。

| 指標 | 人間(2009) | AI lo=0.0 | AI lo=0.3 | AI lo=0.6 |
|---|---:|---:|---:|---:|
| 副露率 Δ | +2.81 | +0.52 | -5.01 | +2.10 |
| 立直率 Δ | +3.13 | +4.74 | +11.57 | +4.99 |
| 放銃率 Δ | — | +0.32 | +0.80 | -1.04 |
| 流局率 Δ | +4.08 | +3.66 | +4.16 | +4.02 |

## Monitor（1v3 自己対戦・aka-conditional）

集計元: 上記赤条件別 Δ と同一ログ。

| 指標 | 人間(2009) | AI lo=0.0 | AI lo=0.3 | AI lo=0.6 |
|---|---:|---:|---:|---:|
| 赤保持ノーテン率 | — | 57.40% | 52.40% | 54.79% |
| 赤平均切り順 | — | 10.04 | 10.59 | 10.03 |

## 評価卓構成（Phase4 `sweep_eval` と同方式）

| 評価 | 1席 (challenger) | 3席 (champion) | ログ / 集計対象 |
|---|---|---|---|
| **1v3 自己対戦** | 学習済み `model.pth` (`name=mortal`) | 同一 `model.pth` を `eval/<run>/champion.pth` にコピー (`name=baseline`) | `eval/<run>/1v3/` · `Stat.from_dir(..., 'mortal')` · aka-conditional |
| 学習時 test_play | 学習中 `mortal` | 固定 `grp_baseline.pth` (`[baseline.test]`) | `runs/phase4d/<run>/test_play/` · avg_rank≈1.0（参考外） |

1v3 起動: `run_sweep_eval.py` → `champion.pth` に `model.pth` をコピー → `run_one_vs_three.py` → `py_vs_py(challenger, champion)`。

## 1v3 自己対戦サマリ（打牌統計・Phase4 同条件）

seed_key=42, games=400, mortal 席のみ集計。

| run | avg_rank | 和了率 | 放銃率 | 副露率 | 立直率 | 流局率 |
|---|---:|---:|---:|---:|---:|---:|
| lo=0.0 | 2.495 | 20.14% | 12.83% | 11.95% | 26.38% | 19.72% |
| lo=0.3 | 2.485 | 20.63% | 13.26% | 17.59% | 23.49% | 18.35% |
| lo=0.6 | 2.525 | 20.61% | 14.53% | 20.29% | 25.42% | 17.40% |

## （参考外）学習時 test_play — `grp_baseline.pth` 固定3席

学習ループ内 `TestPlayer.test_play`: 1席=学習中モデル、3席=`/home/gamba/mahjong/runs/grp_baseline.pth`。自己対戦ではない。

| run | avg_rank | 和了率 | 放銃率 | 副露率 | 立直率 | 流局率 |
|---|---:|---:|---:|---:|---:|---:|
| lo=0.0 | 1.010 | 36.51% | 0.23% | 8.85% | 37.83% | 62.30% |
| lo=0.3 | 1.006 | 39.70% | 0.22% | 5.95% | 39.64% | 59.08% |
| lo=0.6 | 1.005 | 37.75% | 0.17% | 7.80% | 35.19% | 61.11% |

## （参考外・壊れた母集団）赤条件別 Δ — 初回 md 掲載値

母集団照合前の掲載値。数値は健全 1v3 再集計と一致（ソース未明示のため参考外として残す）。

| 指標 | 人間(2009) | AI lo=0.0 | AI lo=0.3 | AI lo=0.6 |
|---|---:|---:|---:|---:|
| 副露率 Δ | +2.81 | +0.52 | -5.01 | +2.10 |
| 立直率 Δ | +3.13 | +4.74 | +11.57 | +4.99 |
| 放銃率 Δ | — | +0.32 | +0.80 | -1.04 |
| 流局率 Δ | +4.08 | +3.66 | +4.16 | +4.02 |

### （参考外）Monitor — 初回 md 掲載値

| 指標 | 人間(2009) | AI lo=0.0 | AI lo=0.3 | AI lo=0.6 |
|---|---:|---:|---:|---:|
| 赤保持ノーテン率 | — | 57.40% | 52.40% | 54.79% |
| 赤平均切り順 | — | 10.04 | 10.59 | 10.03 |
