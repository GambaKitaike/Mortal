# Phase 4d — 赤保持→チップ実現

集計: mortal 席（AI）/ 全席（人間2009）。赤保持局 = 局終了スナップショット `aka_held>0`（preprocess 同定義）。

| 列 | ログ |
|---|---|
| 人間(2009) | `/home/gamba/mahjong/data/tenhou/2009/*.mjson` |
| lo=0.0 | `/home/gamba/mahjong/runs/phase4d/eval/phase4d_lo00/1v3` |
| lo=0.3 | `/home/gamba/mahjong/runs/phase4d/eval/phase4d_lo03/1v3` |
| lo=0.6 | `/home/gamba/mahjong/runs/phase4d/eval/phase4d_lo06/1v3` |

Tool: `freeparlor/scripts/analyze_chip_realize.py`

## 母集団

| 指標 | 人間(2009) | lo=0.0 | lo=0.3 | lo=0.6 |
|---|---:|---:|---:|---:|
| 赤保持局数 | 111555 | 1666 | 1652 | 1637 |

## chip_realize_rate / aka_chip_realize_rate

| 指標 | 人間(2009) | lo=0.0 | lo=0.3 | lo=0.6 |
|---|---:|---:|---:|---:|
| chip_realize_rate | 21.75% | 18.13% | 20.52% | 20.16% |
| aka_chip_realize_rate | 21.75% | 18.13% | 20.52% | 20.16% |

## 赤保持局 — 和了経路内訳

| 指標 | 人間(2009) | lo=0.0 | lo=0.3 | lo=0.6 |
|---|---:|---:|---:|---:|
| 和了率 | 21.75% | 18.13% | 20.52% | 20.16% |
| 立直和了率 | 9.19% | 11.76% | 12.89% | 12.34% |
| 鳴き和了率 | 8.98% | 3.36% | 3.03% | 4.46% |

## 赤保持局あたり平均チップ枚数（net）

| 指標 | 人間(2009) | lo=0.0 | lo=0.3 | lo=0.6 |
|---|---:|---:|---:|---:|
| 平均チップ枚数 | 0.354 | 0.249 | 0.334 | 0.250 |
