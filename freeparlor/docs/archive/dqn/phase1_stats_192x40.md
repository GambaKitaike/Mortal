# Phase 1 Playstyle Stats — 192×40 Self-Play (2026-06-20)

## Setup

| Item | Value |
|---|---|
| Model | 192×40 `mortal_gen1.pth` (champion = challenger) |
| Logs | `/home/gamba/mahjong/runs/1v3/*.json.gz` |
| Games | 400 (4 seats × 100 seeds) |
| Tool | `libriichi.stat.Stat.from_dir(log_dir, 'mortal')` |
| Player name | `mortal` (matches `[1v3.challenger].name`) |

## Summary Table

| Metric | Value | 鳳凰卓目安 | Δ |
|---|---:|---:|---|
| 1st rate | 24.75% | ~25% | ≈ |
| 2nd rate | 25.25% | ~25% | ≈ |
| 3rd rate | 25.00% | ~25% | ≈ |
| 4th rate | 25.00% | ~25% | ≈ |
| Tobi rate | 5.00% | ~5–8% | ≈ |
| Avg rank | 2.503 | 2.5 | ≈ |
| Win rate (和了率) | 22.27% | ~22% | ≈ |
| Deal-in rate (放銃率) | 14.31% | ~12% | +2.3pp |
| Riichi rate (立直率) | 15.67% | ~18% | −2.3pp |
| Call rate (副露率) | 30.39% | ~35% | −4.6pp |
| Ryukyoku rate (流局率) | 11.26% | — | — |
| Yakuman rate | 0.00% | — | — |
| Nagashi mangan rate | 0.00% | — | — |

## Derived / Secondary Metrics

| Metric | Value |
|---|---:|
| Avg winning Δscore | 5505 |
| Avg deal-in Δscore | −4560 |
| Winning rate after riichi | 49.78% |
| Deal-in rate after riichi | 16.64% |
| Avg riichi turn | 7.59 |
| Winning rate after call | 35.84% |
| Deal-in rate after call | 14.58% |
| Chasing riichi rate | 15.46% |
| Riichi chased rate | 16.20% |
| Avg number of calls | 1.54 |
| Dealer wins / all dealer rounds | 22.57% |

## Health Assessment

**Overall: 健全。** 同型モデル4人自己対戦（400局）として、順位分布・和了率は理論値どおり。打牌スタイルは鳳凰卓の人間平均に近い。

- **順位分布 (24.8 / 25.3 / 25.0 / 25.0%)** — 同一モデル同士の期待値（各25%）と一致。avg_rank 2.503、avg rank pt −0.11 もノイズ範囲。
- **和了率 22.27%** — 鳳凰目安 ~22% とほぼ一致。健全。
- **放銃率 14.31%** — 鳳凰 ~12% よりやや高い（+2.3pp）。自己対戦では相手も同じ守備力のため、人間卓との単純比較は参考程度。過剰な攻撃性は見られない。
- **立直率 15.67%** — 鳳凰 ~18% よりやや低い（−2.3pp）。副露寄りのバランスと整合。
- **副露率 30.39%** — 鳳凰 ~35% よりやや低い（−4.6pp）。やや守備・ダマ寄りだが極端ではない。
- **流局率 11.26%** — 特記事項なし。
- **役満・流し満貫 0%** — 400局では統計的に十分起きうる。異常とは言えない。

**解釈上の注意:** 4席すべて同一192×40モデル。順位・pt は理論上フラットになるため、和了率・放銃率・立直率・副露率の絶対値が主な健全性指標。人間卓との差は「同レベルAI同士の均衡」vs「人間混合卓」の違いも含む。

## Raw Output

```
Games            400
Rounds           4334
Rounds as dealer 1081

1st (rate)       99 (0.247500)
2nd (rate)       101 (0.252500)
3rd (rate)       100 (0.250000)
4th (rate)       100 (0.250000)
Tobi(rate)       20 (0.050000)
Avg rank         2.502500
Total rank pt    -45
Avg rank pt      -0.112500
Total Δscore     -14000
Avg game Δscore  -35.000000
Avg round Δscore -3.230272

Win rate      0.222658
Deal-in rate  0.143055
Call rate     0.303876
Riichi rate   0.156668
Ryukyoku rate 0.112598

Avg winning Δscore               5504.870466
Avg winning Δscore as dealer     7302.868852
Avg winning Δscore as non-dealer 4896.393897
Avg riichi winning Δscore        7409.763314
Avg open winning Δscore          4189.406780
Avg dama winning Δscore          5356.774194
Avg ryukyoku Δscore              3.073770

Avg winning turn        10.636269
Avg riichi winning turn 10.840237
Avg open winning turn   10.447034
Avg dama winning turn   10.767742

Avg deal-in turn                 11.046774
Avg deal-in Δscore               -4560.161290
Avg deal-in Δscore to dealer     -6167.948718
Avg deal-in Δscore to non-dealer -4019.612069

Chasing riichi rate       0.154639
Riichi chased rate        0.162003
Winning rate after riichi 0.497791
Deal-in rate after riichi 0.166421
Avg riichi turn           7.586156
Avg riichi Δscore         2911.340206

Avg number of calls     1.542901
Winning rate after call 0.358390
Deal-in rate after call 0.145786
Avg call Δscore         860.592255

Dealer wins/all dealer rounds  0.225717
Dealer wins/all wins           0.252850
Deal-in to dealer/all deal-ins 0.251613

Yakuman (rate)        0 (0.000000000)
Nagashi mangan (rate) 0 (0.000000000)
```
