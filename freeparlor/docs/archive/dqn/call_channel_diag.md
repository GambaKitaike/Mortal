# 鳴き和了チャネル診断 (call_channel_diag)

**日付:** 2026-06-30

## 実行条件

- コマンド: `python call_channel_diag.py --log-dirs /home/gamba/mahjong/runs/online_diag_b/train_play/client0 /home/gamba/mahjong/runs/online_cql_mqw03/train_play/client0 /home/gamba/mahjong/runs/online_main/train_play/client0 --labels diag_b mqw03 main --version 4`

- **diag_b**: `/home/gamba/mahjong/runs/online_diag_b/train_play/client0` — ファイル 800、game 3198、スキップ 2、総局 28672
- **mqw03**: `/home/gamba/mahjong/runs/online_cql_mqw03/train_play/client0` — ファイル 0、game 0、スキップ 0、総局 0
- **main**: `/home/gamba/mahjong/runs/online_main/train_play/client0` — ファイル 800、game 3184、スキップ 16、総局 29783
- **教師データ**: `/home/gamba/mahjong/data/tenhou/2009/*.mjson` サンプル 500 ファイル、game 1923、総局 20573

## Part A: 和了時内訳（排他分割）

副露和了 = trainee和了 ∧ 鳴き(38–42)あり / 立直和了 = trainee和了 ∧ 門前 ∧ 立直(37) / ダマ和了 = trainee和了 ∧ 門前 ∧ 立直なし。

| ソース | trainee和了 | 副露 | % | 立直 | % | ダマ | % | 副露和了赤率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| human | 0 | 0 | 49.29% | 0 | 42.04% | 0 | 8.67% | nan% |
| diag_b | 3352 | 165 | 4.92% | 2934 | 87.53% | 253 | 7.55% | 39.39% |
| mqw03 | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0.00% | 0.00% |
| main | 2651 | 1161 | 43.79% | 1489 | 56.17% | 1 | 0.04% | 34.11% |
| teacher | 4350 | 2053 | 47.20% | 1715 | 39.43% | 582 | 13.38% | 42.33% |

## Part B: explore / skill 切り分け（代表: diag_b）

### B1: 鳴ける時に鳴いているか

| 指標 | 分子 | 分母 | P |
|---|---:|---:|---:|
| P(call \| call legal) | 38288 | 93764 | 40.83% |
| P(chi \| chi legal) | 14845 | 56006 | 26.51% |
| P(pon \| pon legal) | 20647 | 35419 | 58.29% |

### B2: 鳴いた局が和了に化けるか

| 指標 | 値 |
|---|---:|
| 副露局数 | 19717 |
| 門前局数 | 8955 |
| P(和了 \| 副露局) | 0.84% |
| P(和了 \| 門前局) | 35.59% |

#### 副露局の結末内訳

| 結末 | 局数 | 副露局比 |
|---|---:|---:|
| trainee和了 | 165 | 0.84% |
| 他家和了 | 8571 | 43.47% |
| 流局 | 10981 | 55.69% |

## 結論

- **ckpt 間で副露和了率が二極化**（正常≈43.79%: main / 異常≈4.92%: diag_b）。人間=49.29% → 故障は universal ではなく **diag_b 固有の回帰**。
- B1(diag_b) P(call|legal)=40.83% はそこそこ (chi=26.51%, pon=58.29%) → **explore 不足ではない**（鳴ける局面では約4割鳴いている）。
- **B2(diag_b) P(和了|副露局)=0.84% ≪ P(和了|門前局)=35.59%** → 鳴いた手が死ぬ。(C)探索注入単独では不足。流局偏重(55.69%) → 形/役が作れていない疑い。
- 教師副露和了率=47.20%（人間並み）。正常 ckpt **main**=43.79% は教師に近いが、**diag_b**=4.92% だけ崩れている → 教師/正常 ckpt には副露和了があるのに特定 ckpt で潰れた → **報酬/CQL 設計 or diag_b 固有の学習経路**を疑う。
- mqw03: train_play/client0 にログなし（該当なし）。Part B 代表は mqw03 不在のため **diag_b** で実施。

### 判定根拠（action ID）

- 副露: trainee action ∈ [38, 39, 40, 41, 42]
- 立直: trainee action == 37
