# 診断B — online 弱 CQL 導入の副露影響切り分け

**日付:** 2026-06-27  
**run:** `/home/gamba/mahjong/runs/online_diag_b/`  
**状態:** step2000 test_play 完了。cleanup 正常終了。

---

## 目的

offline では `cql_loss` が OOD 行動（無差別鳴き含む）を抑えていたが、online 本番は `cql_loss=0`（`train.py` で `not online` 時のみ計算）。  
この保守性喪失が online 副露暴走の主因かを、弱い CQL を online に残して短時間確認する。

**判定基準（事前）:**

| 結果 | 解釈 |
|---|---|
| cql 弱導入で step2000 副露が抑制（≈17–35% 帯） | 主因候補は **cql_loss=0** |
| 抑制されない（≈61% 維持） | 主因は cql **以外** |

---

## 実験条件

| 項目 | online_main（参照） | 診断B |
|---|---|---|
| warm-start | phase4d lo=0.3 | 同一 |
| lambda_opp | 0.3 | 0.3（変更なし） |
| cql_loss | **0** | **あり** ← 唯一の変更 |
| enable_online | false（既定） | **true** |
| min_q_weight | 5（未使用） | **1**（offline 5 の弱版） |
| beta_sel | 0→0.3（step2000–4000 ramp） | 同一 |
| 構成 | server×1, trainer×1, client×3 | 同一 |
| test_play | self-play 3000 半荘 | 同一 |
| 目標 step | — | **2000** |

**コード変更:** `mortal/train.py` に `enable_cql_online = config['cql'].get('enable_online', False)` を追加。  
`if not online or enable_cql_online:` で CQL を online でも計算可能に（本番 config では既定 false のまま）。

config: `/home/gamba/mahjong/runs/online_diag_b/config.toml`

---

## タイムライン

| 時刻 | イベント |
|---|---|
| 01:13 | 起動（warmstart step0 復元） |
| 01:26 | Step0 ループ成立、学習開始 |
| 02:12 | total steps: 2,000（test_play 開始） |
| **02:34** | **step2000 test_play 完了** |
| 02:34 | `run_online_diag.sh` DONE、cleanup 正常（プロセス全停止） |

所要: 起動→step2000 test_play 約 **1時間20分**。

---

## 結果 ★

### step2000 test_play（診断B の主結果）

| 指標 | 診断B (cql弱) | online_main | phase4d warmstart |
|---:|---:|---:|---:|
| **副露率** | **13.61%** | **61.25%** | **17.59%** |
| 立直率 | **34.18%** | — | 23.49% |
| 和了率 | 20.73% | — | 20.63% |
| 放銃率 | 14.53% | — | 13.26% |
| 流局率 | 17.96% | — | 18.35% |
| beta_sel | **≈0.0** | **0.0** | — |

出典:

```
2026-06-27 02:34:12  test_play behavior: agari=20.73% houjuu=14.53% fuuro=13.61% riichi=34.18% ryukyoku=17.96%
```

---

## 判定

| 事前仮説 | 結果 |
|---|---|
| cql 弱導入で副露抑制 → 主因 cql_loss=0 | **支持** |
| 抑制されない → 主因は別 | 棄却 |

**数値:** step2000 副露 **13.61%**（online_main 61.25% 比 **−47.64pp**、warmstart 17.59% 比 **−3.98pp**）。  
行動プロファイルも warmstart に近い（立直 34% vs warmstart 23%、副露・流局も同帯）。

---

## 赤選択性（集計のみ）

**ckpt:** step 2,000（`result_step2000.txt`）  
**学習:** なし（集計のみ）

### 背景

online cql=0 → 無差別鳴き（副露 61%, 赤選択性 Δ=−2.74pp）。  
診断B → 副露 13.61% に正常化。  
「赤を選択的に鳴く健全」か「cql が鳴きを潰しただけ = offline 天井の再来」かを、phase4c/4d と同一軸で確認。

### 集計条件

| 項目 | 内容 |
|---|---|
| 赤条件軸 | phase4c / `analyze_aka_conditional.py` と同一（`has_aka` = 局内いずれかの時点で手牌+副露の赤 ≥ 1） |
| 副露 | chi / pon / daiminkan |
| 集計席 | mortal 席のみ |
| test_play | self-play 3000 半荘（29,556 局） |
| 赤保持局 | 局終了スナップショット `aka_held>0`（`analyze_chip_realize.py` 同定義） |

### Step 1: 赤条件別副露率 ★最重要

#### 絶対値（副露率）

| 条件 | 人間(2009) | phase4d lo=0.3 | online step26000 | **診断B step2000** |
|---:|---:|---:|---:|---:|
| **赤あり** | **35.82%** | **14.90%** | **55.45%** | **17.00%** |
| **赤なし** | **33.01%** | **19.92%** | **58.19%** | **14.08%** |
| **Δ (赤あり−赤なし)** | **+2.81pp** | **−5.01pp** | **−2.74pp** | **+2.92pp** |

#### 母集団

| | 人間(2009) | phase4d lo=0.3 | online step26000 | **診断B step2000** |
|---|---:|---:|---:|---:|
| 赤あり局数 (`has_aka`) | 135,682 | 1,832 | 13,177 | **13,691** |
| 赤なし局数 | 161,230 | 2,119 | 14,009 | **15,865** |
| 全体局数 | 296,912 | 3,951 | 27,186 | **29,556** |
| has_aka 率 | 45.70% | 46.37% | 48.47% | **46.32%** |

#### 数値の読み（解釈は保留）

- 診断B: Δ **+2.92pp**（赤あり 17.00% > 赤なし 14.08%）。人間 +2.81pp と同符号。
- 副露絶対値は phase4d warmstart 帯（14–17%）に近く、online_main step26000（55–58%）からは大きく乖離。
- 全体副露率（train.py `behavior_fuuro`、暗槓/加槓含む）= **13.61%**。赤条件別（chi/pon/大明槓のみ）全体 = **15.44%**。

### Step 3: chip実現率 × 副露 × 鳴き和了の赤含有

#### 副露率 vs chip実現率

| | 全体副露率 | chip実現率（赤保持局） | 鳴き和了率（赤保持局） | 赤保持局数 n |
|---|---:|---:|---:|---:|
| phase4d lo=0.3 (1v3) | 17.59% | 20.52% | 3.03% | 1,652 |
| online step26000 (ログ) | 56.63% | 23.26% | 4.33% | 9,091 |
| **診断B step2000** | **13.61%** ※ | **18.56%** | **2.70%** | **12,658** |

※ train.py `behavior_fuuro`（暗槓/加槓含む）。赤条件別 chi/pon/大明槓のみ = 15.44%。

#### 鳴き和了の赤含有（和了時 `agari_detail.num_aka > 0`）

集計: mortal 席の鳴き和了局のみ（赤保持を問わない）。

| | 鳴き和了数 | 赤含有（num_aka>0） | 赤非含有（num_aka=0） |
|---|---:|---:|---:|
| phase4d lo=0.3 (1v3) | 185 | 29.19% (54) | 70.81% (131) |
| online step26000 | 1,712 | 35.16% (602) | 63.90% (1,094) |
| **診断B step2000** | **1,055** | **35.45% (374)** | **64.55% (681)** |

#### 赤保持局の和了経路内訳（step2000 ログ）

| 指標 | phase4d lo=0.3 | online step26000 | **診断B step2000** |
|---|---:|---:|---:|
| 赤保持局数 | 1,652 | 9,091 | **12,658** |
| chip実現率 | 20.52% | 23.26% | **18.56%** (2,349/12,658) |
| 鳴き和了率 | 3.03% | 4.33% | **2.70%** (342/12,658) |

参考: 鳴き和了率 offline 天井 = 3.03%、目標 = 6%。

#### 判定の肝（事前基準・結論は保留）

| パターン | 条件 |
|---|---|
| スイートスポット候補 | Δ が正（人間 +2.81pp 方向）**かつ** 鳴き和了率 > offline 天井 3% |
| offline 天井逆戻り | Δ がフラット/負 **かつ** 鳴き和了率 ≈ 3% **かつ** 副露 warmstart 帯 |

**実施内容:** 集計のみ。学習・cql / beta_sel / lambda_opp 変更なし。上記判定基準に対する結論は未実施。

---

## 集計メタ

| 項目 | パス |
|---|---|
| config | `/home/gamba/mahjong/runs/online_diag_b/config.toml` |
| trainer ログ | `/home/gamba/mahjong/runs/online_diag_b/logs/trainer.log` |
| step2000 結果 | `/home/gamba/mahjong/runs/online_diag_b/result_step2000.txt` |
| 起動スクリプト | `/home/gamba/mahjong/runs/run_online_diag.sh` |
| 総合報告 | `freeparlor/docs/online_diag_fuuro_summary.md` |
| 診断A | `freeparlor/docs/online_diag_a_lambda_opp_zero.md` |
| 赤選択性比較 | `freeparlor/docs/online_fuuro50_aka_selectivity.md` |
| test_play ログ | `/home/gamba/mahjong/runs/online_diag_b/test_play`（3000 半荘） |
| 赤条件別 | `python freeparlor/scripts/analyze_aka_conditional.py --models diag_b_step2000:/home/gamba/mahjong/runs/online_diag_b/test_play` |
| chip実現・鳴き和了・赤含有 | `analyze_chip_realize.py` 相当ロジック（ログから直接集計） |

**実施内容:** 診断B のみ。lambda_opp と cql は同時変更していない。
