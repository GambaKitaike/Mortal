# 副露率ジャンプ考古学 — β=1 激減後の「一気に上げた変更」

**日付:** 2026-07-01  
**方法:** 既存 run ログ・config・docs・git の読み取りのみ（新規学習なし）

---

## 要約（結論先出し）

| 境目 | 副露率 Δ | 特定した変更 | 分類 | 確度 |
|---|---:|---|---|---|
| **A** β=1 → β=0.3（Phase4 再学習） | **+13.2pp**（1v3: 0.51→13.67%） | `[env] beta = 1.0 → 0.3` | **(a) 報酬** | **Yes** |
| **B** Phase4d lo=0.3 → online step2000 | **+43.7pp**（17.59→61.25%） | `online=true` + **online 時 cql_loss=0** + self-play test_play | **(b) 学習構成** | **Yes** |
| C Phase4 β=0.3 → Phase4d lo=0.3 | +3.9pp（13.67→17.59%） | `lambda_opp=0.3` 導入 | (a) 報酬 | Yes（小幅） |

**ユーザー記憶の「一気に上げた」に最も一致するのは境目 B**（+43.7pp、2日以内）。  
境目 A は β=1 崩壊からの**最初の回復**（+13pp 級）で、同日〜翌日の β スイープ。

**棄却された候補:** `lambda_opp` は offline では +3.9pp 程度。online 暴走の主因ではない（診断A: lo=0 でも 83.81%）。

---

## Step 1: 副露率時系列表

### 集計方法

- **打牌統計**（副露率・和了率・流局率）: `libriichi.stat.Stat.from_dir(log_dir, 'mortal')` — 2026-07-01 再計算
- **副露和了率**: `call_channel_diag` Part A 定義 = trainee和了 に対する副露和了比率。offline Phase4 系は Part A 未実行のため、記録済み docs のみ引用（空白は「—」）
- **avg_rank**: competitive 指標は `test_play vs baseline×3` のみ参考。1v3 self-play / online self_play は ≈2.5 で無意味

### 評価経路の凡例

| 記号 | 意味 |
|---|---|
| **1v3-SP** | 1席=mortal(model.pth) vs 3席=同一 champion.pth 自己対戦（Phase4/4d 標準） |
| **TP-comp** | test_play: 1席=学習中 mortal vs 3席=grp_baseline（avg_rank 参考可） |
| **TP-SP** | test_play self_play=true（4席同一 ckpt、avg_rank≈2.5） |

### 時系列表（Phase4 → 4c → 4d → online）

| # | 時期 | run / ckpt | 評価 | offline/online | 副露率 | 副露和了率 | 和了率 | 流局率 | β | λ_opp | noten | min_q | cql_online | 根拠ログ |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 06-22 | P4 **β=1.0** 192×40 | **TP-comp** | offline | **4.39%** | 0.14%† | 14.03% | 84.36% | 1.0 | — | — | 5 | — | `beta1_192x40/test_play` n=3000 |
| 2 | 06-22 | P4 **β=1.0** | **1v3-SP** | offline | **0.51%** | — | 9.39% | 62.63% | 1.0 | — | — | 5 | — | `sweep_eval/beta1/1v3` n=400 |
| 3 | 06-22 | P4 β=0 | 1v3-SP | offline | 16.97% | — | 21.38% | 15.45% | 0 | — | — | 5 | — | `sweep_eval/beta0/1v3` |
| 4 | 06-22~23 | P4 β=0.1 | 1v3-SP | offline | 11.56% | — | 20.12% | 19.95% | 0.1 | — | — | 5 | — | `sweep_eval/beta0_1/1v3` |
| 5 | 06-23 | P4 **β=0.3** | **1v3-SP** | offline | **13.67%** | — | 20.61% | 18.58% | 0.3 | — | — | 5 | — | `sweep_eval/beta0_3/1v3` |
| 6 | 06-23 | P4 β=0.3 | TP-comp | offline | 7.77% | — | 36.40% | 62.31% | 0.3 | — | — | 5 | — | `beta0_3_192x40/test_play` |
| 7 | 06-23 | P4 β=0.5 | 1v3-SP | offline | 7.42% | — | 19.59% | 22.75% | 0.5 | — | — | 5 | — | `sweep_eval/beta0_5/1v3` |
| 8 | 06-23 | **Phase4c**（分析のみ） | — | — | — | — | — | — | 各β | — | — | — | — | 独立 run なし。β=0.3 行 #5 が代表 |
| 9 | 06-24 | P4d lo=0.0 | 1v3-SP | offline | 11.95% | — | 20.14% | 19.72% | 0.3 | **0.0** | 0.0 | 5 | — | `phase4d/eval/phase4d_lo00/1v3` |
| 10 | 06-24 | P4d **lo=0.3** | **1v3-SP** | offline | **17.59%** | 3.03%‡ | 20.63% | 18.35% | 0.3 | **0.3** | 0.0 | 5 | — | `phase4d/eval/phase4d_lo03/1v3` |
| 11 | 06-24 | P4d lo=0.6 | 1v3-SP | offline | 20.29% | — | 20.61% | 17.40% | 0.3 | 0.6 | 0.0 | 5 | — | `phase4d/eval/phase4d_lo06/1v3` |
| 12 | 06-25 | online **warmstart** | TP-SP | offline ckpt | 16.58% | 3.03%‡ | 20.79% | 17.59% | 0.3 | 0.3 | 0.0 | 5 | — | `online_main/eval_phase4d_test_play` |
| 13 | 06-26 | online_main **step2000** | **TP-SP** | **online** | **61.25%** | 7.24%‡ | — | — | 0.3 | 0.3 | 0.0 | 5 | **false** | TB‡‡ / `online_fuuro50_aka_selectivity.md` |
| 14 | 06-26 | online_main step26000 | TP-SP | online | 56.63% | 3.48%‡ | 20.09% | 20.40% | 0.3 | 0.3 | 0.0 | 5 | false | `online_main/test_play` n=3000 |
| 15 | 06-26 | **diag_a** step2000 | TP-SP | online | 65.46% | — | 21.53% | 14.66% | 0.3 | **0.0** | 0.0 | 5 | false | `online_diag_a/test_play` |
| 16 | 06-27 | **diag_b** step2000 | TP-SP | online | **13.61%** | 4.92%§ | 20.73% | 17.96% | 0.3 | 0.3 | 0.0 | 1 | **true** | `online_diag_b/test_play` |
| 17 | 06-28~29 | mqw03 step2000 | TP-SP | online | 35.61% | 36.60%§ | 21.45% | 15.02% | 0.3 | 0.3 | 0.0 | 0.3 | true | `online_cql_mqw03/test_play_step2000` |
| 18 | 06-29 | mqw03 final (step16000) | TP-SP | online | 42.72% | 28.21%§ | 20.97% | 16.96% | 0.3 | 0.3 | 0.0 | 0.3 | true | `online_cql_mqw03/test_play` |

脚注:

- † `beta1_pnl_salvage.md` 再計算値（TP-comp）。call_channel Part A 未実行
- ‡ 赤保持局鳴き和了率（`aka_held_call_win_rate`）— 副露和了率の proxy。offline baseline = 3.03%
- ‡‡ step2000 の json.gz は上書き消滅。TB `test_play_behavior_fuuro` + monitor.log が一次証拠
- § call_channel Part A（trainee和了内の副露和了%）— `mqw03_call_channel.md`, `call_channel_diag.md`

### 境目の特定

```
副露率 (1v3-SP / 同一軸)
0.51%  β=1 ──(+13.2pp)──► 13.67%  β=0.3 ──(+3.9pp)──► 17.59%  P4d lo=0.3
                                                              │
                                    (+43.7pp, online step2000)│
                                                              ▼
                                                         61.25%  online_main
```

**最大ジャンプ = 行 #10 → #13**（offline 17.59% → online 61.25%）。  
**β=1 からの直接回復 = 行 #2 → #5**（0.51% → 13.67%）。

---

## Step 2: 境目 config diff

### 境目 A: β=1 → β=0.3（Phase4 再学習）

**比較:** `phase4_chip_beta1_192x40.toml` vs `phase4_chip_beta0_3_192x40.toml`  
**日付:** 2026-06-22（β=1 本番）→ 2026-06-22 22:42（β=0.3 学習開始、`beta0_3_192x40/train.log`）

| パラメータ | β=1 | β=0.3 | 副露への影響 |
|---|---|---|---|
| **`[env] beta`** | **1.0** | **0.3** | **主因** — チップ報酬全量→健全域 |
| lambda_opp | （未設定=0） | （未設定=0） | 変化なし |
| online | false | false | 変化なし |
| min_q_weight | 5 | 5 | 変化なし |
| その他 | paths のみ | paths のみ | — |

**副露率:** 1v3 0.51% → 13.67%（**+13.16pp**）。TP-comp 4.39% → 7.77%（+3.38pp）。

---

### 境目 B: Phase4d lo=0.3 warmstart → online_main step2000 ★最大ジャンプ

**比較:** `phase4d_lo03_beta0_3_192x40.toml` vs `runs/online_main/config.toml`（warmstart 時点）  
**日付:** 2026-06-24（P4d 学習完了）→ 2026-06-26 03:20（online step2000, `monitor.log`）

| パラメータ | P4d lo=0.3 (offline) | online_main | 備考 |
|---|---|---|---|
| **`[control] online`** | **false** | **true** | offline 教師データ → online 自己対戦 |
| **`[test_play] self_play`** | （未設定=false） | **true** | 4席同一 ckpt 評価 |
| **`cql_loss` 計算** | 常にあり | **`not online` 時のみ**（= online デフォルト **0**） | `train.py` L273–274 |
| **`[cql] enable_online`** | — | **未設定=false** | diag_b で true+min_q=1 なら 13.61% に抑制 |
| dataset globs | tenhou 2009 | **[]**（空） | replay buffer 駆動 |
| train_play | default 1本 | **client0–2** 並列 | online 自己対戦 |
| beta_sel_* | — | max=0.3, warmup=2000 | step2000 時 beta_sel≈0 |
| freeze_bn.mortal | false | **true** | BN 固定 |
| test_every | 20000 | **2000** | 早期 test_play |
| **beta** | 0.3 | 0.3 | **変化なし** |
| **lambda_opp** | 0.3 | 0.3 | **変化なし** |
| noten_factor | 0.0 | 0.0 | 変化なし |
| min_q_weight | 5 | 5 | 変化なし（ただし online では cql 未適用） |

**副露率:** 17.59%（1v3-SP）→ **61.25%**（TP-SP step2000, TB）= **+43.66pp**

**切り分け結果（診断A/B, 2026-06-26~27）:**

| 仮説 | 操作 | step2000 副露 | 判定 |
|---|---|---:|---|
| lambda_opp が主因 | diag_a: **λ=0** | **83.81%**（61%超） | **棄却** |
| cql_loss=0 が主因 | diag_b: **enable_online=true, min_q=1** | **13.61%** | **支持** |
| warmstart 自体 | P4d lo=0.3 1v3 | 17.59% | 正常 |

---

### 境目 C: Phase4 β=0.3 → Phase4d lo=0.3（参考）

**比較:** `phase4_chip_beta0_3_192x40.toml` vs `phase4d_lo03_beta0_3_192x40.toml`  
**日付:** 2026-06-23 → 2026-06-24

| パラメータ | P4 β=0.3 | P4d lo=0.3 |
|---|---|---|
| beta | 0.3 | 0.3 |
| **lambda_opp** | — | **0.3** |
| **noten_factor** | — | **0.0** |
| その他 | — | paths のみ |

**副露率:** 13.67% → 17.59%（**+3.92pp**）。lambda_opp スイープ内でも lo=0.0→0.3 で +5.64pp（`phase4d_sweep_results.md`）。

---

## Step 3: docs / git 裏取り

### docs 引用（時系列）

| 日付 | ファイル | 引用要点 |
|---|---|---|
| 06-22 | `phase4_chip.md` | 「β=1.0 では副露 **16.97% → 0.51%** と激減」「健全域 **β≤0.3**」 |
| 06-23 | `next_steps.md` L27 | 「β=1.0 で打牌崩壊（**副露0.51%**・流局62.6%）」 |
| 06-23 | `phase4c_human_aka_conditional.md` | Phase4c = 分析フェーズ。独立 run なし |
| 06-25 | `phase4d_aka_opp_probe.md` | lambda_opp 導入意図:「offline で副露の符号が反転するかを測るプローブ」 |
| 06-25 | `online_main_report.md` L86–98 | warmstart 副露 **17.59%** を baseline として online 開始 |
| 06-26 | `online_fuuro50_aka_selectivity.md` L59–65 | step2000 副露 **61.25%**（warmstart 17.59% から +43.7pp） |
| 06-27 | `online_diag_fuuro_summary.md` L59–77 | **cql_loss=0 が主因**、lambda_opp は主因ではない |
| 06-29 | `next_steps_2.md` L77 | 「min_q_weight=0 (online_main): **無差別鳴き(副露61%暴走)**」 |

### git コミット（時系列）

| 日付 | hash | メッセージ | 副露関連変更 |
|---|---|---|---|
| 06-22 | `db334e4` | phase4: chip reward beta + libriichi agari_detail | `reward_calculator.py` に β·chip 項 |
| 06-22 | `7e97801` | phase4: chip reward beta=1 production | β=1 run 実行 |
| 06-23 | `05ee679` | phase4: beta sweep results and configs | β スイープ各点 config |
| 06-23 | `3e67b4b` | phase4c: human aka-conditional analysis | コード変更なし（分析） |
| 06-25 | `e119ecd` | phase4d: probe partially success | **`lambda_opp` / `noten_factor` 追加**（`reward_calculator.py`, `dataloader.py`） |
| 06-25 | `c18dd1a` | Enhance chip reward mechanism and integrate beta selection | **online TD / beta_sel**、`train.py` online 分岐 |
| 06-26~27 | （run ログ） | online_main / diag_a / diag_b 実行 | config diff + 診断結果が docs に記録 |
| 06-29 | `eb104f2` | Enhance logging for player and server | `train.py` behavior ログ改善（事後） |

`reward_calculator.py` の opp 項（e119ecd）:

```python
opp = -beta * lambda_opp * chip_value * aka_held * w * fire
```

`train.py` の CQL 分岐（online 暴走の mechanistic 根拠）:

```python
if not online or enable_cql_online:
    cql_loss = q_main.logsumexp(-1).mean() - q.mean()
```

online_main は `enable_cql_online` 未設定 → online 時 **cql_loss=0**。

---

## 結論

### 判定: **Yes（2段階、用途別）**

1. **β=1 崩壊（4.39% / 0.51%）から offline 健全域へ戻した変更**  
   - **変更:** `[env] beta = 1.0 → 0.3` による Phase4 再学習（2026-06-22~23）  
   - **分類:** **(a) 報酬の変更**  
   - **根拠:** config diff（Step2 境目A）、`phase4_chip.md` β カーブ、`Stat.from_dir` 再計算（Step1 #2→#5）

2. **ユーザー記憶の「一気に上げた」に該当する最大ジャンプ**  
   - **変更:** **offline → online 自己対戦** + **`train.py` の online 時 cql_loss 無効化**（`enable_cql_online` デフォルト false）  
   - **分類:** **(b) 学習構成の変更**（報酬パラメータ β/λ_opp は warmstart から不変）  
   - **根拠:** Step1 #10→#13（+43.7pp）、Step2 境目B config diff、診断A/B（`online_diag_fuuro_summary.md`）、git `c18dd1a`

### 棄却・部分確定

| 候補 | 結果 |
|---|---|
| lambda_opp 導入（P4d） | offline で +3.9pp のみ。online 61% の主因 **ではない**（diag_a で棄却） |
| lambda_opp が online で鳴きを増幅 | **棄却** — lo=0 でも 83.81% |
| beta_sel warmup | step2000 時 beta_sel=0（monitor.log）。副露 61% は beta_sel 以前から発生 |
| min_q_weight 変更 | online_main では cql 自体が off。diag_b で min_q=1 導入時 13.61% に **抑制** |

### 復元できなかった空白

- offline Phase4 各 β の **call_channel Part A 副露和了率**（ログは存在、Part A 未実行）
- online_main **step2000 の test_play json.gz**（上書き消滅。TB + monitor.log で代替）
- online_main の **step 別 test_play ディレクトリ**（`test_play_step*` なし。mqw03 のみ step 別保存）

---

## 再現コマンド（読み取りのみ）

```bash
# Step1 打牌統計（例）
export MORTAL_CFG=/home/gamba/mahjong/Mortal/mortal/config.example.toml
/home/gamba/miniconda3/envs/mortal/bin/python -c "
import sys; sys.path.insert(0,'/home/gamba/mahjong/Mortal/mortal')
from libriichi.stat import Stat
for label, d in [
  ('P4b1_1v3','/home/gamba/mahjong/runs/phase4/sweep_eval/beta1/1v3'),
  ('P4b03_1v3','/home/gamba/mahjong/runs/phase4/sweep_eval/beta0_3/1v3'),
  ('P4d_lo03','/home/gamba/mahjong/runs/phase4d/eval/phase4d_lo03/1v3'),
  ('online_final','/home/gamba/mahjong/runs/online_main/test_play'),
  ('diag_b','/home/gamba/mahjong/runs/online_diag_b/test_play'),
]:
  s=Stat.from_dir(d,'mortal',True)
  print(label, 'fuuro=%.2f%% agari=%.2f%% ryukyoku=%.2f%%' % (
    s.fuuro_rate*100, s.agari_rate*100, s.ryukyoku_rate*100))
"

# Step2 config diff
diff -u freeparlor/configs/phase4_chip_beta1_192x40.toml \
        freeparlor/configs/phase4_chip_beta0_3_192x40.toml
diff -u freeparlor/configs/phase4d_lo03_beta0_3_192x40.toml \
        /home/gamba/mahjong/runs/online_main/config.toml
```

---

## sanity checklist

- [x] 副露率は Stat 打牌統計（avg_rank と混同しない旨を表に明記）
- [x] run ごとに offline/online・評価経路（1v3-SP / TP-comp / TP-SP）を区別
- [x] 変更特定は config diff / git / docs の一次証拠に紐づけ（憶測なし）
- [x] 新規学習・GPU 実験なし
