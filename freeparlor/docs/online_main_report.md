# Q_chip Online 本番学習 — セットアップ・中間報告

**日付:** 2026-06-25  
**環境:** WSL2 (mahjong), conda `mortal`, RTX 5060 単機（8151 MiB）  
**目的:** layer1–3 検証済み Q_chip TD を online 自己対戦で初稼働。warmup 明けに `beta_sel` 0→0.3 を入れ、赤保持局の鳴き和了率が offline 天井（3%台）を超えられるか監視付きで検証する。

---

## 概要

| 項目 | 内容 |
|---|---|
| run ディレクトリ | `/home/gamba/mahjong/runs/online_main/` |
| warm-start | `phase4d_lo03/mortal.pth` → `mortal_warmstart.pth`（steps=0, optimizer/scheduler 除外） |
| 構成 | server×1, trainer×1, **client×3**（throughput 計測と同方式） |
| 停止監視 | `monitor_main.py`（test_play ごとに判定 → `monitor.log` 追記） |
| 現状 | **学習稼働中**（2026-06-26 集計検証時点 step24000+） |

---

## 実装・成果物

### runs/online_main/

| パス | 内容 |
|---|---|
| `config.toml` | 本番 config（`test_every=2000`, `test_play.self_play=true`） |
| `baseline.json` | 停止条件基準値（phase4d lo=0.3 1v3 eval から算出） |
| `compute_baseline.py` | baseline 再計算スクリプト |
| `monitor_main.py` | 停止条件付き監視ループ |
| `run.sh` | server / trainer / client×3 / monitor 一括起動 |
| `mortal_warmstart.pth` | warm-start ckpt |
| `mortal.pth` | 現行学習 state（warm-start コピー） |
| `logs/` | server / trainer / client ログ |
| `tb/` | TensorBoard |
| `monitor.log` | 監視判定ログ（Step 0 成立・monitor 起動後に生成） |
| `eval_phase4d_test_play.py` | phase4d 同経路再評価スクリプト（集計検証用） |
| `eval_phase4d_test_play.log` | 再評価結果ログ |
| `report.json` / `report.md` | 停止時に自動生成（未生成） |

### コード変更（最小）

| ファイル | 変更 |
|---|---|
| `mortal/player.py` | `[test_play] self_play = true` で test_play を自己対戦化（avg_rank≈2.5 を baseline と比較可能に） |
| `mortal/train.py` | `test_play/behavior/ryukyoku` を TensorBoard に追加 |

**触っていないもの:** Q_main 経路、報酬合成、層1/2、`beta_sel` / `chip_weight` / `n_step` の学習中変更。

---

## 学習パラメータ

```toml
# [env]
beta_sel_max = 0.3
beta_sel_warmup_steps = 2000
beta_sel_ramp_steps = 2000
chip_n_step = 3
chip_target_tau = 0.005
chip_weight = 1.0
lambda_opp = 0.3

# [control]
save_every = 400
test_every = 2000
submit_every = 400
batch_size = 128
version = 4

# [resnet]
conv_channels = 192
num_blocks = 40
```

### beta_sel スケジュール

| steps | beta_sel |
|---:|---:|
| 0 – 1999 | 0.0 |
| 2000 – 3999 | 0 → 0.3 線形 |
| ≥ 4000 | 0.3 固定 |

---

## Baseline（phase4d lo=0.3）

集計: `analyze_chip_realize.analyze_eval_dir` + `Stat.from_dir(..., 'mortal')`  
ログ: `/home/gamba/mahjong/runs/phase4d/eval/phase4d_lo03/1v3`（400 半荘 self-play）

| 指標 | 値 | 停止条件での用途 |
|---|---:|---|
| **赤保持局 鳴き和了率** ★ | **3.03%** | 成功: ≥6.0% / 天井: 傾き≈0 かつ 6%未満 |
| **放銃率** ★ | **13.26%** | 緊急: baseline+5pp超 / 成功: +3pp以内 |
| **avg_rank** | **2.485** | 緊急・成功: ±0.15 |
| 流局率 | 18.35% | 記録 |
| 和了率 | 20.63% | 参考 |
| 副露率 | 17.59% | 参考 |
| 立直率 | 23.49% | 参考 |
| 赤保持局数 | 1652 | 母集団 |
| aka_chip_realize_rate | 20.52% | 補助 |

出典: `runs/online_main/baseline.json`（2026-06-25 22:07 算出）

---

## 監視指標（test_play ごと）

TensorBoard タグ（既存 + 追加）:

| タグ | 内容 |
|---|---|
| `test_play/aka_held_call_win_rate` | 赤保持局 鳴き和了率 ★主指標 |
| `test_play/behavior/houjuu` | 放銃率 ★健全性 |
| `test_play/avg_ranking` | 平均順位 |
| `test_play/behavior/ryukyoku` | 流局率（今回追加） |
| `loss/chip_loss`, `loss/dqn_loss` | finite 確認 |
| `hparam/beta_sel` | スケジュール確認 |
| `test_play/aka_held_chip_realize_rate` | 補助 |

test_play 条件: **3000 局 self-play**（`test_play.self_play=true`）

---

## 停止条件（monitor_main.py）

### 緊急停止（即時・自動）

| 条件 |
|---|
| 放銃率 > baseline + **5pp** |
| avg_rank が 2.485 から **±0.15** 超乖離 |
| chip_loss / dqn_loss が NaN・inf、または直近中央値の **10 倍超** が継続 |

### 成功判定（報告停止）

`step > 4000`（beta_sel=0.3 フル稼働）後、以下を**同時**に満たす:

- 鳴き和了率 ≥ **6.0%**
- 放銃率 ≤ baseline + **3pp**
- avg_rank が 2.485 ± **0.15** 内

### 天井判定（報告停止）

beta_sel=0.3 フル稼働後 **最低 4000 step** 経過、かつ:

- 直近 test_play 数点の鳴き和了率線形回帰傾き ≈ **0**（< 0.05 pp/eval）
- 鳴き和了率 **6% 未満**

### 継続

上記いずれにも該当しない場合は継続。ramp 中（step 2000–4000）に放銃率が上昇中でも +3pp 以内なら `WATCH` ログを出して続行。

---

## 起動タイムライン

| 時刻 (JST) | イベント |
|---|---|
| 22:07 | baseline 算出完了（1v3 eval から） |
| 22:08 | 初回 `run.sh` 起動（server / trainer / client×3） |
| 22:09–22:32 | 初回 client 生成進行。ただし **online_throughput の stale client** が同一 port 5000 に接続し、800 ファイルを trainer に drain（**データ混入**） |
| 22:32 | trainer Step 0 付近まで進行したが、混入のため **再起動判断** |
| 22:43 | 旧 run の monitor が stale PID で残留（後に停止） |
| **22:56** | **クリーン再起動** — buffer クリア、warmstart から `mortal.pth` 復元、stale throughput client 停止 |
| 22:56– | client×3 が初回 800 局生成中（`beta_sel=0.0`）。Step 0 未成立 → monitor 未起動 |

---

## 現状スナップショット（報告時点）

| 項目 | 状態 |
|---|---|
| プロセス | server / trainer / client×3 / `run.sh` 稼働中 |
| trainer steps | **0**（初回 drain 待ち） |
| client submit | **0 回**（初回 800 局バッチ生成中） |
| beta_sel | **0.0** |
| monitor.log | **未生成**（Step 0 成立後に monitor 起動） |
| test_play 結果 | **なし**（step 0 時点で 1 回目が走る予定） |
| TensorBoard | イベントファイル生成済み、scalar データは Step 0 以降 |

### 見込みタイムライン（3 client 並列、~0.67 step/s 換算）

| milestone | 目安 |
|---|---|
| Step 0 成立 | 初回 800 局×3 完了後（~30–45 分/batch、GPU 共有） |
| step 2000（beta_sel ramp 開始） | Step 0 後 ~50 分 |
| step 4000（beta_sel=0.3 固定） | Step 0 後 ~100 分 |
| 1 回目 test_play（step 2000） | ramp 開始と同時 |

---

## 起動・監視コマンド

```bash
# 起動（baseline 未生成時は自動算出）
/home/gamba/mahjong/runs/online_main/run.sh 72

# 監視
tail -f /home/gamba/mahjong/runs/online_main/monitor.log
tail -f /home/gamba/mahjong/runs/online_main/logs/trainer.log
tensorboard --logdir /home/gamba/mahjong/runs/online_main/tb
```

環境変数: `MORTAL_CFG=/home/gamba/mahjong/runs/online_main/config.toml`

---

## 主指標推移（step 2000–24000）

test_play: **3000 半荘 self-play**、固定 seed `[10000, 10750) / 0x2000`（trainer ログ確認）。  
鳴き和了率出典: TensorBoard `test_play/aka_held_call_win_rate`。放銃率: TB サブディレクトリ `test_play_behavior_houjuu`（タグ `test_play/behavior`）。

| step | beta_sel | 鳴き和了率% | n (近似) | k | 95% CI | 放銃率% |
|-----:|---------:|------------:|---------:|--:|--------|--------:|
| 2000 | 0.000 | 7.24 | 10130 | 734 | [6.74, 7.75] | 15.78 |
| 4000 | 0.300 | 4.84 | 10130 | 490 | [4.42, 5.26] | 16.58 |
| 6000 | 0.300 | 4.24 | 10130 | 430 | [3.85, 4.64] | 16.37 |
| 8000 | 0.300 | 3.90 | 10130 | 395 | [3.52, 4.27] | 16.07 |
| 10000 | 0.300 | 4.32 | 10130 | 437 | [3.92, 4.71] | 16.00 |
| 12000 | 0.300 | 3.99 | 10130 | 404 | [3.61, 4.37] | 16.41 |
| 14000 | 0.300 | 4.46 | 10130 | 451 | [4.05, 4.86] | 14.65 |
| 16000 | 0.300 | 6.13 | 10130 | 621 | [5.66, 6.60] | 16.30 |
| 18000 | 0.300 | 6.75 | 10130 | 684 | [6.26, 7.24] | 15.12 |
| 20000 | 0.300 | 6.21 | 10130 | 629 | [5.74, 6.68] | 15.96 |
| 22000 | 0.300 | 4.15 | 10130 | 420 | [3.76, 4.54] | 16.31 |
| 24000 | 0.300 | 3.48 | **10130** | **353** | **[3.13, 3.84]** | 15.19 |

- **n 実測:** step24000 残存 `test_play/` ログから `analyze_chip_realize` → `aka_held_kyoku=10130`, `call_win=353`
- 他 step の n/k は上記実測 n で近似（同一 seed だが方策変化で局経路は変わりうる）
- 95% CI: 二項正規近似 `p ± 1.96·sqrt(p(1-p)/n)`
- baseline 鳴き和了率 3.03%（1v3 eval, n=1652）、放銃率 13.26%

---

## 2026-06-26 鳴き和了率 集計検証

**目的:** step16000–20000 の山（6–6.75%）と warmup step2000 の 7.24% を、母集団 n・beta_sel 実装・評価経路の観点で切り分け。学習は継続、パラメータ変更なし。

### Step 1: 母集団 n

- test_play ログは毎回 `shutil.rmtree` で上書き。過去 step の n は遡及不可
- step24000 実測 n=**10130**（3000 半荘）。数十～百ではなく **数千規模**
- 3% 台の標準誤差 ≈ ±0.35pp、7% 台 ≈ ±0.50pp
- 16000–20000 の CI は互いに重なるが、8000(3.90%) や 24000(3.48%) とは非重複

### Step 2: warmup 7.24% の切り分け

#### beta_sel 実装

| 確認項目 | 結果 |
|---------|------|
| `train.py` | `test_play(..., beta_sel=calc_beta_sel(steps))` |
| `player.py` → `MortalEngine` | `beta_sel` を engine に渡す |
| `engine.py` v4 | `beta_sel > 0` のときのみ `q_total(q_main, q_chip, beta_sel)` |
| step2000 TB `hparam/beta_sel` | **0.0000** |
| `calc_beta_sel(2000)` | warmup 終端 t=0 → **0.0** |

**test_play が beta_sel スケジュールを無視する実装ズレは未検出。**

#### phase4d lo=0.3 同経路再評価

スクリプト: `runs/online_main/eval_phase4d_test_play.py`  
config: `online_main/config.toml`（`self_play=true`, 3000 局）  
ckpt: `phase4d_lo03/mortal.pth`（warmstart 元・**未学習**）

| 条件 | beta_sel | 鳴き和了率% | n | k | 95% CI | 放銃率% | 副露率% |
|-----|---------:|------------:|--:|--:|--------|--------:|--------:|
| 同経路再評価 | 0.0 | **3.07** | 12820 | 393 | [2.77, 3.36] | 13.67 | 17.15 |
| 同経路再評価 | 0.3 | **3.02** | 12972 | 392 | [2.73, 3.32] | 13.44 | 16.03 |
| baseline (1v3 eval, 400半荘) | — | **3.03** | 1652 | 50 | ±0.83pp | 13.26 | 17.59 |
| online_main step2000 | 0.0 | **7.24** | ~10130 | ~734 | [6.74, 7.75] | 15.78 | — |
| online_main step24000 | 0.3 | **3.48** | 10130 | 353 | [3.13, 3.84] | 15.19 | 50.74 |

ログ: `runs/online_main/eval_phase4d_test_play.log`

#### 評価経路の比較（参考）

| 経路 | 鳴き和了率% | 副露率% | avg_rank | 備考 |
|-----|------------:|--------:|---------:|------|
| phase4d 学習時 test_play（vs baseline 3席） | 1.29 | 5.95 | 1.006 | `self_play=false`、参考外 |
| phase4d 1v3 eval（baseline 定義） | 3.03 | 17.59 | 2.485 | 400 半荘 |
| phase4d self_play 再評価 | 3.02–3.07 | 16–17 | 2.500 | 3000 半荘、今回 |
| online_main self_play step24000 | 3.48 | 50.74 | 2.500 | online 学習後 |

**観察（解釈は保留）:**

- fresh phase4d ckpt + `beta_sel=0` 同経路 → **3.07%**（baseline 3.03% と同程度）。beta_sel 0/0.3 の差はほぼなし
- step2000 の 7.24% は beta_sel≠0 では説明できない。当時の重みは warmstart から **2000 step 学習済み**
- online 学習後は副露率が ~17% → ~51% に増加（self_play 経路）

### Step 3: 放銃率（step18000 近傍）

- `save_every=400` だが `mortal.pth` 上書きのみで **step18000 ckpt は未残存**
- TB サブディレクトリから遡及: step18000 放銃率 **15.12%**（baseline +1.86pp）
- step16000–20000 山: 鳴き和了率上昇と引き換えの放銃率急増は TB 上は見えない（15–16% 台）

### 集計メタ

| 項目 | 内容 |
|------|------|
| 再評価スクリプト | `runs/online_main/eval_phase4d_test_play.py` |
| 再評価ログ | `runs/online_main/eval_phase4d_test_play.log` |
| monitor | `runs/online_main/monitor.log` |
| TB | `runs/online_main/tb/` |
| 検証実施 | 学習継続中（beta_sel / chip_weight 未変更） |

---

## 解釈メモ

- 初回起動の throughput client 混入は buffer クリア + warmstart 復元で対処済み
- 鳴き和了率の成功/天井判定は本検証では未実施（土台確認のみ）
- 関連 doc: `online_r_chip_layer3.md`, `online_throughput_parallel.md`, `phase4d_chip_realize.md`

---

## 2026-06-26 障害・修正・再起動

### 障害

| 項目 | 内容 |
|---|---|
| 発生 | 23:18、trainer step 3/400 でクラッシュ |
| エラー | `RuntimeError: No such file or directory` — drain 内 gzip を dataloader が読み込み中にファイル消失 |
| 原因 | (1) server `handle_drain` が毎回 **drain_dir 全体を削除**してから move → 他 trainer / stale プロセスの drain と競合 (2) stale `run_train.py` / `run_client.py` が port 5000 を共有 |
| 副作用 | trainer 停止中も client が生成継続、server が `beta_sel=0.3` を返す（stale trainer の param 汚染の疑い） |

### 修正

| ファイル | 変更 |
|---|---|
| `mortal/server.py` | drain ごとに **`drain/{generation}/` サブディレクトリ**を作成。旧 drain を削除しない |
| `mortal/train.py` | epoch 終了後に **自分の drain サブディレクトリを削除**。空 file_list は skip |
| `runs/online_main/run.sh` | 起動前に **全 stale プロセス停止** + buffer/drain クリア + warmstart 復元。**trainer watchdog**（クラッシュ時 10 秒後に自動再起動） |

### 再起動（2026-06-26 01:56 JST）

- 全プロセス停止 → buffer/drain クリア → warmstart から `mortal.pth` 復元
- server / trainer(watchdog) / client×3 起動
- `beta_sel=0.0` を確認（正常）
- Step 0 成立待ち（初回 800 局生成中、~30 分見込み）

---

## 関連パス

| 用途 | パス |
|---|---|
| 本番 config | `/home/gamba/mahjong/runs/online_main/config.toml` |
| baseline | `/home/gamba/mahjong/runs/online_main/baseline.json` |
| warm-start 元 | `/home/gamba/mahjong/runs/phase4d/phase4d_lo03/mortal.pth` |
| 再評価ログ | `/home/gamba/mahjong/runs/online_main/eval_phase4d_test_play.log` |
| 最終 ckpt（停止時） | `/home/gamba/mahjong/runs/online_main/mortal.pth` |
