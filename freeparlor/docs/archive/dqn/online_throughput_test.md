# Online 3プロセス疎通 + スループット計測

**日付:** 2026-06-25  
**環境:** WSL2 (mahjong), conda `mortal`, RTX 5060 単機  
**目的:** online 自己対戦学習（Q_chip TD・layer3）の 3 プロセス疎通と速度計測（性能評価はしない）

## 前提

| 項目 | 値 |
|---|---|
| 起動 | `runs/` spawn ランチャ経由（CUDA fork 回避） |
| warm-start | phase4d lo=0.3 ckpt → `mortal_warmstart.pth`（steps=0, optimizer/scheduler 除外） |
| モデル | 192×40, version=4 |
| 設定 | `/home/gamba/mahjong/runs/online_throughput/config.toml` |
| beta_sel | warmup 中（2000 step 未満 → 0 維持） |

### 起動コマンド

```bash
export MORTAL_CFG=/home/gamba/mahjong/runs/online_throughput/config.toml

# 3 プロセス（別ターミナル）
python /home/gamba/mahjong/runs/run_server.py
python /home/gamba/mahjong/runs/run_train.py
python /home/gamba/mahjong/runs/run_client.py
```

---

## Step 0: 3プロセス疎通

**結果: 循環成立**

| チェックポイント | 結果 |
|---|---|
| server 起動 | `listening on 127.0.0.1:5000` |
| client `get_param` | `param has been updated (beta_sel=0.0)` |
| 対局生成 → `submit_replay` | 1 回目 submit: 16:22:10（起動後 ~455 秒） |
| server `buffer_dir` | `total buffer size: 800` |
| trainer `drain()` | `files transferred to trainer: 800`（16:22:13） |
| 学習 step 進行 | `total steps: 0` → 400 → 800 → 1200 |
| **beta_sel 配線（layer3）** | client ログ全件 `beta_sel=0.0` |

初回ループ確立まで: trainer 起動 → 初回 submit まで **~455 秒**（生成待ち）。

### 途中の起動失敗（解消済み）

| 回 | 原因 | 対処 |
|---|---|---|
| 1 | port 5000 占有 | `fuser -k 5000/tcp` |
| 2 | warmstart ckpt の optimizer param 数不一致（chip_net 追加後） | optimizer/scheduler 除外、saved config `online=false` |
| 3 | warmstart から scaler も除外して KeyError | scaler を残して再生成 |

---

## Step 1: スループット（30 分計測窓）

計測窓: Step 0 成立後 **1803 秒**（monitor 出力）。  
全体 wall time（Step 0 待ち含む）: **2825 秒**。

### 生成（client）

| 指標 | 値 |
|---|---|
| `submit_replay` 回数（30 分窓） | 5 |
| 生成 game 数/分 | **133.1**（5 × 800 games / 30.05 min） |
| submit ファイル数/分 | **133.1**（1 game = 1 `.json.gz`） |
| 1 batch 生成時間 | **~435–480 秒** / 800 games |

### drain（trainer）

| 指標 | 値 |
|---|---|
| drain 回数（30 分窓） | 5 |
| 1 drain あたり file 数 | **800**（固定） |
| 1 drain あたり move 数（推定） | **~25,600**（200 step × batch 128） |
| drain 空待ち | **あり**（生成律速） |
| 初回 drain 待ち | **455 秒**（16:14:38 → 16:22:13） |
| 2 回目以降 drain 待ち（例） | **~306 秒**（16:25:55 → 16:31:01） |
| submit → drain 遅延（2 回目以降） | **4–5 秒** |

### 学習

| 指標 | 値 |
|---|---|
| steps（30 分窓） | 0 → 800（Δ **800**） |
| step/秒 | **0.444** |
| warmup 2000 step 換算 | **4508 秒**（~75.1 分） |
| 全 run 最終 step（停止後 ckpt） | **1200** |

### GPU（nvidia-smi、60 秒間隔サンプル）

| 指標 | 値 |
|---|---|
| 利用率 avg / peak | **30.1%** / **67%** |
| VRAM peak | **4746 MiB** / 8151 MiB |

---

## Step 2: 健全性（数値のみ）

| 指標 | step 400 | step 800 | step 1200 |
|---|---:|---:|---:|
| `dqn_loss` | 33.50 | 25.49 | 31.27 |
| `chip_loss` | 0.184 | 0.105 | 0.107 |
| `hparam/beta_sel` | 0.0 | 0.0 | 0.0 |

- 全 loss 値: **finite**（NaN なし）
- `beta_sel`: **0.0** 維持（1200 step < warmup 2000）
- 重みテンソル: **finite** 確認済み

---

## 成果物

| パス | 内容 |
|---|---|
| `runs/online_throughput/config.toml` | online 設定 |
| `runs/online_throughput/mortal_warmstart.pth` | warm-start ckpt（steps=0） |
| `runs/online_throughput/logs/server.log` | server ログ |
| `runs/online_throughput/logs/client.log` | client ログ |
| `runs/online_throughput/logs/trainer.log` | trainer ログ |
| `runs/online_throughput/report.json` | monitor 出力（JSON） |
| `runs/online_throughput/tb/` | TensorBoard イベント |
| `runs/run_server.py` | server spawn ランチャ |
| `runs/run_client.py` | client spawn ランチャ |

### コード変更（計測用）

- `mortal/client.py`: `get_param` 受取時に `beta_sel` をログ出力

---

## 関連

| ドキュメント | 内容 |
|---|---|
| `freeparlor/docs/online_replay_buffer.md` | 3 プロセス構成・drain フロー |
| `freeparlor/docs/online_r_chip_layer3.md` | Q_chip / β_sel 配線 |
| `freeparlor/docs/online_r_chip_layer2.md` | TD トランジション（dataloader） |
