# Online 生成律速緩和 — client 並列化再計測

**日付:** 2026-06-25  
**環境:** WSL2 (mahjong), conda `mortal`, RTX 5060 単機（8151 MiB）  
**目的:** client 並列化で生成律速が緩和するか計測（性能評価はしない）

## 前提

前回 throughput test（1 client）と同一 config・warm-start（`mortal_warmstart.pth` steps=0）。  
server / trainer は 1 プロセス、client のみ N 並列（`TRAIN_PLAY_PROFILE=client0..N`、log_dir 分離）。

---

## client 並列数

| 並列数 | OOM | 備考 |
|---:|---|---|
| 2 | **なし** | VRAM peak **5520 MiB** |
| 3 | **なし** | VRAM peak **6558 MiB**（8 GB 枠内、余裕 ~1.6 GB） |

---

## 差分表（15 分計測窓、Step 0 成立後）

| 指標 | 前回 1 client（30 分窓） | 2 client | 3 client |
|---|---:|---:|---:|
| client 並列数 | 1 | 2 | 3 |
| 生成 game 数/分 | **133.1** | **106.4** | **159.5** |
| submit ファイル/分 | **133.1** | **106.4** | **159.5** |
| drain 回数 | 5 | 2 | 4 |
| drain 空待ち（初回） | 455 秒 | 1104 秒 | 111 秒 |
| drain 空待ち（2 回目以降・平均） | **~306 秒** | **~0.7 秒** | **~0.8 秒** |
| drain 空待ち（2 回目以降・最大） | ~455 秒 | 0.7 秒 | 1.0 秒 |
| step/秒 | **0.444** | **0.443** | **0.665** |
| warmup 2000 step 換算 | **75.1 分** | **75.2 分** | **50.1 分** |
| GPU 利用率 avg / peak | **30.1%** / **67%** | **38.1%** / **81%** | **33.3%** / **83%** |
| VRAM peak | **4746 MiB** | **5520 MiB** | **6558 MiB** |

### drain 空待ちの内訳（trainer ログ、param submitted → file list size）

**2 client:** `[1104.4, 0.7]` 秒（2 回目以降: `[0.7]`）

**3 client:** `[110.8, 1.0, 0.5, 1.0]` 秒（2 回目以降: `[1.0, 0.5, 1.0]`、平均 **0.8 秒**）

---

## Step 2: 健全性

| 指標 | 2 client | 3 client |
|---|---|---|
| `dqn_loss`（最終 TB） | 30.68 | 33.77 |
| `chip_loss`（最終 TB） | 0.194 | 0.180 |
| `beta_sel` | 0.0 | 0.0 |
| OOM | 0 件 | 0 件 |

全 loss: **finite**。`beta_sel`: **0.0** 維持。

---

## 計測条件メモ

- 計測窓: Step 0 成立後 **902 秒**（~15 分）。monitor: `monitor_parallel.py`。
- **2 client:** 15 分窓内の submit は **2 回**（計 1600 games）。Step 0 待ち ~30 分（GPU 共有で 1 client あたり ~24 分/batch）。
- **3 client:** 15 分窓の **最初 1 分に 3 submit 集中**（2400 games）後、14 分間 submit なし。Step 0 時点で steps=400 まで進行済み。
- 3 client 時: drain 1 回あたり最大 **5600 files**（capacity 1600 超過分はバッファに蓄積、overflow なし）。

---

## 成果物

| パス | 内容 |
|---|---|
| `runs/online_throughput/run_test_parallel.sh` | 並列 client 起動 + 15 分計測 |
| `runs/online_throughput/monitor_parallel.py` | 多 client ログ集計 + drain 待ち解析 |
| `runs/online_throughput/report_parallel2.json` | 2 client レポート |
| `runs/online_throughput/report_parallel3.json` | 3 client レポート |
| `runs/online_throughput/logs_parallel2/` | 2 client ログ |
| `runs/online_throughput/logs_parallel3/` | 3 client ログ |
| `runs/online_throughput/config.toml` | `[train_play.client0/1/2]` 追加 |

---

## 関連

| ドキュメント | 内容 |
|---|---|
| `freeparlor/docs/online_throughput_test.md` | 前回 1 client 30 分計測 |
| `freeparlor/docs/online_replay_buffer.md` | 3 プロセス構成 |
