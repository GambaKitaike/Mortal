# PPO P2 スモーク — 結果 (2026-07-02)

**設計書:** `ppo_migration_design.md` §7 P2  
**ブランチ:** `ppo-migration`  
**run dir:** `/home/gamba/mahjong/runs/ppo/smoke_p2/`

---

## 1. Run 構成

| 項目 | 値 |
|---|---|
| init_checkpoint | `runs/phase4/beta1_huber_192x40/mortal.pth` |
| アーキテクチャ | 192×40, version=4 |
| online 構成 | server×1 / trainer×1 / client×3 |
| 学習量 | 400 step（`ppo.max_steps=400`） |
| 起動 | `freeparlor/scripts/run_ppo_p2_smoke.sh` |
| eval | `freeparlor/scripts/run_eval_ppo_smoke_sanity.py`（100半荘自己対戦、単独プロセス） |

起動前チェック（学習・eval 共通）: `pkill` 残党確認 + `ss -tlnp \| grep 5000` ゼロ確認。

---

## 2. 学習監視値（trainer.log / tb）

400 step 全ログあり。NaN / FloatingPointError: **0 件**。chip 解決エラー: **0 件**。

| 指標 | step 1 | step 400 | 判定メモ |
|---|---:|---:|---|
| total loss | 28.80 | 13.96 | 発散なし |
| policy loss (π) | 0.045 | 0.003 | — |
| value loss (V) | 57.53 | 27.95 | — |
| entropy (H) | 1.015 | 1.616 | 単調崩落なし（上昇） |
| clip 域外比率 | 37.8% | 12.1% | 平均 56.5%、max 85.2%（§8 目安 ~30% 超過） |
| explained variance | −0.19 | 0.0008 | 終盤でゼロ付近 |
| trajectory mismatch | — | — | client 警告 **13 件**（期待 0） |

tb 最終値（step 400）: `loss/total=4.99`, `loss/entropy=1.77`, `ppo/clip_fraction=0.43`, `ppo/explained_variance=−0.004`。

学習時間: 15:29 起動 → 15:55 step 400 到達（約 26 分）。

---

## 3. eval_sanity 結果（100 半荘）

**実行:** 2026-07-02 18:19–18:39（`run_eval_ppo_smoke_sanity.sh`、ログ改修後）

| 指標 | 値 |
|---|---:|
| avg_rank | **2.5000** |
| 放銃率 (houjuu) | 7.87% |
| 和了率 (agari) | 14.84% |
| 副露率 (fuuro) | 0.00% |
| 立直率 (riichi) | 14.91% |
| test_play json.gz | **400 件** |

進捗ログ（10 局ごと）例: `対局 10/100 完了 (json.gz=40)` … `対局 100/100 完了 (json.gz=400)`。

---

## 4. ハング事後調査

### 4.1 eval_sanity の起動経路

**単独 Python プロセス**（server / client 不要）。

```
run_eval_ppo_smoke_sanity.sh
  ├─ pkill + ss -tlnp | grep 5000（起動前チェック）
  └─ PYTHONUNBUFFERED=1 conda run -n mortal python eval_ppo_smoke_sanity.py
       ├─ config 読込（MORTAL_CFG=.../smoke_p2/config.toml）
       ├─ checkpoint ロード（mortal.pth + actor_critic）
       └─ TestPlayer.test_play_ppo → libriichi OneVsThree.py_vs_py（arena 自己対戦）
```

旧来の直接起動（チェックなし）:

```bash
MORTAL_CFG=/home/gamba/mahjong/runs/ppo/smoke_p2/config.toml \
PYTHONPATH=/home/gamba/mahjong/Mortal/mortal \
conda run --no-capture-output -n mortal python \
  /home/gamba/mahjong/Mortal/freeparlor/scripts/eval_ppo_smoke_sanity.py
```

### 4.2 15:55 学習終了 → 16:58 eval 起動まで（シェル履歴）

`~/.bash_history` に **15:55–16:58 の該当コマンドは存在しない**（この区間の eval 起動は Cursor エージェント経由。対話シェル履歴ベースの実行記録なし）。

**ファイルタイムスタンプ（事実のみ）:**

| 時刻 | イベント |
|---|---|
| 15:55:26 | trainer.log: step 400 到達、trainer 内 test_play 開始（`seed [10000,10100)`） |
| 15:55:27 | trainer.log: `Terminated`（inline test_play 中断） |
| 16:17:59 | `logs/eval_sanity.log` 作成（Birth） |
| 16:58:28 | `logs/eval_sanity.log` 更新（0 byte のまま） |
| 16:58 | `test_play/` ディレクトリ mtime 更新（中身なし） |

**エージェント実行記録:** 16:58 前後に `eval_ppo_smoke_sanity.py` を Cursor Shell から複数回起動。WSL 再起動を挟み、eval ログは空のまま完走せず（json.gz 未生成）。

**推定原因（証跡付き）:** trainer 内 test_play と eval_sanity がいずれも OneVsThree GPU 自己対戦。15:55 の `Terminated` は run スクリプト cleanup または WSL 落ち。16:17–16:58 の eval は起動直後にプロセス/WSL 喪失し、**PYTHON 出力バッファ + ログ未整備のため無言に見えた**。改修後の再走（§3）は 19 分で完走。

### 4.3 ログ改修（本タスク）

- `PYTHONUNBUFFERED=1` + `FlushingFileHandler`（即時 flush）
- 段階ログ: config 読了 / checkpoint ロード / arena 起動 / 対局 N/100（10 局刻み）
- eval 起動前チェックを `run_eval_ppo_smoke_sanity.sh` に追加
- 環境メモ: `next_steps_2.md` §7、`README.md` インフラ節

---

## 5. 調整履歴

| 試行 | 変更 | 結果 |
|---|---|---|
| 1 | なし（初回 smoke） | 学習 400 step 完走。clip 平均高め、mismatch 13 |
| — | c_ent / lr 再試行 | **未実施**（深追いせず記録のみ） |

---

## 6. 結論

> **Q: P2 完了条件（NaN 無し・ratio 正常・エントロピー健全・avg_rank 崩壊無し・chip/mismatch エラー 0）を満たしたか？**

**No（条件付き部分合格）**

| 条件 | 結果 | 証跡 |
|---|---|---|
| NaN 無し | **Yes** | trainer.log, nan_errors=0 |
| ratio 正常（clip <~30%） | **No** | clip 平均 56.5%、max 85.2% |
| エントロピー健全 | **Yes** | H: 1.01 → 1.62（崩落なし） |
| avg_rank 崩壊無し | **Yes** | eval avg_rank=2.5000 |
| chip エラー 0 | **Yes** | client logs, chip_errors=0 |
| mismatch エラー 0 | **No** | 13 件（client WARNING） |

**総評:** PPO 配管（400 step 学習・checkpoint 保存・100 局 eval）は動作確認済み。clip 比率と trajectory mismatch は P3 前に要調査。eval ハングはログ/起動前チェック改修で再発防止。

---

## 7. 再現コマンド

```bash
# 学習（400 step）
bash /home/gamba/mahjong/Mortal/freeparlor/scripts/run_ppo_p2_smoke.sh

# eval（100 半荘 sanity）
bash /home/gamba/mahjong/Mortal/freeparlor/scripts/run_eval_ppo_smoke_sanity.sh

# 指標集計
conda run -n mortal python /home/gamba/mahjong/Mortal/freeparlor/scripts/collect_ppo_p2_metrics.py
```
