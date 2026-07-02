# PPO P2 診断再走 — OOM 対策後 (2026-07-02)

**ブランチ:** `ppo-migration`  
**run dir:** `/home/gamba/mahjong/runs/ppo/smoke_p2/`  
**設計書:** `ppo_migration_design.md` §7 P2 / §8 監視

---

## 0. インフラ（OOM 対策）

| 項目 | 値 |
|---|---|
| `.wslconfig` memory | 24GB（据え置き） |
| `.wslconfig` swap | **16GB**（8GB→16GB 増量。32GB 禁止・ホスト32GB） |
| 再起動後 `nvidia-smi` | **OK** — RTX 5060, Driver 596.49, CUDA 13.2 |
| 前回 OOM (21:04) | trainer ~16GB + client×3 ~6GB → 24GB 上限超過 + init.scope oom-kill |

**運用ルール（`next_steps_2.md` §7 追記）:**
- GPU ワークロード常に1系統（学習と eval 同時禁止）
- 診断 config: `test_every > max_steps` で inline test_play 無効
- 学習・eval とも tmux 起動（`run_ppo_p2_smoke.sh` / `run_eval_ppo_smoke_sanity.sh`）

---

## 1. 診断【1】control eval — mortal_init.pth（最優先）

**実行:** 2026-07-02 21:24–21:45、tmux `ppo_eval_init`、他プロセスゼロ確認後。  
**checkpoint:** `mortal_init.pth`（β=1 Huber 192×40、steps=0）  
**seed:** `[10000, 10100)` — step400 eval と同一範囲

| 指標 | init (今回) | step400 (§3 前回) |
|---:|---:|---:|
| avg_rank | 2.5000 | 2.5000 |
| fuuro | **17.29%** | **0.00%** |
| agari | 20.55% | 14.84% |
| houjuu | 12.18% | 7.87% |
| riichi | 23.65% | 14.91% |
| json.gz | 400 件 | 400 件 |

### 判定

> **init で fuuro ≈17% → 学習が鳴きを殺した。** eval 経路バグではない。

step400 checkpoint 単独 eval の fuuro=0% は、init 比 −17pp の学習効果（悪化）。
PPO 400 step が副露方策を潰している。eval 分岐の mask/argmax 調査は不要（【1】クローズ）。

---

## 2. 診断【3】計装再走 — client×3 / 400 step

> **交絡注記:** 本再走は key-join 修正込みのため、mismatch=0 および lag/clip は初回 P2 と厳密比較不能。

**実行:** 2026-07-02 21:52–22:21、tmux `ppo_p2_smoke`、inline test_play **無効**  
（`test_every=100000 > max_steps=400` + `train_ppo.py` 終了時 skip）  
**OOM:** **なし**（Mem peak ~11Gi / swap 未使用）。client×2 への降格は不要。

| 指標 | 初回 smoke (§2) | 今回再走 |
|---|---:|---:|
| mismatch | 13 | **0** |
| chip errors | 0 | 0 |
| NaN | 0 | 0 |
| clip (step399 epoch4) | — | 0.287 |
| clip (epoch1 mean 全step) | 56.5% | 53.3%（step0=0 含む） |
| tb clip_fraction @400 | — | 0.327 |
| 学習時間 | ~26 min | ~29 min |

### param_version lag（`ppo_diag.jsonl` batch_lag, n=400）

| lag | 件数 |
|---:|---:|
| 0 | 50 |
| 1 | 190 |
| 2 | 150 |
| 3 | 10 |

mean=1.30, median=1.0, max=3 — client×3 並列で許容域。

### epoch clip 推移（step399）

| epoch | clip | ratio mean±std | lag |
|---:|---:|---|---:|
| 1 | 0.290 | 1.018±0.206 | 1 |
| 4 | 0.287 | 1.016±0.215 | 1 |

epoch1 から既に ~29%（初回の「epoch 重ねで暴れる」より staleness/lag 寄り）。

### advantage 正規化（コード現状・修正なし）

`train_ppo.py` `train_on_trajectories`: GAE 後 `(adv - mean) / (std + 1e-8)` を
**minibatch 単位**（`minibatch_size=512` で trajectory テンソルを chunk）に適用。

### 現行 `[ppo]` config（`ppo_p2_smoke.toml`）

```
enabled=true eps_clip=0.2 c_vf=0.5 c_ent=0.01 gae_lambda=0.95 gamma_disc=1.0
tau_init=1.0 huber_delta=15.0 lr=3e-4 ppo_epochs=4 minibatch_size=512 max_steps=400
init_checkpoint=phase4/beta1_huber_192x40/mortal.pth
control.test_every=100000  # inline test_play 無効
```

---

## 3. 診断【2】mismatch — 初回 P2 内訳

**旧ログ:** 初回 P2 (2026-07-02 15:29–15:55) の `client*.log` は【3】再走 (21:52) で**上書き消滅**。
件数 **13** は `ppo_p2_smoke.md` §2 / commit `1a64262` 時点の `grep` カウンタのみ残存。

**再解析 (forensic):** 同一 config + `client.py@1a64262` (cursor-based pending) で 400 step 再実行
(2026-07-02 23:33–00:27)。件数 **7**（確率的ブレ; 機構は同一）。

| client | 件数 (forensic) | タイミング (trainer step 近似) |
|---|---:|---|
| client0 | 2 | step ~57 付近 (23:38, 23:42) |
| client1 | 3 | step ~19–57 + 1件 post-run (00:26) |
| client2 | 2 | step ~19–66 (23:39, 23:43) |

**内訳パターン:** 全件 `trajectory step count mismatch, skipping game`（旧 client は step 数不一致のみ記録）。
client×3 並列で pending cursor と `sorted(file_list)` の game step 数がずれるタイミングで散発。
根治: key-join (`game_id`/`seq`) — 【3】再走で mismatch=0 を確認済み。

---

## 4. 再現コマンド

```bash
# 【1】control eval init のみ
EVAL_LABEL=init EVAL_CHECKPOINT=/home/gamba/mahjong/runs/ppo/smoke_p2/mortal_init.pth \
  bash freeparlor/scripts/run_eval_ppo_smoke_sanity.sh

# 【3】計装スモーク（client×3 既定）
bash freeparlor/scripts/run_ppo_p2_smoke.sh

# 計装集計
conda run -n mortal python freeparlor/scripts/summarize_ppo_diag.py
conda run -n mortal python freeparlor/scripts/collect_ppo_p2_metrics.py
```
