# PPO P3 — Stage1 本走 (2026-07-04 発進)

**設計:** `ppo_migration_design.md` §5.1 / §7 P3  
**run dir (1回目):** `ppo_p3_aborted_20260704_040100`（step 200 SIGTERM）  
**run dir (2回目):** `ppo_p3_aborted2_20260704_044030`（step ~3654、プール I/O スラッシング ~0.06 step/s）  
**run dir (3回目):** `ppo_p3_aborted3_perf_gate_20260704_214600`（step 251、旧 0.2 閾値ゲート中止 → 再判定で安定確認）  
**run dir (4回目):** `ppo_p3_aborted4_20260704_225400`（~step 50、trajectory skip 大量発生 → データ整合性例外で中止）  
**run dir (5回目):** `stage1_20260705_023140`（**本走 GO** — 2026-07-05 02:31 JST 発進）  
**run dir (5回目・初回):** `stage1_20260705_014852`（step 520 まで進行後、monitor grep pipefail で誤停止 → 修正再発進）  
**ブランチ:** `ppo-migration`  

---

## インシデント — trajectory skip 再発 (2026-07-04 23:54, 5回目)

**保全 dir:** `/home/gamba/mahjong/runs/ppo/ppo_p3_aborted4_20260704_225400/`

### 署名（pre-fix ログ解析）

| 項目 | 値 |
|---|---|
| len=0 型（キー迷子） | **0** — game_key 形式は一致 |
| 主パターン | `len(steps)=期待±k`（k∈{1,2,3}、±1 が最多） |
| pending_had_key | skip 時 **True**（キーは存在、step 数のみ不一致） |

### 分布（finalize 総数 vs skip — 損失率分母）

| run | submit 回数 (3 client 計) | game slots (×5) | skip (mismatch) | 備考 |
|---|---:|---:|---:|---|
| aborted3 ゲート 30min | 30 | 150 | **349** | ~77% が 1 session 当たり概算 |
| aborted4 本走 ~12min | 15 | 75 | **146** | 同上 |

**偏り:** seed/split/席に有意な偏りなし（split a–d / client0–2 ほぼ均等）。

### 根因

1. **kan_select 未記録** — `mortal.rs` が kan フェーズを `record=false`、かつ daiminkan で `need_kan_select` 漏れ。
2. **loader obs 数 ≠ runtime 記録数** — GameplayLoader の 4-event window と arena poll の決定境界が一致しないケースがあり、厳密 count 比較で全ゲーム skip。

### 修正 (2026-07-04 23:30–24:00)

| 項目 | 内容 |
|---|---|
| kan 記録 | kan_select を `record=true` + 独立 seq で記録 |
| daiminkan | `need_kan_select` に `can_daiminkan` を含める |
| at_kyoku | 記録時に `end_kyoku` 連動カウンタを step に保存 |
| finalize | **記録 step 数を正**とし loader は grp/chip のみ。count 不一致は INFO `loader size delta`（skip しない） |
| 拡張ログ | skip 時 game_key / expected / actual / pending / seed / split / 席 |
| 検定 (13) | 単独 client 52 半荘自己対戦 → join **100%** assert |
| libriichi.so | `cargo build --release -p libriichi` → `mortal/libriichi.so` 更新必須 |

### 検定 (13) 結果

```
games=52 joined=52 key_missing=0 mismatch=0 orphan=0
ALL 13 CHECKS PASSED
```

### 再発進条件

- 開始報告で `trajectory step count mismatch` = **0**、`loader size delta` は INFO のみ
- illegal_action_fallback / chip_err / NaN = 0

---

## インシデント — 汚染 run 中止 (2026-07-04)

**保全 dir:** `/home/gamba/mahjong/runs/ppo/ppo_p3_aborted_20260703_064427/`（旧 `stage1_20260703_064427`）  
**到達 step:** ~4000（trainer checkpoint 保存済み、本走判定には不適）

### 原因連鎖

1. **訓練 client に rule-based agari guard ON** — `client.py` で `PPOEngine(..., enable_rule_based_agari_guard=True)` を指定。設計 §4 の on-policy 前提（行動 ∼ π、logp_old 整合）に対し、Rust 側で和了拒否時に別行動へ上書き → **trajectory の (action, logp_old) が実際の方策分布と乖離**。
2. **相手 pool も guard ON** — `player.py` `_make_opponent_engine` 同様。trainee 以外の席でも行動上書き（pool は trajectory 非記録だが対局分布は歪む）。
3. **サイレント fallback** — `ppo_engine._pick_actions` の残余ループ（非法行動補正）が発動してもログなし。fp32/-1e9 化後は NaN 経路のみ想定だが、非ゼロ時の検知手段がなかった。
4. **構成 assert 欠如** — 起動時に guard/eval_mode/record_trajectory を検証する検定がなく、上記が本走 step ~60 まで気づかれず進行。
5. **game_key 形式不一致**（再スタート時発覚）— `game.rs` の `{:#x}` とログファイル名の decimal key が不一致 → trajectory orphan。`game_key` を `"{seed}_{key}_{split}"` に統一。
6. **プール I/O スラッシング**（2回目 run 性能劣化）— `PPOOpponentPoolEngine._ensure_weights` が `_react_batch` の ckpt グループ毎に `torch.load` → `load_state_dict` を実行。1 batch 内で最大 7 ckpt × 3 client が競合し、**実効 ~0.06 step/s**（17h で step 3654）。`_game_ckpt` も無制限成長。

### 修正 (2026-07-04)

| 項目 | 内容 |
|---|---|
| ガード範囲 | 訓練 client / pool: guard **OFF**（デフォルト）。eval (`test_play` 系): guard **ON** 維持 |
| 設計書 §4 | 「訓練 rollout への行動上書き（rule-based guard 含む）は禁止。eval は本家準拠で guard ON」追記 |
| カウンタ | `illegal_action_fallback_count` を client ログ出力（非ゼロで WARNING） |
| Rust | guard 発火時 `stderr` 1 行（eval 可視化） |
| 検定 (10) | trainee: guard=False / eval_mode=False / record_trajectory=True |
| 検定 (11) | pool engine: `pending_steps` なし（trajectory 混入防止） |
| プール常駐キャッシュ | `_models: dict[ckpt→(Brain,AC)]`、上限 `past_k+2=7`、超過時 pool 外 ckpt から evict |
| 検定 (12) | 同一 obs で old-load ≡ 常駐 cache logits（atol=1e-6、2 ckpt） |
| pool 行動選択 | `pick_actions_from_logits`（fp32/-1e9）に PPOEngine と共通化 |
| `_game_ckpt` | 上限 1000 超で現 batch step_meta 外キーを掃除 |

### 再スタート判断

- step 4000 時点の checkpoint は on-policy 汚染の疑いがあり **学習継続不可**（1回目 abort）。
- 2回目 run（`ppo_p3_aborted2_*`）は checkpoint 保存済み。常駐キャッシュ後 **~0.14 step/s** に改善（旧 0.2 閾値は撤回、下記ゲート参照）。
- 汚染 run は `ppo_p3_aborted_*` として削除せず保全（forensic 用）。
- init は従来通り `beta1_huber_192x40`、config は前回 P3 同一（lr=2e-5, τ=1.0, c_ent=0.01, 16k, save_every=2000）。

### 性能ゲート（再判定 — 2026-07-04、レビュー承認済み）

**旧判定:** 30min 全体 0.139 step/s < 0.2 → 中止（`aborted3`）。  
**方針変更:** **0.2 閾値はプール無しスモーク基準のため撤回**。残差（~0.14 vs 0.2）は trainer バッチ細分化（4 epoch × minibatch）+ 3 client GPU 競合の**構造コスト**と判断（レビュー承認済み）。  
**新判定:** 30min ログを前半 15min / 後半 15min に分割し、実効 step/s の差が **20% 以内 → 安定**。後半が明確に遅い（リーク疑い）場合のみ停止。

| 区間 | step 増分 | 実効 step/s |
|---|---:|---:|
| 前半 15min (21:54:25–22:09:25) | 135 (1→136) | **0.150** |
| 後半 15min (22:09:25–22:24:25) | 115 (136→251) | **0.128** |
| 全体 30min | 250 | **0.139** |
| 前半/後半差 | — | **14.8%** |

**判定:** **安定（PASS）** — 0.139 step/s を構造的定常値として受理、本走 GO。完走見込み **~32h**。

### 性能バックログ（Stage1 完走まで着手禁止）

| 項目 | 内容 |
|---|---|
| pool `.item()` ループ | `_react_batch` の per-element `.item()` をベクトル化 |
| pool checkpoint 形式 | `step_*.pth` を weights-only 保存にしてロード軽量化 |

### VRAM 見積り（常駐 7 組、実測 2026-07-04）

| 項目 | 値 |
|---|---|
| baseline | 2368 MiB |
| 7 pairs 常駐後 | **2992 MiB** (+624 MiB) |
| GPU total | 8151 MiB (RTX 5060) |
| 計測 | `freeparlor/scripts/measure_pool_vram.py --pairs 7` |

---

## 0. 着手前チェック（実装ギャップ）

| 項目 | 状態 | 備考 |
|---|---|---|
| (a) 相手プール §4 | **実装済** | `opponent_pool` + `PPOOpponentPoolEngine` |
| (b) action_mass 赤条件 | **実装済** | `pi_call_given_possible_aka_held` / `_no_aka` / `aka_over_no_aka` + n |
| (c) GRP calibration §8 | **実装済** | client pred/actual rank、trainer 2k step ごと `grp_calibration` |
| (d) ガード範囲・検定 (10)(11) | **本走前修正** | 2026-07-04 再スタート準備で適用 |
| (e) プール常駐キャッシュ・検定 (12) | **再々スタート前修正** | 2026-07-04 pool I/O スラッシング対策 |

---

## 1. 開始報告 (本走 — 2026-07-04 22:40 JST)

### 開始

| 項目 | 値 |
|---|---|
| 開始時刻 | **2026-07-04 22:40:37 JST**（step 1） |
| tmux | `ppo_p3_20260704_223200` |
| run dir | `/home/gamba/mahjong/runs/ppo/stage1_20260704_223200/` |
| 定常 step/s | **0.139**（ゲート再判定） |
| 完走見込み | **~32h**（ETA ~2026-07-06 06:40 JST） |

### 検定 (10)(11)(12)

```
ALL 12 CHECKS PASSED
(12) step_000000/000001 old-load ≡ resident cache (atol=1e-6)
```

### Mem 30min（ゲート run `aborted3` 参考）

| 時刻 | Mem used | available | GPU mem | GPU util |
|---|---:|---:|---:|---:|
| 21:56 | 1.4 GiB | 22 GiB | 1680 / 8151 MiB | — |
| 22:21 | 10 GiB | 12 GiB | 5808 / 8151 MiB | 78% |

### 監視期待値（凍結ルール — 4項目 + NaN）

| 項目 | 期待 | 非ゼロ時 |
|---|---|---|
| `trajectory step count mismatch` | **0** | **停止** |
| `illegal_action_fallback_count` | **0** | **停止** |
| `online chip resolution failed` | **0** | **停止** |
| `loader size delta` (INFO) | **0** | **報告のみ**（記録漏れ新種の早期シグナル） |
| trainer NaN | **0** | **停止** |

step 100 時点で上記4項目が全 **0** なら、判定窓まで凍結。

---

## 1d. 開始報告 (5回目 — 2026-07-05 02:00 JST)

### 開始

| 項目 | 値 |
|---|---|
| 開始時刻 | **2026-07-05 02:00:52 JST**（step 1、初回 run `014852`） |
| 本走 tmux | `ppo_p3_20260705_023140` |
| 本走 run dir | `/home/gamba/mahjong/runs/ppo/stage1_20260705_023140/` |
| 定常 step/s | **0.284**（30min 実測、初回 run） |
| 完走見込み | **~16h**（0.284 step/s × 16000 step） |

### 検定 (13)

```
games=52 joined=52 key_missing=0 mismatch=0 orphan=0 loader_delta=22
ALL 13 CHECKS PASSED
```

### step 100 監視4項目（初回 run `014852` @ step 104）

| 項目 | 値 | 判定 |
|---|---:|---|
| `trajectory step count mismatch` | **0** | OK |
| `illegal_action_fallback_count` | **0** | OK |
| `online chip resolution failed` | **0** | OK |
| `loader size delta` (INFO) | **64** | 報告（非ゼロ — **凍結不可**） |

必須3項目は全 0。loader_delta 非ゼロのため監視は継続。

### Mem 30min（初回 run、step 1 から 30min）

| 時刻 | Mem used | available | GPU mem | GPU util |
|---|---:|---:|---:|---:|
| 02:30 | 12 GiB | 10 GiB | 6549 / 8151 MiB | 57% |

### インシデント — monitor pipefail 誤停止

初回 run `014852` は step 520 まで正常進行後、`grep` 零件時 exit 1 × `pipefail` で monitor ループが誤停止。修正（`|| true`）後 `023140` で再発進。

---

## 1c. ゲート run 記録 (3回目 — aborted3, 2026-07-04 21:45 JST)

旧 0.2 閾値で中止。再判定により安定確認（上記性能ゲート節）。

---

## 1b. 開始報告 (2回目 — 2026-07-04 04:07 JST, aborted2)

### Config 全文

```toml
# /home/gamba/mahjong/runs/ppo/stage1_20260704_040100/config.toml
[control]
version = 4
online = true
save_every = 2000
test_every = 100000   # inline test_play 無効 (> max_steps)
submit_every = 50

[ppo]
enabled = true
lr = 2e-5
tau_init = 1.0
c_ent = 0.01
eps_clip = 0.2
ppo_epochs = 4
minibatch_size = 512
max_steps = 16000
init_checkpoint = '/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth'

[opponent_pool]
enabled = true
past_k = 5
latest_prob = 0.5
```

（その他 train_play client×3, env α=β=γ=1, resnet 192×40 — P2c 同型）

### 検定 (10)(11) ログ

```
(10) train client engine config (guard OFF / eval_mode False / record_trajectory)
  trainee dump: {'name': 'trainee', 'enable_rule_based_agari_guard': False, 'eval_mode': False, 'record_trajectory': True, 'has_pending_steps': True}
  PASS: trainee guard=False eval_mode=False record_trajectory=True
(11) opponent pool engine has no pending_steps
  pool dump: {'name': 'opp_pool', 'enable_rule_based_agari_guard': False, 'has_pending_steps': False}
  PASS: pool engine guard=False, no pending_steps / record_trajectory
ALL 11 CHECKS PASSED
```

client 起動時も同一 dump を出力（例 client0 step_meta 後）:
```
trainee engine config dump: guard=False eval_mode=False record_trajectory=True
opponent pool engine config dump: guard=False has_pending_steps=False
```

### カウンタ初期値

| カウンタ | 初回 session 後 (client0/1/2) |
|---|---|
| `illegal_action_fallback_count` | **0**（全 client、WARNING なし） |
| `trajectory orphan` / `game key missing` | **0** |
| `[agari_guard]` stderr（訓練 client） | 0 行（期待通り） |

### 計装確認

| イベント | 期待 | 起動後 (step ~6 確認) |
|---|---|---|
| `action_mass` | π(鳴き\|可能∧赤) / π(鳴き\|可能∧赤なし) + n | **OK** — trainer step 5–6 進行、diag 出力確認 |
| `advantage_decomp` | P2c 同 | **OK** |
| `kyoku_reward_decomp` | P2c 同 | **OK** |
| `grp_calibration` | step 2000/4000/… | pending |

### Mem 1h 推移（5分間隔、`logs/mem_monitor.log`）

| 時刻 | Mem used | available | Swap | GPU mem |
|---|---:|---:|---:|---:|
| 04:07:21 | 1.4 GiB | 22 GiB | 32 MiB | 1680 / 8151 MiB (8%) |

※ 1h 分は run 中に `mem_monitor.log` へ追記継続。

### 起動前残党チェック

- port 5000: clear
- GPU: idle 確認後起動
- libriichi.so: `PYO3_PYTHON=$CONDA_PREFIX/bin/python cargo build` で再ビルド済み（import OK）
- 中間 run `033843` は game_key 不一致で trajectory 全 orphan → `ppo_p3_aborted_20260704_033843` に保全

### 本走 GO 時点スナップショット (04:24 JST)

| 項目 | 値 |
|---|---|
| step | **112** / 16000 |
| orphan / fallback / chip_err / NaN | 全 **0** |
| Mem | 10 GiB used / 12 GiB avail |
| GPU | ~5.1 GiB / 8151 MiB (65%) |
| tmux | `ppo_p3_20260704_040100` alive |

---

## 2. 途中評価（未実施 — checkpoint 待ち）

| step | eval_sanity 100半荘 | 打牌統計 | action_mass |
|---:|---|---|---|
| 4000 | pending | pending | pending |
| 8000 | pending | pending | pending |
| 12000 | pending | pending | pending |
| 16000 | pending | pending | pending |

---

## 3. 判定（§5.1 事前固定）

判定窓 step 8000–16000、赤保持局面鳴き試行率:
- 初期方策比 2 倍未満 **かつ** 上昇トレンド無し → Stage2

**結果:** pending

---

## 4. 結果集計

pending（run 完走後に更新）
