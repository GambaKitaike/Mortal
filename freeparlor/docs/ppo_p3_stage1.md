# PPO P3 — Stage1 本走 (2026-07-04 発進)

**設計:** `ppo_migration_design.md` §5.1 / §7 P3  
**run dir (1回目):** `ppo_p3_aborted_20260704_040100`（step 200 SIGTERM）  
**run dir (2回目):** `ppo_p3_aborted2_20260704_044030`（step ~3654、プール I/O スラッシング ~0.06 step/s）  
**run dir (3回目):** `ppo_p3_aborted3_perf_gate_20260704_214600`（step 251、旧 0.2 閾値ゲート中止 → 再判定で安定確認）  
**run dir (4回目):** `ppo_p3_aborted4_20260704_225400`（~step 50、trajectory skip 大量発生 → データ整合性例外で中止）  
**run dir (5回目):** `ppo_p3_aborted5_20260705_043200`（step ~1400、`sum(done)=1` 系統不一致 → 中止）  
**run dir (6回目):** `ppo_p3_aborted6_20260705_053301`（step 40、client 欠員で停滞 → daiminkan 経路修正後に中止）  
**run dir (7回目):** `stage1_20260705_072900`（**本走 GO** — daiminkan リバート + watchdog + verify 本番一致化後）  
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
| daiminkan | ~~`need_kan_select` に `can_daiminkan` を含める~~ → **2026-07-05 リバート**（下記 §daiminkan 根拠） |
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

### 改修バックログ（Stage1 完走後 — 判定窓中は凍結のため着手禁止）

| 項目 | 内容 |
|---|---|
| `.traj` に `at_kyoku` 永続化 | `ppo_transport.TrajectoryBatch` / `numpy_trajectory_to_batch` に `at_kyoku` を追加し、drain 後も runtime 局ラベルを直接読めるようにする（§1e 循環排除検証の前提） |

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

### 監視期待値（凍結ルール — 必須3項目 + NaN、loader_delta 除外）

| 項目 | 期待 | 非ゼロ時 |
|---|---|---|
| `trajectory step count mismatch` | **0** | **停止** |
| `illegal_action_fallback_count` | **0** | **停止** |
| `online chip resolution failed` | **0** | **停止** |
| `loader size delta` (INFO) | 任意 | **報告のみ**（局境界非関与のクラス未特定 delta — 下記 §1e） |
| trainer NaN | **0** | **停止** |

必須3項目 + NaN が全 **0** なら、判定窓まで**凍結**（2026-07-05 §1e 宣言）。

---

## 1e. loader_delta 特性（2026-07-05 03:15 JST 解析）

**対象 run:** `stage1_20260705_023140`（走行継続中、解析時 step **559**）  
**方法:** drain 直近 50 ゲームの `.traj` + train_play `json.gz`（セッション間 rmtree 対策で `/tmp/loader_delta_archive` に退避コピー）。GameplayLoader obs 列と traj `action` 列を NW 整列し、runtime 側 `at_kyoku` を loader ラベルで推定（`nice 19`・CPU のみ・シングルプロセス）。

### 1. 局境界一致（凍結必要条件）

| 検証 | 結果 |
|---|---|
| loader 局数 `len(grp)` vs runtime 推定 `max(at_kyoku)+1` | **50 / 50 一致** |

不一致 **0 件**。delta があっても局を跨ぐズレは検出されず。

### 2. delta 内訳（50 ゲーム）

| 項目 | 値 |
|---|---:|
| delta ≠ 0 のゲーム | **28 / 50**（56%） |
| 方向 runtime > loader | **22** |
| 方向 loader > runtime | **6** |
| \|delta\| = 1 | **22** |
| \|delta\| = 2 | **5** |
| \|delta\| = 3 | **1** |
| 整列ギャップ位置のイベント文脈（38 サイト） | **other** 100%（riichi / kan / hora / 流局 / riichi 直後に単一収束せず） |

run 全体（client ログ 282 件）でも \|delta\|=1 が **75%**（±1 合計 211/282）。**局境界に非関与のクラス未特定 delta**（±1 中心、方向混在、文脈 other 100%）。GameplayLoader 4-event window vs arena kan_select 記録差（§インシデント trajectory skip 修正と同型）の候補だが単一クラスに収束せず。**判定窓で異常挙動が出た場合の調査対象として保持**。整合性（kyoku 数・chip・join）とは独立。

### 3. at_kyoku 永続化診断（2026-07-05 03:45 JST 追記）

**結論: `.traj` に `at_kyoku` 非永続** — 循環排除版の基準1（記録済み `at_kyoku` 直接読み vs loader 局数）は **drain ペイロードのみでは実行不可**。

| 確認 | 結果 |
|---|---|
| drain `.traj` スキーマ (`ppo_trajectory_v2`) | `obs`, `action`, `logp_old`, `mask`, `reward`, `done`, reward 成分, grp rank — **`at_kyoku` なし** |
| `/tmp/loader_delta_archive` | `json.gz` のみ退避（`.traj` は drain 参照） |
| 上記 §1 局境界 50/50 | NW 整列で loader ラベルから runtime `at_kyoku` を**推定**（loader 依存あり） |
| `done` からの局数復元 | 不可 — packed `.traj` の `done` は局末マークだが `at_kyoku` 非保持のため独立検証にならない |

**drain 生ペイロード 1件（目視）:**

```
file: 53_10089_4723950512957204642_d.traj
game_key: 10089_4723950512957204642_d
format: ppo_trajectory_v2
persisted keys: action, done, format, grp_actual_rank, grp_pred_rank,
                logp_old, mask, obs, param_version, reward,
                reward_chip, reward_grp, reward_sotensu
at_kyoku in payload: False
steps: 176 / loader kyoku (grp): 9
```

**finalize 直前（buffer 経路・メモリ内）** — `ppo_engine.py` の pending step には `at_kyoku` が存在（`step_meta[3]` ← `mortal.rs` の `kyoku_counters`）。`client.py` は `at_kyoku = [int(s['at_kyoku']) for s in steps]` で参照後、`numpy_trajectory_to_batch` → `pack_trajectory` で **落ちる**。disk 上の drain `.traj` からは runtime 局ラベルを復元できない。

改修は **Stage1 完走後**（上記「改修バックログ」）。判定窓中はコード凍結のため触らない。

### 4. 凍結宣言

| 項目 | 解析時点 @ step 559 |
|---|---:|
| `trajectory step count mismatch` | **0** |
| `illegal_action_fallback_count` | **0** |
| `online chip resolution failed` | **0** |
| trainer NaN | **0** |
| `loader size delta` (INFO) | **282**（累計 — **監視除外**） |

**判定:** 必須3項目 + NaN 全 0、局境界 50/50 一致（NW 推定ベース、§3 参照）→ **loader_delta は局境界非関与のクラス未特定 delta として凍結条件から除外**。**本走は判定窓（step 8000–16000）まで凍結**（2026-07-05 03:15 JST 宣言、§3 追記 03:45 JST）。

### 5. 循環排除版（done×end_kyoku 全数）（2026-07-05 04:06 JST 解析）

**対象 run:** `stage1_20260705_023140`（走行継続中、解析時 step **1314**）  
**方法:** drain/buffer `.traj` を消費前に `/tmp/traj_archive` へ随時退避コピー（**1240 件**、run 継続中も追記）。対応 `json.gz` は `/tmp/loader_delta_archive`（**1000 件**、train_play rmtree 対策の既存退避 + 解析中追記）。GameplayLoader **非経由**で `json.gz` 行 JSON の `type=="end_kyoku"` を生カウント。(a) は退避 `.traj` の `sum(done)`（`ppo_transport.unpack_trajectory`・CPU・`nice 19`・シングルプロセス）。

#### スナップショット

| 項目 | 値 |
|---|---:|
| `/tmp/traj_archive` `.traj` | **1240** |
| `/tmp/loader_delta_archive` `.json.gz` | **1000** |
| ペア成立（同一 `game_key`） | **940 / 1240** |
| json 欠落（seed 10000–10024 × 3 session × 4 split） | **300** — train_play rmtree 前未退避 |

#### 全数突合（940 ペア）

| 検証 | 結果 |
|---|---:|
| `sum(done) == end_kyoku` | **0 / 940** |
| `sum(done) == end_kyoku − 1`（天和級候補） | **2** — 目視分類 **0 件**（下記） |
| その他不一致 | **938 / 940** |

**`sum(done)` 分布（940 ペア）:** `{1: 940}` — **全ペアで `sum(done)=1` 固定**。  
**`end_kyoku` 分布（940 ペア）:** min **2** / mean **10.28** / max **18**。  
典型不一致: `sum(done)=1` vs `end_kyoku≈10`（Δ = **−9** 中心）。

#### 天和級候補（2 件）目視

| game_key | sum(done) | end_kyoku | 分類 |
|---|---:|---:|---|
| `10047_9886577253662374531_c` | 1 | 2 | **非天和** — 2 局ログ、trainee（seat 2）全局で打牌あり |
| `10070_7649423605990334612_d` | 1 | 2 | **非天和** — 同上（seat 3） |

天和級（trainee 無判断局）: **0 件**（期待 ≈0 — 該当パターンは確認されず）。2 件は **`sum(done)=1` 固定の副作用**で `end_kyoku=2` のとき偶然 `end_kyoku−1` 式を満たしただけ。

#### 根因（938 件 + 上記 2 件）

packed `.traj` の `done` / `reward` / `reward_*` は **半荘末 1 step のみ非ゼロ**（`done=True` も 1 箇所）。`assign_rewards_and_dones` は `at_kyoku` 別の last step に局末 `done` を立てる設計だが、finalize 時の step `at_kyoku` が実質単一値（おそらく全 step `0`）→ **局数分の `done` が立っていない**。loader 非依存の循環排除検証として **基準1は未達**。

#### 判定（基準1）

| 項目 | 結果 |
|---|---|
| 循環排除版 基準1（`sum(done)` × raw `end_kyoku` 全件一致） | **未クローズ** — 940 ペア中一致 **0**、天和級 **0**、その他 **938** |
| 凍結見直し | **要** — 上記系統的不一致（`sum(done)=1` vs 平均 ~10 局） |
| 改修バックログ `at_kyoku` 永続化 | **残す** — 診断利便性向上（凍結解除後着手） |

---

## インシデント — at_kyoku 全ゼロ / sum(done)=1 (2026-07-05 04:32, 5回目中止)

**保全 dir:** `/home/gamba/mahjong/runs/ppo/ppo_p3_aborted5_20260705_043200/`  
**証拠:** `/tmp/traj_archive`（1240 `.traj`）、`/tmp/loader_delta_archive`（1000 `.json.gz`）— 削除せず保持

### 1-(a) `game.rs:119` 周辺 — `Index` の意味

```89:120:libriichi/src/arena/game.rs
// Poll::End 時:
for idx in &self.indexes {
    agents[idx.agent_idx].end_kyoku(idx.player_id_idx)?;
}
```

| フィールド | 意味 | trainee 側の値域 |
|---|---|---|
| `agent_idx` | `agents[]` 内のエージェント番号（0=challenger/trainee, 1=champion/pool） | **常に 0** |
| `player_id_idx` | エージェント内バッチスロット（`MortalBatchAgent` の `game_keys[]` / `kyoku_counters[]` インデックス） | `0 .. seed_count×4−1`（1 client 20 半荘なら 0–19） |

`set_scene` / `get_reaction` / `start_game` / `end_kyoku` は **いずれも同一 `player_id_idx`** を渡す設計（座席番号 0–3 ではない）。

### 1-(b) mortal.rs index 空間の突合

| 経路 | index 引数 | 空間 |
|---|---|---|
| `start_game(index, game_key)` | `idx.player_id_idx` | バッチスロット |
| `set_scene(index, …)` → stamp `at_kyoku` | `idx.player_id_idx` | 同上 |
| `end_kyoku(index)` | `idx.player_id_idx` | 同上 |

**結論:** index 空間は設計上一致。**仮説「index 空間不一致」は棄却。**

### 1-(c) 最小再現（eprintln 実測、修正前）

| 観測 | 旧 `.so`（Jul 4 23:28 ビルド） | 新 `.so`（c08271f 反映後） |
|---|---|---|
| `pending` の `at_kyoku` unique | **`[0]` のみ**（4 半荘） | `[0..max_kyoku]`（例: `_a` → 0–3、`_c` → 0–12） |
| `end_kyoku` eprintln | **0 行**（`BatchAgent` デフォルト no-op .so） | `index=0 counter 0→1→…→4` 等、増分確認 |
| stamp eprintln（index=0, game `_a`） | 常に `at_kyoku=0` | 初局後 `at_kyoku=1` 以降に遷移 |

**根因:** `libriichi.so` が **c08271f（`MortalBatchAgent::end_kyoku` + `kyoku_counters`）より前**にビルドされており、run 発進時 pre-flight が import のみで **カウンタ実装の有無を検証していなかった**。`end_kyoku` は `BatchAgent` trait デフォルト（no-op）のまま → `kyoku_counters` 不変 → 全 step `at_kyoku=0` → `assign_rewards_and_dones` が半荘末 1 `done` のみ。

### 修正方針（最小）

| 選択 | 内容 | 理由 |
|---|---|---|
| **採用** | `libriichi.so` 再ビルド・デプロイ + 検定 (14) を発進前必須化 | ソースは既に正しい；arena/mortal の index 変更不要 |
| 不採用 | arena 側 index マッピング変更 | 1-(b) で空間一致を確認、変更不要 |
| 追加 | `ppo_transport` に `at_kyoku` 永続化 | 診断・(14) が packed `.traj` から直接検証可能に |

### 検定 (14) — 報酬配置 end-to-end

単独 client 20 半荘（`seed_count=5`）自己対戦 + finalize。各 `.traj` × `json.gz` で assert:
- `sum(done) == end_kyoku` イベント数
- `at_kyoku` 連番、各局セグメントに `done` ちょうど 1
- done step の `reward` / `reward_*` がログ独立計算と一致（atol=1e-5）、非 done は 0

**(14) は join・kan 記録・at_kyoku・報酬合成を一括検出する発進前最上位検定。**

---

## daiminkan / kan_select 根拠 (2026-07-05, run #6→#7)

### 問題

run #6 step 40 で client 全滅。調査の過程で `need_kan_select` への `can_daiminkan` 追加（c08271f）が
daiminkan-only 局面で **kan_select フェーズ + action 42 の二段記録**を生み、loader との差分・経路複雑化の温床になっていた。

### リバート方針

| 項目 | 内容 |
|---|---|
| `need_kan_select` | `can_ankan \|\| can_kakan` のみ（c08271f 以前に復帰）。daiminkan-only は kan_select に入らない |
| daiminkan 実行 | action 42 が `kan_select_idx=None` で直接 `Event::Daiminkan`（本家経路、`mortal.rs` L549–570） |
| kan_select の daiminkan | GameplayLoader 側は選択肢1の**空判断**用。実判断（主反応 step）は runtime 記録済み |
| loader カウント | runtime 正典化により厳密一致は不要。差分は INFO `loader size delta` のみ |
| ankan/kakan 記録 | c08271f 本体（`record=true` + 独立 seq）は**維持** |

### 検定・監視の追加 (run #7)

| 項目 | 内容 |
|---|---|
| 検定 (13)(14) | `build_production_trainee_engine()` 経由（`enable_quick_eval=False` 等本番同一）、dump diff==空 assert |
| 検定 (15) | seed=(1,0xBEEF) + action-42 stub で daiminkan 直接経路を**必ず**踏む |
| watchdog | サブシェル `set +e` 隔離、client/trainer 各 3 回/時超過で run 停止 + `monitor.log` |
| 凍結条件 | **step 100 到達 + 必須3項目 0 + alive clients 3/3** |

---

## 1f. 開始報告 (7回目 — 2026-07-05 07:40 JST)

### 中止 (run #6)

| 項目 | 値 |
|---|---|
| 保全 dir | `/home/gamba/mahjong/runs/ppo/ppo_p3_aborted6_20260705_053301/` |
| 到達 step | **40**（client 欠員停滞、実害なし） |

### 開始 (run #7)

| 項目 | 値 |
|---|---|
| 開始時刻 | **2026-07-05 07:40:47 JST**（trainer init） |
| tmux | `ppo_p3_20260705_072900` |
| run dir | `/home/gamba/mahjong/runs/ppo/stage1_20260705_072900/` |
| 実 checkpoint dir | `/home/gamba/mahjong/runs/ppo/stage1_20260705_053301/`（config `state_file` 参照先） |

### pause / resume（2026-07-05）

| 項目 | 値 |
|---|---|
| 計画停止 | **step 10000** @ **2026-07-05 18:49:19 JST** |
| 再開 | **未実施**（preflight + 検定 (16) 通過後、ユーザー確認待ち） |
| 詳細 | `freeparlor/docs/ppo_p3_pause_resume.md` |

判定窓 8000–16000 の集計は **run7a (8000–10000) + run7b (10000–16000)** を global step で連結。step 10000 に運用上の継ぎ目あり（on-policy 連続性のみ一時リセット、optimizer・プール・config は継続）。

再開コマンド（preflight のみ）:

```bash
bash freeparlor/scripts/run_ppo_p3_resume.sh
```

### 検定 (15) — pre-flight ログ

```
(13) games=52 joined=52 key_missing=0 mismatch=0 orphan=0 loader_delta=27
(14) games=20 passed=20 end_kyoku/done/reward all OK
(15) daiminkan game=1_48879_b action42_present=True
ALL 15 CHECKS PASSED
```

### 凍結条件（改訂）

**step 100 到達 + 必須3項目 0 + alive clients 3/3** の3点セット。

| 項目 | 閾値 |
|---|---|
| `trajectory step count mismatch` | 0（停止） |
| `illegal_action_fallback_count` | 0（停止） |
| `online chip resolution failed` | 0（停止） |
| `alive clients` | **3/3**（欠員で停止） |
| `loader size delta` | INFO のみ（非致命） |

---

### 開始

| 項目 | 値 |
|---|---|
| 開始時刻 | **2026-07-05 06:29:31 JST**（step 1） |
| tmux / foreground | `ppo_p3_20260705_053301`（2回目 foreground 再起動で本発進） |
| run dir | `/home/gamba/mahjong/runs/ppo/stage1_20260705_053301/` |
| 到達 step（報告時点） | **40** / 16000 |
| commit | `efd8f2e`（at_kyoku 修正 + 検定 14 + pre-flight rebuild） |

### 検定 (14) — pre-flight ログ

```
(13) games=52 joined=52 key_missing=0 mismatch=0 orphan=0 loader_delta=40
(14) games=20 passed=20 end_kyoku/done/reward all OK
ALL 14 CHECKS PASSED
```

### step 40 監視4項目

| 項目 | 値 | 判定 |
|---|---:|---|
| `trajectory step count mismatch` | **0** | OK |
| `illegal_action_fallback_count` | **0** | OK |
| `online chip resolution failed` | **0** | OK |
| `loader size delta` (INFO) | **37** | 報告（非致命・凍結除外） |

**凍結:** step 100 到達後に再確認予定。必須3項目 + (14) PASS で本走継続中。

### インシデント — client 停止で step 40 停滞（2026-07-05 06:34 JST 以降）

| 項目 | 値 |
|---|---|
| 最終 trainer step | **40**（06:34:45 以降ログ無更新） |
| `run_client` | **不在**（server/trainer のみ生存） |
| 監視ポール | 60回×30s で step 40 固定（step 100 未到達） |
| 必須3項目 @ step 40 | mismatch=0, fallback=0, chip=0 |

**対応:** inner.sh の client watchdog 再起動を確認。foreground セッション生存中。

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

必須3項目は全 0。loader_delta 非ゼロのため当時は凍結不可だったが、§1e 解析（50/50 局境界一致）により **loader_delta 除外で凍結**（2026-07-05）。

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
