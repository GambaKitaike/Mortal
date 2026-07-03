# PPO P3 — Stage1 本走 (2026-07-04 再スタート)

**設計:** `ppo_migration_design.md` §5.1 / §7 P3  
**run dir:** `/home/gamba/mahjong/runs/ppo/stage1_20260704_040100/`  
**tmux:** `ppo_p3_20260704_040100`  
**ブランチ:** `ppo-migration`

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

### 修正 (2026-07-04)

| 項目 | 内容 |
|---|---|
| ガード範囲 | 訓練 client / pool: guard **OFF**（デフォルト）。eval (`test_play` 系): guard **ON** 維持 |
| 設計書 §4 | 「訓練 rollout への行動上書き（rule-based guard 含む）は禁止。eval は本家準拠で guard ON」追記 |
| カウンタ | `illegal_action_fallback_count` を client ログ出力（非ゼロで WARNING） |
| Rust | guard 発火時 `stderr` 1 行（eval 可視化） |
| 検定 (10) | trainee: guard=False / eval_mode=False / record_trajectory=True |
| 検定 (11) | pool engine: `pending_steps` なし（trajectory 混入防止） |

### 再スタート判断

- step 4000 時点の checkpoint は on-policy 汚染の疑いがあり **学習継続不可**。
- 汚染 run は `ppo_p3_aborted_*` として削除せず保全（forensic 用）。
- init は従来通り `beta1_huber_192x40`、config は前回 P3 同一（lr=2e-5, τ=1.0, c_ent=0.01, 16k, save_every=2000）。

---

## 0. 着手前チェック（実装ギャップ）

| 項目 | 状態 | 備考 |
|---|---|---|
| (a) 相手プール §4 | **実装済** | `opponent_pool` + `PPOOpponentPoolEngine` |
| (b) action_mass 赤条件 | **実装済** | `pi_call_given_possible_aka_held` / `_no_aka` / `aka_over_no_aka` + n |
| (c) GRP calibration §8 | **実装済** | client pred/actual rank、trainer 2k step ごと `grp_calibration` |
| (d) ガード範囲・検定 (10)(11) | **本走前修正** | 2026-07-04 再スタート準備で適用 |

---

## 1. 開始報告 (2026-07-04 04:07 JST)

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
