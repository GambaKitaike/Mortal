# PPO P3 — Stage1 本走 (2026-07-04 再スタート)

**設計:** `ppo_migration_design.md` §5.1 / §7 P3  
**run dir:** `/home/gamba/mahjong/runs/ppo/stage1_<suffix>/`（再スタート後に更新）  
**tmux:** `ppo_p3_<suffix>`  
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

## 1. 開始報告

（再スタート起動後に記入）

### Config 全文

pending

### 検定 (10)(11) ログ

pending

### カウンタ初期値

| カウンタ | 初回 session 後 |
|---|---|
| `illegal_action_fallback_count` | pending |
| `[agari_guard]` stderr（訓練 client） | 期待 0 行 |

### 計装確認

| イベント | 期待 | 起動後 |
|---|---|---|
| `action_mass` | π(鳴き\|可能∧赤) / π(鳴き\|可能∧赤なし) + n | pending |
| `advantage_decomp` | P2c 同 | pending |
| `kyoku_reward_decomp` | P2c 同 | pending |
| `grp_calibration` | step 2000/4000/… | pending |

### Mem 1h 推移（5分間隔、`logs/mem_monitor.log`）

pending

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
