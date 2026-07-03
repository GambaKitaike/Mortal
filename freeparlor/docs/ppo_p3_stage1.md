# PPO P3 — Stage1 本走 (2026-07-03)

**設計:** `ppo_migration_design.md` §5.1 / §7 P3  
**run dir:** `/home/gamba/mahjong/runs/ppo/stage1_20260703_064427/`  
**tmux:** `ppo_p3_20260703_064427`  
**ブランチ:** `ppo-migration`

---

## 0. 着手前チェック（実装ギャップ）

| 項目 | 状態 | 備考 |
|---|---|---|
| (a) 相手プール §4 | **本走前に実装** | 旧: `baseline_engine`(grp_baseline DQN) 固定。新: `opponent_pool` + `PPOOpponentPoolEngine` — 最新50% / 過去K=5一様50%、per-game sample |
| (b) action_mass 赤条件 | **本走前に拡張** | `pi_call_given_possible_aka_held` / `_no_aka` / `aka_over_no_aka` + n |
| (c) GRP calibration §8 | **本走前に追加** | client で pred/actual rank、trainer で 2k step ごと `grp_calibration` イベント |

---

## 1. 開始報告 (2026-07-03 06:44 JST)

### Config 全文

```toml
# /home/gamba/mahjong/runs/ppo/stage1_20260703_064427/config.toml
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

### 計装確認

| イベント | 期待 | 起動後 (step ~60 確認) |
|---|---|---|
| `action_mass` | π(鳴き\|可能∧赤) / π(鳴き\|可能∧赤なし) + n | **OK** — 例 step0: π_aka=0.229 (n=15), π_no_aka=0.130 (n=12), ratio=1.77 |
| `advantage_decomp` | P2c 同 | **OK** — 60 行/step 同期 |
| `kyoku_reward_decomp` | P2c 同 | **OK** |
| `grp_calibration` | step 2000/4000/… | pending（2k 到達後） |

### Mem 1h 推移（5分間隔、`logs/mem_monitor.log`）

| 時刻 | Mem used | available | Swap | GPU mem |
|---|---:|---:|---:|---:|
| 06:56:17 | 1.4 GiB | 22 GiB | 36 MiB | 1277 / 8151 MiB (1%) |

※ 1h 分は run 中に `mem_monitor.log` へ追記継続。

### 起動前残党チェック

- port 5000: clear（run スクリプト内 pkill + fuser）
- GPU: idle 確認後起動
- 初回起動で opponent pool fallback が DQN ckpt を読み `KeyError: actor_critic` → **修正** (`step_000000.pth` 生成 + DQN fallback loader)
- q_proxy 形状不一致 → **修正** (06:52 再起動)
- grp_calib `nonlocal` 漏れ → **修正** (06:56 再起動、step 60 まで正常進行)

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
