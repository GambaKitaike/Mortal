# PPO P1 配管 — 実装サマリ (2026-07-02)

**設計書:** `ppo_migration_design.md` §7 P1  
**ブランチ:** `ppo-migration`（`main` の DQN 経路は無変更）

---

## 実装差分サマリ

### 1. モデル (`mortal/model.py`)

| 追加 | 内容 |
|---|---|
| `PolicyHead` / `ValueHead` | DQN v2/v3/v4 と同 MLP 構成（v4: Linear、v2/v3: Linear→Mish→Linear） |
| `ActorCritic` | `policy_head(phi)/τ` + `value_head(phi)`；τ は config `[ppo].tau_init`（default 1.0） |
| `load_actor_critic_from_dqn_checkpoint` | `a_head` → PolicyHead、`v_head` → ValueHead 重みコピー |
| `load_ppo_from_mortal_checkpoint` | β=1 Huber 192×40 `mortal.pth` から ActorCritic 初期化 |
| `dqn_a_head_logits` | π₀ 一致検定用 |

| 撤去（ppo-migration のみ） | 内容 |
|---|---|
| `v_chip_head` / `a_chip_head` / `chip_net` | Phase5 遺産 |
| `ChipDQNTarget` | 全廃 |

**strict=False 適用範囲（コメント明記済み）:** レガシー checkpoint の `chip_net` / `v_chip_head` / `a_chip_head` は無視。`net` / `v_head` / `a_head`（main heads）は必須一致。

### 2. 学習 (`mortal/train_ppo.py` 新規、`train.py` 温存)

- **損失:** PPO clipped surrogate (ε=0.2) + c_vf·Huber(V, R̂_GAE, δ=15) − c_ent·H(π)
- **advantage:** per-batch 正規化；critic ターゲットは生スケール
- **GAE:** γ_disc=1.0, λ=0.95；エピソード=局、局末 terminal (V=0 bootstrap)
- **報酬:** `reward_design_teacherfree.md` §2 三項（α=β=γ=1, chip_value=5.0, opp なし）
- **CQL:** 未実装（意図的）
- **config:** `[ppo]` セクション（`mortal/config.py` defaults + `freeparlor/configs/ppo_p1.toml`）

### 3. PPO コアモジュール（新規）

| ファイル | 役割 |
|---|---|
| `mortal/ppo.py` | masked softmax/logp、GAE、PPO loss、報酬合成 |
| `mortal/ppo_transport.py` | trajectory `(obs, action, logp_old, mask, reward, done)` pack/unpack |
| `mortal/ppo_engine.py` | 行動時 `logp_old` 保存；訓練 client は `Categorical(masked_logits).sample()` の純 π サンプリング |
| `mortal/chip_from_log.py` | 自己対戦 json.gz から局単位 chip_delta を取得（online PPO 報酬用） |
| `mortal/ppo_dataloader.py` | 局末 sparse reward / done 付与、logp 再計算 |

### 4. 転送 (client → server → trainer)

- **client** (`mortal/client.py`): PPO 有効時 `PPOEngine` + `.traj` ペイロード生成
- **server** (`mortal/server.py`): `actor_critic_param` / `ppo_enabled` 対応
- **common** (`mortal/common.py`): `submit_param(..., use_ppo=True)` で ActorCritic 配信
- **player** (`mortal/player.py`): `train_play_ppo()` 追加

---

## §7.1 単体 sanity 実行ログ

```
PPO P1 sanity verification
checkpoint: /home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth
online log: /home/gamba/mahjong/runs_archive_0620/test_play/10299_8192_d.json.gz

(1) π₀ consistency
  PASS: softmax(policy/τ) ≡ softmax(a_head/τ)
(2) illegal action mask
  PASS: illegal actions have π=0
(3) GAE 3-step hand calc
  adv=[5.252500057220459, 3.950000047683716, 1.0]
  PASS: GAE matches manual 3-step
(4) logp_old client vs trainer
  PASS: client logp_old == trainer recompute
(5) reward composition vs calc_delta_blend
  PASS: compose helper unit cases
  PASS: calc_delta_blend ≡ compose (score / chip / mixed)
(6) legacy checkpoint load after chip head removal
  PASS: strict=False ignores chip_net; ActorCritic loads main heads
(7) pure π sampling
  TV distance=0.00437, illegal=0
  PASS: samples match softmax(masked logits), no illegal actions
(8) online chip from log
  player_id=0 n_kyoku=8 nonzero_kyoku=3 sum=-5.0
  PASS: log-derived chip_delta has non-zero entries

ALL 8 CHECKS PASSED
```

**実行コマンド:**

```bash
conda activate mortal
python freeparlor/scripts/verify_ppo_p1.py \
  --checkpoint /home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth \
  --grp-state /home/gamba/mahjong/runs/grp.pth \
  --online-log /home/gamba/mahjong/runs_archive_0620/test_play/10299_8192_d.json.gz
```

生ログ: `freeparlor/docs/ppo_p1_verify_log.txt`

---

## レビュー指摘と修正 (2026-07-02)

P1 レビューで **2 件のバグ** が指摘された。いずれも PPO オンライン経路のみ。DQN 経路・offline 学習は無変更。

### バグ 1: 行動選択が π と不一致（`ppo_engine.py`）

| | 内容 |
|---|---|
| **症状** | `boltzmann_epsilon` 混合 + `top_p` + `ε=0` 時 argmax により、PPO の on-policy 前提（行動 ∼ π）が破れ、`logp_old` と実際の行動分布が乖離 |
| **修正** | greedy/boltzmann/top_p を PPO 経路から全廃。訓練 client は常に `masked_logits = logits.masked_fill(~mask, -inf)` → `Categorical(logits=masked_logits).sample()`。`logp_old` は同一 masked logits から gather。評価用 `eval_mode=True` で argmax を残す |
| **config** | `[train_play.default]` から `boltzmann_epsilon` / `boltzmann_temp` / `top_p` を PPO config から削除（DQN 用 config は温存） |
| **検定 (7)** | N=10000 サンプルで TV 距離 < 0.02、違法行動 0 件 — **PASS** |

### バグ 2: online チップが npz 不在時にサイレント zeros（`client.py`）

| | 内容 |
|---|---|
| **症状** | `_finalize_ppo_trajectories` が `chip_dir/*.npz` を参照し、不在時 `np.zeros` で報酬の chip 項が消える |
| **修正** | 自己対戦 `json.gz` を直接パース。`chip_from_log.load_kyoku_chip_deltas_from_log` が arena `meta.chip_delta` または `preprocess_chips.hora_chip_deltas` リプレイで局単位 chip を取得。解決不能時は `RuntimeError` + error log（サイレント zeros 禁止）。PPO config から `chip_dir` 削除 |
| **検定 (8)** | チップ和了を含む自己対戦ログ 1 ファイルで非ゼロ chip_delta を assert — **PASS** |

---

## 結論

> **Q: P1 完了条件（§7.1 sanity 8 項目全 PASS + 配管実装 + レビュー修正）を満たしたか？**  
> **A: Yes**

**証跡:**
1. 上記実行ログ — 8/8 PASS（2026-07-02 実行）
2. `freeparlor/scripts/verify_ppo_p1.py` — assert ベース再現可能（検定 7・8 追加済み）
3. `ppo-migration` ブランチ — 純 π サンプリング、log 由来 chip、PolicyHead/ValueHead、GAE、PPO loss、trajectory 転送
4. `main` ブランチ — DQN 経路保全

**次フェーズ (P2):** 小予算 GPU スモーク（NaN/ratio/entropy/avg_rank 監視、c_ent/τ/lr 調整）
