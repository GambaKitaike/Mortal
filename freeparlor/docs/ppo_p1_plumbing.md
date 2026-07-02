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
| `mortal/ppo_engine.py` | 行動時 `logp_old` 保存 |
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

ALL 6 CHECKS PASSED
```

**実行コマンド:**

```bash
conda activate mortal
python freeparlor/scripts/verify_ppo_p1.py \
  --checkpoint /home/gamba/mahjong/runs/phase4/beta1_huber_192x40/mortal.pth \
  --grp-state /home/gamba/mahjong/runs/grp.pth
```

生ログ: `freeparlor/docs/ppo_p1_verify_log.txt`

---

## 結論

> **Q: P1 完了条件（§7.1 sanity 6 項目全 PASS + 配管実装）を満たしたか？**  
> **A: Yes**

**証跡:**
1. 上記実行ログ — 6/6 PASS（2026-07-02 実行）
2. `freeparlor/scripts/verify_ppo_p1.py` — assert ベース再現可能
3. `ppo-migration` ブランチ — PolicyHead/ValueHead、GAE、PPO loss、trajectory 転送、chip/CQL/opp 撤去
4. `main` ブランチ — 未変更（DQN 経路保全）

**次フェーズ (P2):** 小予算 GPU スモーク（NaN/ratio/entropy/avg_rank 監視、c_ent/τ/lr 調整）
