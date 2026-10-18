# Online TD チップ報酬 — 層3: Q_chip ヘッド + target net + n-step TD + β_sel warmup

**日付:** 2026-06-25  
**スコープ:** `model.py`, `train.py`, collate 配線（`dataloader.py`）、online 行動選択配線

## 目的

チップ価値を Q_main から分離し、dueling 第2ヘッド Q_chip として TD 学習する。
行動選択は `Q_total = Q_main + β_sel·Q_chip`。Q_main（rank+素点+順位の MC+GRP）は不変更。

```
層1  arena ログ生成時に meta.chip_delta を付与          ✅
層2  dataloader / TD transition へ配線                  ✅
層3  Q_chip ヘッド + n-step TD + target + β_sel       ✅ 本タスク
```

## 前提（層1・層2 検証済み・未変更）

- 層2 entry 末尾4要素: `next_obs`, `next_mask`, `done_chip`, `r_chip`（非 oracle idx 6–9）
- oracle 時は idx=1 に `invisible_obs` 挿入、TD 4 要素は末尾のまま
- `r_chip` は局の trainee 最終 move に集約済み（局末のみ非0、多ロン合算済み）
- online は `player_names=['trainee']`、`cql_loss=0`

## 設計判断

| 項目 | 決定 |
|---|---|
| version | **4** — 現行 control.version に合わせ `net` + `chip_net` |
| Q_chip 構造 | Q_main と同形 dueling: `v_chip + a_chip - mean(a_chip over mask)` |
| phi | Brain 共有。Q_chip ヘッドのみ新規パラメータ |
| target net | chip ヘッドのみ Polyak。phi は online Brain を no_grad + freeze_bn で流用 |
| n-step | n=3、局内 truncate（`done_chip` / game 末）。train 側で game 配列から合成 |
| bootstrap | Double-DQN: `a'* = argmax Q_total`（online）、評価 `Q_chip_tgt(s', a'*)` |
| β_sel | 案A: step 固定スケジュール。行動選択のみ。chip_loss は非依存 |
| 後方互換 | 旧 state（chip ヘッド無し）→ chip 系新規 init で warm-start 継続 |

## 変更ファイル

| ファイル | 内容 |
|---|---|
| `mortal/model.py` | `chip_net`, `dueling_q`, `q_total`, `ChipDQNTarget` |
| `mortal/train.py` | n-step chip_loss, Polyak, β_sel, TB, state 保存/読込 |
| `mortal/dataloader.py` | game 単位バッファ, `compute_nstep_chip`, `collate_moves` |
| `mortal/engine.py` | 行動選択 `Q_total` |
| `mortal/common.py` | `submit_param` に `beta_sel` 追加 |
| `mortal/server.py` | `beta_sel` 保持・配布 |
| `mortal/client.py` | `get_param` で `beta_sel` 受取 → `TrainPlayer` |
| `mortal/player.py` | test/online に β_sel、赤保持局 metrics |
| `mortal/config.py` | env デフォルト追加 |
| `mortal/config.example.toml` | 同上 + `games_per_batch` |
| `freeparlor/scripts/verify_layer3_chip.py` | 検証ゲート 1–5（新規） |
| `freeparlor/configs/layer3_verify_64x10.toml` | 疎通検証用 config（新規） |

### model.py — Q_chip ヘッド（version=4）

```python
# Q_main: net(1024 → 1+ACTION_SPACE) → dueling_q
# Q_chip: chip_net(1024 → 1+ACTION_SPACE) → dueling_q（同 mask 処理）

def forward(self, phi, mask, *, return_q_chip=False):
    q = dueling_q(*self._main_heads(phi), mask)
    if not return_q_chip:
        return q                          # 後方互換: 既存呼び出しは Q_main のみ
    q_chip = dueling_q(*self._chip_heads(phi), mask)
    return q, q_chip

def q_total(q_main, q_chip, beta_sel):
    return q_main + beta_sel * q_chip
```

**パラメータ数（64×10 smoke 時）:**

| モジュール | params |
|---|---:|
| mortal (Brain) | 1,569,864 |
| dqn (Q_main + Q_chip) | 96,350 |
| chip_target | 48,175 |
| aux_net | 4,096 |

### target net — ChipDQNTarget

- `θ_chip_tgt ← τ·θ_chip + (1-τ)·θ_chip_tgt`、τ = `chip_target_tau`、毎 optimizer step
- 初期化: `θ_chip_tgt = θ_chip` のコピー
- target Q_chip 計算時: online Brain の phi を no_grad で入力（BN running stats 二重更新なし）

### train.py — n-step TD loss

```python
# n-step 集約（game 内 per-move r_chip から）
R = Σ_{k=0..n-1} γ^k · r_chip[i+k]     # done_chip で truncate

# bootstrap（非 done かつ n ステップ完走時）
target = R + γ^n · Q_chip_tgt(s_{i+n}, a'*)
a'* = argmax_a Q_total(s_{i+n}, a)     # online, mask 済み argmax

chip_loss = MSE(Q_chip(s,a), target.detach())
total_loss = dqn_loss + chip_loss * chip_weight + next_rank_loss * next_rank_weight
```

**触っていないもの:** Q_main MC+GRP 経路、`calc_delta_blend`、`reward_calculator`、層2 entry 先頭6列

### dataloader — collate 配線

層2 entry 構造は不変。n-step 合成のため buffer を **game 単位**（move entry の list）で保持し、
`collate_moves` が game バッチからランダム move を `batch_size` 件サンプルして n-step target を付与。

```python
def compute_nstep_chip(game, start_idx, n_step, gamma):
    # done_chip=1 または game 末で truncate → bootstrap なし
    # 否则 boot_obs/mask = game[start_idx + steps_used]
```

DataLoader: `batch_size=games_per_batch`（default 4）、`collate_fn=make_collate_fn(batch_size, n_step, gamma)`

### β_sel warmup（案A）

| steps | β_sel |
|---|---:|
| `< beta_sel_warmup_steps` | 0 |
| `warmup ≤ steps < warmup + ramp` | 0 → `beta_sel_max` 線形 |
| 以降 | `beta_sel_max` |

配布経路: `train.py` → `submit_param(beta_sel=...)` → `server` → `client` → `MortalEngine`

### config 追加（`[env]`）

| キー | デフォルト | 用途 |
|---|---:|---|
| `beta_sel_max` | 0.3 | 行動選択係数上限 |
| `beta_sel_warmup_steps` | 2000 | warmup 終了 step |
| `beta_sel_ramp_steps` | 2000 | ramp 長 |
| `chip_n_step` | 3 | n-step 長 |
| `chip_target_tau` | 0.005 | Polyak τ |
| `chip_weight` | 1.0 | chip_loss 重み（β_sel とは別） |

`[dataset].games_per_batch = 4`（collate 用）

### state_dict

```python
state = {
    'mortal': ...,
    'current_dqn': ...,      # chip_net 含む
    'chip_target': ...,      # chip ヘッドのみ
    'aux_net': ...,
    ...
}
```

旧 ckpt 読込: `dqn.load_state_dict(..., strict=False)` → chip ヘッド新規 init → `chip_target.copy_from(dqn)`

### TensorBoard 追加

| タグ | 内容 |
|---|---|
| `loss/chip_loss` | chip TD 損失 |
| `q_chip_predicted` | Q_chip 予測分布 |
| `hparam/beta_sel` | 現在の β_sel |
| `test_play/aka_held_call_win_rate` | 赤保持局の鳴き和了率 |
| `test_play/aka_held_chip_realize_rate` | 赤保持局チップ実現率 |

既存: `dqn_loss`, `next_rank_loss`, 放銃率/流局率/avg_rank 等（test_play Stat）

## 検証

**スクリプト:** `freeparlor/scripts/verify_layer3_chip.py`

```bash
cd mortal
MORTAL_CFG=../freeparlor/configs/layer3_verify_64x10.toml PYTHONPATH=. \
  /home/gamba/miniconda3/envs/mortal/bin/python \
  ../freeparlor/scripts/verify_layer3_chip.py \
  --config ../freeparlor/configs/layer3_verify_64x10.toml \
  --state-file /home/gamba/mahjong/runs/phase4d/phase4d_lo03/mortal.pth
```

### Gate 1: 回帰（chip_weight=0, beta_sel=0）

**条件:** offline 1 epoch、`chip_weight=0`, `beta_sel_max=0`

| 項目 | 結果 |
|---|---|
| 学習完走 | **PASS** |
| Q_main 経路 | dqn_loss / next_rank_loss 計算・backward 正常 |
| 判定 | **PASS** |

注: 旧コードとの bit 一致ではなく、chip 無効時に Q_main 経路のみで完走することを確認。
collate 変更によりバッチ構成は統計的に同一分布（厳密同一シード一致は非保証）。

### Gate 2: Q_chip forward + target Polyak

**条件:** version=4, τ=0.005, online chip 重み +1.0 後 Polyak 1 step

| 項目 | 結果 |
|---|---|
| target 重み変化 max Δ | **0.005000** |
| 判定 | **PASS** |

### Gate 3: n-step 集約 truncate

**条件:** 合成 game 5 move、局末 move に `r_chip=2`, `done_chip=1`

| start i | R | boot_done | 期待 |
|---:|---:|:---:|---|
| 0 | 0.0 | False | 3 step 完走 → bootstrap |
| 1 | 0.0 | False | 同上 |
| 2 | 2.0 | True | move 4 で done → truncate |
| 3 | 2.0 | True | 同上 |
| 4 | 2.0 | True | 局末 |

| 項目 | 結果 |
|---|---|
| 判定 | **PASS** |

### Gate 4: 64×10 疎通（chip_loss 非発散）

**条件:** 64×10 resnet, 1 tenhou ファイル, batch=32, 25+ steps, `chip_weight=1.0`

| 項目 | step 25 値 |
|---|---:|
| `loss/chip_loss` | 0.0031 |
| `loss/dqn_loss` | 2.55 |
| `hparam/beta_sel` | 0.0（warmup 中） |
| NaN / inf | なし |

| 項目 | 結果 |
|---|---|
| 判定 | **PASS**（性能判断は対象外） |

### Gate 5: warm-start（4d lo=0.3 ckpt）

**条件:** `/home/gamba/mahjong/runs/phase4d/phase4d_lo03/mortal.pth`（chip ヘッド無し）

| 項目 | 結果 |
|---|---|
| missing keys | `chip_net.weight`, `chip_net.bias` |
| forward 正常 | Q_chip / Q_chip_tgt とも finite |
| 判定 | **PASS** |

**総合結果: PASS**

## 実装時の修正メモ

1. **Polyak src/tgt 反転:** 初版で online↔target が逆だったため修正（Gate 2 で検出）
2. **bootstrap argmax:** mask 外 action を選ぶと Q=-inf → chip_loss=inf。`masked_fill(~mask, -inf).argmax` に修正
3. **game 単位 collate:** シャッフル済み単 move バッチでは n-step 合成不可のため、buffer を game list 化

## 関連

| ドキュメント / スクリプト | 内容 |
|---|---|
| `freeparlor/docs/online_r_chip_layer1.md` | 層1: arena hora chip_delta |
| `freeparlor/docs/online_r_chip_layer2.md` | 層2: TD transition entry |
| `freeparlor/scripts/verify_layer3_chip.py` | 層3 検証ゲート |
| `freeparlor/configs/layer3_verify_64x10.toml` | 疎通 config |
| `freeparlor/docs/phase4d_chip_realize.md` | 赤保持局 metrics 定義 |
