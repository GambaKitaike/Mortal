# Phase 4 Result — Chip Reward β (2026-06-22)

## チップ規定

| 項目 | 枚数 |
|---|---|
| 赤ドラ | 1枚/枚 |
| 一発 | 1枚 |
| 裏ドラ | 1枚/枚 |
| 役満 | 5枚 |
| ツモ | 合計×3（全員から回収） |
| ロン | 合計×1（放銃者のみ） |

換算: `1チップ = 5.0`（千点単位）

報酬式:

```
reward = alpha * (素点/1000) + beta * (chip_deltas * chip_value) + gamma_pt * 順位点
```

Phase4 初期値: `beta=1.0`, `chip_value=5.0`

## Rust 改修概要

- `libriichi/src/state/agent_helper.rs`: `agari_detail(is_ron, ura_indicators)` 追加（`agari_points` 温存）
- `libriichi/src/state/agari_detail.rs`: pyclass `AgariDetail`（point, fu, han, yakuman, ippatsu, num_aka, num_ura, is_tsumo）
- `libriichi/src/state/getter.rs`: Python 公開 `PlayerState.agari_detail(ura_markers: list[str])`
- ビルド: `PYO3_PYTHON=/home/gamba/miniconda3/envs/mortal/bin/python cargo build -p libriichi --lib --release`

## Step 3 検算結果（代表5局）

| case | ファイル | 目視 | agari_detail | 一致 |
|---|---|---|---|---|
| ura_ron (3m→4m) | 2009022020...28418029.mjson | ura 3m→手牌4m×1 | num_ura=1 | OK |
| yakuman | 2009032223...a062b5bb.mjson | deltas +48000 | yakuman=1 | OK |
| aka | 2009022011...d7935c6d.mjson | 赤5p等 | num_aka=2 | OK |
| ippatsu | 同上 | ippatsu | ippatsu=True | OK |
| ura markers only | 2009022011... (2p,7s) | 裏当たり0枚 | num_ura=0 | OK |

一括チェック: 200ファイル・1821 hora 中、edge case 15件で `cannot agari`（二重和了等）。本番前処理では skip 処理。

## Step 4 前処理

| 項目 | 値 |
|---|---|
| 入力 | `/home/gamba/mahjong/data/tenhou/2009/*.mjson` (6897 files) |
| 出力 | `/home/gamba/mahjong/data/tenhou/chips/<name>.npz` |
| hora 件数 | **62,496**（期待 ~62,932、agari_detail 失敗分を除く） |
| aka 出現率 | 42.7% |
| ura 出現率 | 12.5% |
| ippatsu 出現率 | 7.3% |
| yakuman 出現率 | 0.1% |
| chip base 分布 | 0:29431, 1:22736, 2:7891, 3:1816, … |

## Step 5 β 配線

- `reward_calculator.py`: `calc_delta_blend(..., chip_deltas, beta, chip_value)`
- `dataloader.py`: `chip_dir` から npz 読込、`load_chip_deltas()`
- `[env]`: `beta`, `chip_value`, `chip_dir`

β=0 vs β=1 サンプル（player 0）:

```
beta=0: [14.77, 13.78, 12.91, 12.71, -257.27]
beta=1: [14.77, 23.78,  7.91, 27.71, -257.27]
diff  = chip_deltas * 5.0  ✓
```

## Step 6 疎通学習 (64×10, β=1.0)

| 項目 | 値 |
|---|---|
| config | `freeparlor/configs/phase4_chip_beta1_64x10.toml` |
| GPU | cuda:0 (RTX 5060) |
| steps | 400 (~19,600 samples) |
| NaN/発散 | なし |
| checkpoint | `runs/phase4/beta1_64x10/mortal.pth` |

## 192×40 本番学習 (β=1.0, 1 offline epoch)

| 項目 | 値 |
|---|---|
| config | `freeparlor/configs/phase4_chip_beta1_192x40.toml` |
| アーキ | 192×40 (10,787,456 params) |
| 学習量 | 35,200 steps (1 epoch, batch=128) |
| GPU | cuda:0 (RTX 5060) |
| NaN/発散 | なし |
| dqn_loss @35200 | 40.10 |
| cql_loss @35200 | 1.59 |
| checkpoint | `runs/phase4/beta1_192x40/model.pth` |

## 自己対戦サニティ (同一条件, seed_key=42, 400局)

champion = challenger 自身（`champion.pth` にモデルコピー）。

| モデル | avg_rank | 備考 |
|---|---:|---|
| β=0 (Phase3 balanced) | 2.490 | 10 iter すべて ≈2.5 |
| β=1 (192×40) | 2.500 | 10 iter すべて 2.500 |

評価 log: `runs/phase4/eval/beta0/1v3`, `runs/phase4/eval/beta1/1v3`

## β=0 vs β=1 打牌統計 (192×40 対照)

**比較条件（交絡排除）:**

- 両方 192×40・1 offline epoch 相当の学習量
- **β=0**: Phase3 `balanced`（alpha=1, gamma_pt=1, チップ無し）— 今回同一 seed/400局で再評価
- **β=1**: Phase4 192×40 chip reward（alpha=1, beta=1, gamma_pt=1）
- 差分は **チップ有無のみ** が設計上の唯一の違い

| 指標 | β=0 (balanced) | β=1 (192×40) | 差分(β1−β0) |
|---|---:|---:|---:|
| 和了率 | 21.38% | 9.39% | −11.99pp |
| 放銃率 | 13.98% | 5.73% | −8.24pp |
| 立直率 | 26.23% | 20.88% | −5.36pp |
| 副露率 | 16.97% | 0.51% | **−16.46pp** |
| 平均和了打点 | 6827 | 10033 | **+3206** |
| 流局率 | 15.45% | 62.63% | +47.19pp |
| avg_rank (サニティ) | 2.490 | 2.500 | +0.010 |

**所見:**

- **副露率**: Phase3 素点重視(score_heavy)でも副露は増えなかった（人間データ天井）。β=1 では副露が **16.97% → 0.51%** と激減し、チップ圧で「副露が戻る」どころか門前・高打点待ちへ大きくシフト。
- **平均和了打点**: 6827 → 10033（+3206）。和了時の打点は大幅上昇（赤・裏・高打点報酬の方向性は出ている）が、和了率自体が半減し流局率 62% と極端な受け入れ方。
- **avg_rank**: 自己対戦で両方 ≈2.5 — 評価設定は正常。

## Artifacts

```
runs/phase4/beta1_192x40/
  mortal.pth
  model.pth
  best.pth
  train.log
  tb/
runs/phase4/eval/
  beta0/1v3/   beta0/champion.pth
  beta1/1v3/   beta1/champion.pth
runs/phase4/beta1_64x10/
  mortal.pth
  champion.pth
  train.log
  tb/
data/tenhou/chips/*.npz
freeparlor/scripts/preprocess_chips.py
freeparlor/scripts/verify_agari_detail.py
```

Config templates: `freeparlor/configs/phase4_chip_*.toml`
