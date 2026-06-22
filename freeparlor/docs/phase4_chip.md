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

## β=0 vs β=1 打牌統計

**比較条件:**

- **β=0 参照**: Phase3 `balanced`（192×40, alpha=1, gamma_pt=1, チップ無し）— `phase3_sweep.md`
- **β=1**: Phase4 64×10 @400steps（アーキテクチャ差あり・自己対戦再評価要）

| 指標 | β=0 (Phase3 balanced) | β=1 (Phase4 64×10) | 差分 |
|---|---:|---:|---:|
| 和了率 | 21.38% | (要 self-play 再評価) | — |
| 放銃率 | 13.98% | — | — |
| 立直率 | 26.23% | — | — |
| 副露率 | 16.97% | — | — |
| 平均和了打点 | 6827 | — | — |
| 流局率 | 15.45% | — | — |
| avg_rank | 2.490 | — | — |

> 初回 1v3 は champion=`runs/champion.pth`（baseline）のため avg_rank≈3.3 となり無効。config を `phase4/beta1_64x10/champion.pth`（mortal コピー）に修正済み。自己対戦 400 局の再実行後に上表を更新すること。

## 192×40 本番

未実行（64×10 疎通 OK 後に `phase4_chip_beta1_192x40.toml` で実施予定）。

## Artifacts

```
runs/phase4/beta1_64x10/
  mortal.pth
  champion.pth
  train.log
  tb/
  1v3/          # 初回評価（要再実行）
data/tenhou/chips/*.npz
freeparlor/scripts/preprocess_chips.py
freeparlor/scripts/verify_agari_detail.py
```

Config templates: `freeparlor/configs/phase4_chip_*.toml`
