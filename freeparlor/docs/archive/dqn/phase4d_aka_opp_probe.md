# Phase 4d — 赤取りこぼし損失プローブ (2026-06-23)

## 目的

「赤を保持したまま非和了で局を終えた」ことに per-kyoku の小さなマイナス報酬を与え、赤保持時の押し/鳴きを間接的に促す **プローブ** を実装する。

- 最終解ではなく、offline で副露の符号が反転するかを測るための実験用項
- per-move 報酬チャネルは新設せず、既存 per-kyoku スカラーに 1 項だけ加算
- `train.py` の MC 配分 (`q_target_mc = gamma ** steps_to_done * kyoku_rewards`) は変更しない

## 制約と担保

| 制約 | 対応 |
|---|---|
| `train.py` 不変更 | MC ターゲットのまま |
| `lambda_opp=0.0` で現状再現 | opp 項は `lambda_opp > 0` のときのみ加算。検証済み |
| 既存 chip 項を維持 | `beta * chip_deltas * chip_value` の上に加算 |
| per-kyoku 粒度 | 局終了時スナップショットから 4 配列を前処理 |

## 報酬式

既存式に opp 項を追加:

```
reward = α·(素点/1000) + β·(chip_deltas × chip_value) + γ·順位点 + opp
```

局 k・プレイヤ p の opp 項:

```
fire = 1  if (won==0 and dealt_in==0 and aka_held>0) else 0
w    = 1.0 if tenpai_end else noten_factor

opp[k,p] = -β · lambda_opp · chip_value · aka_held[k,p] · w[k,p] · fire[k,p]
```

config パラメータ（`[env]`、既定値 0 で無効）:

| キー | 既定 | 意味 |
|---|---|---|
| `lambda_opp` | `0.0` | 取りこぼし係数。0 で完全無効 |
| `noten_factor` | `0.0` | ノーテン時の重み（テンパイ = 1.0） |

## PlayerState アクセサ

**Rust 追加不要。** 既存公開 API を使用:

| 用途 | API | 実装 |
|---|---|---|
| 手牌内赤枚数 | `state.akas_in_hand` | `count_hand_aka()` = `sum(akas_in_hand)` |
| テンパイ判定 | `state.shanten` | `is_tenpai_at_end()` = `shanten == 0` |

定義場所: `libriichi/src/state/getter.rs` (`#[pymethods]`)

`count_hand_aka()` は単一関数に分離済み（将来「引いた赤」定義へ差し替え可能）。

## 前処理 (Step 4 拡張)

`freeparlor/scripts/preprocess_chips.py` を拡張。局終了イベント（`hora` / `ryukyoku`）検出時、**`states[p].update()` 呼び出し前**にスナップショット。

| npz キー | dtype | 意味 |
|---|---|---|
| `chips` | int16 (n,4) | 既存 chip delta（和了局のみ非零） |
| `aka_held` | int16 (n,4) | 局終了時手牌内の赤 (0m/0p/0s) 枚数 |
| `tenpai_end` | int8 (n,4) | 局終了時テンパイなら 1 |
| `won` | int8 (n,4) | その局の和了者なら 1 |
| `dealt_in` | int8 (n,4) | ロン和了の放銃者 (target) なら 1 |

- 全 kyoku を `0..max_kyoku` で `np.zeros` 初期化（流局局も含む）
- `won` / `dealt_in` は `hora` イベントから: `won[actor]=1`, ロン時 `dealt_in[target]=1`

### 実行結果 (6897 files 全再実行)

| 項目 | 値 |
|---|---|
| 入力 | `/home/gamba/mahjong/data/tenhou/2009/*.mjson` (6897 files) |
| 出力 | `/home/gamba/mahjong/data/tenhou/chips/<name>.npz` |
| hora 件数 | **62,496** |
| chip aka 出現率 (和了時) | 42.7% |
| chip ura 出現率 | 12.5% |
| ippatsu 出現率 | 7.3% |
| yakuman 出現率 | 0.1% |

### サニティ統計 (player×kyoku セル 296,912 件)

| 指標 | 値 | 備考 |
|---|---|---|
| `aka_held > 0` 率 | **37.57%** | 局終了時に手牌に赤を持つ割合 |
| `fire` 率 (式定義) | **24.39%** | 非和了・非放銃・赤保持 |
| `fire` 率 (テンパイ付き) | **6.81%** | 上記 + テンパイ。`noten_factor=0` 時に opp が効く母集団 |

`aka_held>0` 37.6% は和了時 chip-aka 42.7% より低く、流局・非和了局を含むため整合的。

## コード変更一覧

| ファイル | 変更内容 |
|---|---|
| `mortal/config.py` | `env` に `lambda_opp`, `noten_factor` デフォルト注入 |
| `mortal/config.example.toml` | 同上をドキュメント化 |
| `freeparlor/scripts/preprocess_chips.py` | 4 配列の記録・npz 保存・サニティ集計 |
| `mortal/reward_calculator.py` | `calc_delta_blend` に opp 項 |
| `mortal/dataloader.py` | `load_kyoku_probe_arrays` + config 読込 + 引数渡し |
| `freeparlor/scripts/verify_lambda_opp_zero.py` | `lambda_opp=0` 一致検証（新規） |

**未変更:** `train.py`, `libriichi` (Rust), `libriichi.so`

### 配線

`dataloader.py` → `calc_delta_blend`（呼び出し 1 箇所）:

```python
probe = self.load_kyoku_probe_arrays(file_path, player_id, len(grp_feature))
kyoku_rewards = self.reward_calc.calc_delta_blend(
    ...,
    chip_deltas=chip_deltas, beta=self.beta, chip_value=self.chip_value,
    aka_held=probe['aka_held'], tenpai_end=probe['tenpai_end'],
    won=probe['won'], dealt_in=probe['dealt_in'],
    lambda_opp=self.lambda_opp, noten_factor=self.noten_factor,
)
```

後方互換: npz に新キーが無い場合は `load_kyoku_probe_arrays` がゼロ配列を返す。

## 検証

### 1. `lambda_opp=0.0` 一致

`verify_lambda_opp_zero.py` で 1 ファイル検証:

```
OK: 2009020103gm-00a9-0000-2453a04c.mjson player=0 n_kyoku=34 max_diff=0
```

`lambda_opp=0` かつ opp 配列を渡しても、chip 項までの既存式と要素ごとに一致。

### 2. npz 再生成

6897 ファイル全件再実行済み。サンプル npz キー:

```
['chips', 'aka_held', 'tenpai_end', 'won', 'dealt_in']
```

## 次のステップ（案）

1. config 作成: 例 `lambda_opp=0.1`, `noten_factor=0.0`, 既存 `beta=0.3` 等と組み合わせ
2. 64×10 疎通学習 → 192×40 本番 1 epoch
3. Phase 4c と同様の aka-conditional 分析で副露率・押し率の符号変化を確認
4. 効果が見えなければ `lambda_opp` スイープ or `noten_factor` 調整

検証コマンド:

```bash
# 一致確認
MORTAL_CFG=configs/phase4_chip_beta1_192x40.toml \
  python freeparlor/scripts/verify_lambda_opp_zero.py

# 前処理（データ更新時）
python freeparlor/scripts/preprocess_chips.py
```
