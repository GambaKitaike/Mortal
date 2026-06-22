# Free Parlor Reward — Mortal 報酬設計実験

個人開発・ポートフォリオ向けプロジェクト。オープンソース麻雀AI [Mortal](https://github.com/Equim-chan/Mortal) を土台に、**報酬関数を変えると打牌がどう変わるか**を検証した。強化学習は本プロジェクトで初めて触れた。

Mortal（天鳳・着順特化）の報酬を、フリー雀荘/競技ルール（素点 + ウマオカ + チップ）向けに設計し直し、重みパラメータを変えながらオフラインRL（CQL）で学習・評価した。

---

## 動機・背景

Mortal 本来の報酬は**純着順**（天鳳段位制に最適化）。一方、フリー雀荘や競技麻雀では **素点（レート）+ ウマオカ + チップ（祝儀）** で精算する。ルールが違えば最適打牌も違うため、報酬を作り替える必要があった。

本プロジェクトで設計した報酬式:

```
reward = α · (素点 / 1000) + β · (チップ枚数 × 5.0) + γ · 順位点
```

| 項 | 内容 |
|---|---|
| 素点 | 局ごとの点数増減（千点正規化） |
| チップ | 赤ドラ1枚/枚、一発1枚、裏ドラ1枚/枚、役満5枚。ツモは合計×3、ロンは合計×1 |
| 順位点 | ウマオカ `[+35, +5, −15, −25]`（千点・ゼロ和） |

`α = β = γ = 1` としたとき、実際の金銭精算額と一致する設計（[phase4_chip.md](docs/phase4_chip.md) に検算あり）。

---

## 技術スタック / 構成

| レイヤ | 内容 |
|---|---|
| エミュレータ | **libriichi**（Rust）— 対局進行・点数計算 |
| 学習 | **mortal**（Python + PyTorch）— ResNet + CQL オフラインRL |
| 着順予測 | **GRP**（Group Rank Predictor）— 教師データから学習 |
| 環境 | WSL2 / Ubuntu、RTX 5060（Blackwell sm_120, cu128 PyTorch）、conda、Rust + pyo3 |

本家 Mortal からの fork 運用。改修は `freeparlor/`（設計書・config・分析スクリプト）と、報酬まわりの libriichi / mortal コード。

---

## 実験フェーズ

### Phase 1 — ベースライン確立

素の Mortal（着順報酬）を天鳳2009データで再現。192×40 モデルで自己対戦400局の打牌統計を計測。

| 指標 | 値 | 鳳凰卓目安 |
|---|---:|---|
| 和了率 | 22.27% | ~22% |
| 放銃率 | 14.31% | ~12% |
| 副露率 | 30.39% | ~35% |

人間水準に近い打牌であることを確認し、以降の比較基準とした。詳細: [phase1_stats_192x40.md](docs/phase1_stats_192x40.md)

### Phase 2 — 素点 + ウマオカ報酬への差し替え

初のコード改修。報酬を純着順 → 素点 + ウマオカに変更。同一アーキテクチャ（192×40）で対照実験。

| 指標 | Phase 1 | Phase 2 | Δ |
|---|---:|---:|---:|
| 副露率 | 30.39% | 16.97% | **−13.4pp** |
| 平均和了打点 | 5505 | 6827 | **+1322** |
| 立直率 | 15.67% | 26.23% | +10.6pp |

**報酬を変えると打牌が変わる**ことを実証。詳細: [phase2_result.md](docs/phase2_result.md)

### Phase 3 — α:γ 比スイープ

素点重み α と順位点重み γ を4点でスイープ（192×40、各400局自己対戦）。

| 指標 | rank_only | rank_heavy | balanced | score_heavy |
|---|---:|---:|---:|---:|
| 副露率 | 30.39% | 6.84% | 16.97% | 10.41% |
| 平均和了打点 | 5505 | 7264 | 6827 | 7301 |
| 流局率 | 11.26% | 25.13% | 15.45% | 22.61% |

- **平均打点**は素点重視方向（rank_only → score_heavy）で単調増加（5505 → 7301, +33%）。
- **副露率**は素点重視でも増えない（score_heavy 10.41% < rank_only 30.39%）。オフライン学習は教師データ（人間・着順志向）の分布に制約される**天井**があることを発見。

詳細: [phase3_sweep.md](docs/phase3_sweep.md)

### Phase 4 — チップ（祝儀）報酬の導入

libriichi（Rust）を改修し、pyo3 経由で和了内訳（赤/裏/一発/役満）を Python に公開。チップを正確に勘定（学習データの赤出現率 **42.7%**）。

**β = 1.0（金銭精算忠実）** で打牌が崩壊:

| 指標 | β = 0 | β = 1.0 |
|---|---:|---:|
| 和了率 | 21.38% | 9.39% |
| 副露率 | 16.97% | 0.51% |
| 流局率 | 15.45% | 62.63% |

**β スイープ**（0 / 0.1 / 0.3 / 0.5 / 1.0）で健全域は **β ≤ 0.3** と判明（流局15〜20%、和了20%前後、副露12〜17%）。

**赤ドラ条件別分析**（[phase4_aka_conditional.md](docs/phase4_aka_conditional.md)）: 人間の正着「赤持ち → 鳴いて上がってチップ確保」は**いずれの β でも副露率では確認できない**。代わりに「赤持ち → 立直率↑」が一貫。原因は**赤を抱えて流局した取りこぼし損失が報酬に無い**ことと特定。

詳細: [phase4_chip.md](docs/phase4_chip.md)

---

## 主要な発見

- **報酬設計がルールに対する打牌を定量的に決定づける** — Phase 2 の対照実験で実証（副露 −13pp、平均打点 +1322）。
- **報酬の絶対スケールが学習の安定性 / 打牌の健全性を左右する** — β 過大（1.0）で流局62%・副露0.5%に崩壊。loss は未発散でも打牌は実用域外。
- **オフライン学習には教師データ分布による天井がある** — 素点重視にしても副露率は人間データ水準を超えない（Phase 3）。
- **チップ報酬に「取りこぼし損失」が欠けると、人間の正着を再現できない** — 赤条件別分析で特定（未実装）。

---

## 本家 Mortal との差分

| ファイル | 内容 |
|---|---|
| `libriichi/src/state/agari_detail.rs` | 和了内訳 pyclass（`AgariDetail`） |
| `libriichi/src/state/agent_helper.rs` | `agari_detail()` 実装 |
| `libriichi/src/state/getter.rs` | Python 公開 `PlayerState.agari_detail()` |
| `mortal/reward_calculator.py` | `calc_delta_blend` — 素点 + チップ + 順位点 |
| `mortal/dataloader.py` | 報酬配線、チップ npz 読込 |
| `freeparlor/` | 設計書、各 phase 結果、config テンプレ、前処理/分析スクリプト |

```
$ git diff upstream/main --stat

 freeparlor/configs/.gitkeep                        |   0
 freeparlor/configs/phase1_pipeline.toml.template   | 157 +++++++++++++
 freeparlor/configs/phase1_production.toml.template |  69 ++++++
 freeparlor/configs/phase1_repro_192x40.toml        | 159 +++++++++++++
 freeparlor/configs/phase1_repro_64x10.toml         | 158 +++++++++++++
 freeparlor/configs/phase2_freeparlor.toml          | 160 +++++++++++++
 freeparlor/configs/phase3_sweep_balanced.toml      | 160 +++++++++++++
 freeparlor/configs/phase3_sweep_rank_heavy.toml    | 160 +++++++++++++
 freeparlor/configs/phase3_sweep_rank_only.toml     | 160 +++++++++++++
 freeparlor/configs/phase3_sweep_score_heavy.toml   | 160 +++++++++++++
 freeparlor/configs/phase4_chip_beta0_1_192x40.toml | 162 ++++++++++++++
 freeparlor/configs/phase4_chip_beta0_3_192x40.toml | 162 ++++++++++++++
 freeparlor/configs/phase4_chip_beta0_5_192x40.toml | 162 ++++++++++++++
 freeparlor/configs/phase4_chip_beta0_64x10.toml    | 162 ++++++++++++++
 freeparlor/configs/phase4_chip_beta1_192x40.toml   | 163 ++++++++++++++
 freeparlor/configs/phase4_chip_beta1_64x10.toml    | 163 ++++++++++++++
 freeparlor/configs/phase4_sweep_eval_beta0.toml    | 150 +++++++++++++
 freeparlor/configs/phase4_sweep_eval_beta1.toml    | 153 +++++++++++++
 freeparlor/docs/.gitkeep                           |   0
 freeparlor/docs/libriichi_agari_survey.md          | 249 +++++++++++++++++++++
 freeparlor/docs/phase1_result.md                   | 103 +++++++++
 freeparlor/docs/phase1_stats_192x40.md             | 122 ++++++++++
 freeparlor/docs/phase2_result.md                   | 143 ++++++++++++
 freeparlor/docs/phase3_sweep.md                    |  78 +++++++
 freeparlor/docs/phase4_aka_conditional.md          | 133 +++++++++++
 freeparlor/docs/phase4_chip.md                     | 238 ++++++++++++++++++++
 freeparlor/experiments/.gitkeep                    |   0
 freeparlor/scripts/analyze_aka_conditional.py      | 214 ++++++++++++++++++
 freeparlor/scripts/preprocess_chips.py             | 135 +++++++++++
 freeparlor/scripts/verify_agari_detail.py          | 141 ++++++++++++
 libriichi/src/state/agari_detail.rs                |  22 ++
 libriichi/src/state/agent_helper.rs                | 136 ++++++++++-
 libriichi/src/state/getter.rs                      |  12 +
 libriichi/src/state/mod.rs                         |   3 +
 mortal/dataloader.py                               |  34 ++-
 mortal/reward_calculator.py                        |  17 +-
 36 files changed, 4392 insertions(+), 8 deletions(-)
```

---

## 今後の課題

- チップ報酬に**取りこぼし損失**（赤保有で流局した場合の機会損失）を組み込み、「赤持ち → 確実に上がる」を再現する（**未実装**）。
- **オンライン自己対戦 RL** で人間データの天井を超える（**未実装**・要 GPU）。
- 学習データを2009年1年分から増やし、モデル強度を上げる。
- 天鳳データの利用規約上、本 AI は**非商用・ポートフォリオ用途に限る**（下記参照）。

---

## 注意・ライセンス

- 土台は [Mortal](https://github.com/Equim-chan/Mortal)（**AGPL-3.0**）。本リポジトリも AGPL に従う。
- 学習データは**天鳳鳳凰卓**由来。天鳳の利用規約により、競合製品・商用利用は不可。本プロジェクトは研究・ポートフォリオ目的の**非商用**利用に限定する。

---

## 関連ドキュメント

| ファイル | 内容 |
|---|---|
| [phase1_stats_192x40.md](docs/phase1_stats_192x40.md) | Phase 1 打牌統計 |
| [phase2_result.md](docs/phase2_result.md) | Phase 2 報酬差し替え結果 |
| [phase3_sweep.md](docs/phase3_sweep.md) | Phase 3 α:γ スイープ |
| [phase4_chip.md](docs/phase4_chip.md) | Phase 4 チップ報酬・β スイープ |
| [phase4_aka_conditional.md](docs/phase4_aka_conditional.md) | Phase 4 赤ドラ条件別分析 |
