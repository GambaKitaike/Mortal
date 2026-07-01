# フリー雀荘特化 麻雀AI — 報酬設計の調査基盤

[Mortal](https://github.com/Equim-chan/Mortal)（天鳳＝着順特化の強化学習麻雀AI）を
ベースに、**チップ（祝儀）ありのフリー雀荘ルールで勝てる打牌**を学習させるには
報酬をどう設計すべきかを実験的に解明するプロジェクト。

> **位置づけ — これは検証用モデルであり、商用プロダクトではない。**
> 最終目標は商用フリー雀荘特化麻雀AIだが、それは本リポジトリの延長では作らない。
> 本プロジェクトの目的は「**どんな報酬設計が、チップあり麻雀でどんな打牌効果を生むか**」を、
> 実績ある既存アーキテクチャ（Mortal）の上で**安く・速く調査する**こと。
> ここで得た**知見**（報酬設計・学習の限界・online TDの挙動）を持って、商用版は
> 別途ゼロから実装する想定（後述のライセンス事情・データ事情による）。

---

## 何を調べているか

天鳳ルール（着順がすべて・チップ無し）に最適化された Mortal は、フリー雀荘で
価値を持つ**赤ドラ・一発・裏・役満（＝チップ）を取りに行く打牌**を学ばない。
特に「赤を持ったら鳴いてでも和了る」という、チップルールでの基本戦術が出ない。

報酬式を以下のように拡張し、各項が打牌をどう変えるかを段階的に検証する：

```
reward = α·(素点/1000) + β·(チップ枚数 × 5.0) + γ·順位点(ウマオカ)
       + opp(赤取りこぼし罰)  + [online] β_sel · Q_chip(per-move TD価値)
```

対象ルール: 4人打ち・喰いタン・赤×3・25000持ち30000返し・ウマ10-20・オカあり。

---

## 主要な発見（要約）

| 論点 | 結論 |
|---|---|
| 報酬設計で打牌は動くか | **動く**。素点/チップ/順位点の重みで副露率・和了打点が定量的に変化（Phase2-4） |
| offline で赤の鳴きを学べるか | **学べない（天井あり）**。CQLが「赤で鳴く」を教師データ外として抑制。鳴き和了率が人間8.98%に対しAI3%台で頭打ち（Phase3-4d） |
| online 自己対戦で天井を超えられるか | **一時的に超える、が安定しない**。per-move TD（Q_chip）で鳴き和了率6.7%・赤選択性+2.9pp（人間並み）を記録するが、継続学習で振動し平均3%台に収束（Phase5） |
| 安定化のカギ | CQL強度の単一スイープでは解けない。n_step（局末報酬の逆流距離）・自己対戦の非定常性が振動の主因と推定（Phase6で検証予定） |

**現在地**: 「online TDは offline天井に触れられるが、安定保持はまだできていない」。
失敗ではなく**未到達**——打牌は健全（avg_rank≈2.5、放銃率正常）なまま、目的の
behaviorが安定しない段階。

---

## Phase 一覧

| Phase | 内容 | 結論 |
|---|---|---|
| 0 | 環境構築（WSL2 / RTX5060 / Rust+PyO3） | — |
| 1 | 素 Mortal 再現（192×40, 天鳳2009） | 打牌が人間水準 |
| 2 | 素点+ウマオカ報酬 | 副露 −13.4pp、平均打点 +1322 |
| 3 | α:γ スイープ | **offline天井を発見**（素点重視でも副露増えず） |
| 4 | チップ β 導入（libriichi改修で祝儀を正確勘定） | 健全域 β≤0.3 |
| 4c | 赤条件別 人間 vs AI | 人間は赤持ちで副露+2.81pp、AIは逆 |
| 4d | 赤取りこぼし罰プローブ | offlineの限界を確定（鳴き和了率が天井） |
| 5 | online自己対戦 + per-move Q_chip TD | 天井に触れるが安定せず（上記） |
| 6 | （予定）n_step延長・相手プール混合で安定化 | — |

詳細は `freeparlor/docs/` の各 Phase ドキュメント、最新の全体状況は
`freeparlor/docs/next_steps.md` を参照。

---

## アーキテクチャ

- **ゲームエンジン**: libriichi（Rust）。フリー雀荘ルールのため改修
  （`agari_detail` で赤/裏/一発/役満を公開、arena の hora に `chip_delta` を埋め込み）。
- **学習**: PyTorch。192×40 ResNet エンコーダ + dueling DQN + GRP（順位予測）。
- **offline**: 天鳳2009棋譜から CQL（Conservative Q-Learning）。MC ターゲット。
- **online**: 自己対戦（server / trainer / client×3 の3プロセス）。
  チップ価値を **Q_chip** として分離し per-move TD で学習
  （`Q_total = Q_main + β_sel·Q_chip`）。Q_main の既存設計は不変更。

online チップTDの実装詳細は `freeparlor/docs/online_r_chip_layer{1,2,3}.md`。

---

## 再現方法

### 環境
```bash
# WSL2 / RTX5060 / CUDA / conda env "mortal"
conda activate mortal
```

### libriichi ビルド（Rust改修後）
```bash
cargo build -p libriichi --lib --release
cp target/release/deps/libriichi.so mortal/libriichi.so
```

### offline 学習
```bash
# runs/ の spawn ランチャ経由（WSL2 の CUDA fork 回避のため必須）
MORTAL_CFG=freeparlor/configs/<config>.toml python runs/run_train.py
```

### online 学習（3プロセス）
```bash
# server / trainer / client×3。起動前に stale プロセス停止と port5000 解放を確認
MORTAL_CFG=runs/online_main/config.toml python runs/run_server.py
MORTAL_CFG=runs/online_main/config.toml python runs/run_train.py
MORTAL_CFG=runs/online_main/config.toml python runs/run_client.py  # ×3
```

### 評価
打牌統計で比較する（自己対戦の avg_rank は常に≈2.5 のため）。
赤条件別の分析は `freeparlor/scripts/analyze_aka_conditional.py`、
チップ実現率は `analyze_chip_realize.py`。

---

## 主要パラメータ（現状）

| パラメータ | 値 | 備考 |
|---|---|---|
| α / γ | 1.0 / 1.0 | 素点・順位点の重み |
| β | ≤0.3 | チップ報酬。1.0で打牌崩壊 |
| lambda_opp | 0.3 | 赤取りこぼし罰（offline健全域） |
| enable_cql_online | true | falseだと無差別鳴きが暴走 |
| min_q_weight | 未確定 | 0.3が最良だが不安定。安定解は探索中 |
| chip_n_step | 3 | 延長検討中（振動の主因候補） |
| chip_target_tau | 0.005 | Q_chip target の Polyak |
| chip_weight | 1.0 | chip_loss の損失重み |
| beta_sel_max | 0.3 | warmup 2000 / ramp 2000 step |

---

## 今後（ロードマップ）

### 本リポジトリ内（調査の継続）
- **短期**: n_step延長・相手プール混合で online の振動を抑え、選択的な赤鳴みを
  安定保持できるか検証（Phase6）。
- 評価軸を打牌統計から**チップ込み収支**（期待チップを含む和了点）へ移行。

### 商用版（別実装・本リポジトリの外）
商用フリー雀荘AIは、本リポジトリの延長ではなく**ゼロから実装する**予定。理由は2つ：

1. **ライセンス**: ベースの Mortal は AGPL-3.0-or-later（下記）。ネットワーク提供にも
   ソース公開義務が及ぶため、これを土台にしたクローズドソースの商用製品は作れない。
   本プロジェクトは「AGPLコードで**知見のみ**を得る検証」と位置づけ、商用版は
   コードを継承せず知見だけを用いて独立実装する。
2. **データ事情**: チップありルールの牌譜データは事実上存在しない（天鳳2009は
   チップ無し）。よって商用版は既存棋譜の模倣ではなく、**純粋な強化学習
   （自己対戦中心）**になる見込み。本プロジェクトの online TD・報酬設計の知見が、
   その設計の出発点になる。

---

## ライセンス

本プロジェクトは [Mortal](https://github.com/Equim-chan/Mortal)
（Copyright © 2021-2022 Equim）の派生物であり、**GNU AGPL-3.0-or-later** を継承する。

- **コード**: AGPL-3.0-or-later。本リポジトリを配布・改変・**ネットワーク経由で
  サービス提供**する場合、AGPL の条件（ソースコード提供義務を含む）に従う必要がある。
- 原 Mortal の README は `MORTAL_UPSTREAM_README.md` に保存。
- ロゴ等アセットは原プロジェクトでは CC BY-SA 4.0。

> ⚠️ **商用利用の注意**: AGPL-3.0 はネットワーク利用にもソース公開義務が及ぶ
> （ネットワーク条項）。本コードベースを土台にしたクローズドソース商用製品は
> 原則として作れない。商用版を独立実装する方針なのはこのため（上記ロードマップ参照）。
> ライセンス解釈は最終的に専門家への確認を推奨。