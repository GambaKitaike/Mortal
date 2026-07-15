# フリー雀荘特化 麻雀AI — 報酬設計の調査基盤

[Mortal](https://github.com/Equim-chan/Mortal)（天鳳＝着順特化の強化学習麻雀AI）を
ベースに、**チップ（祝儀）ありのフリー雀荘ルールで勝てる打牌**を学習させるには
報酬をどう設計すべきかを実験的に解明するプロジェクト。

> **位置づけ — これは検証用モデルであり、商用プロダクトではない。**
> 最終目標は商用フリー雀荘特化麻雀AIだが、それは本リポジトリの延長では作らない。
> 本プロジェクトの目的は「**どんな報酬設計が、チップあり麻雀でどんな打牌効果を生むか**」を、
> 実績ある既存アーキテクチャ（Mortal）の上で**安く・速く調査する**こと。
> ここで得た**知見**を持って、商用版は別途ゼロから実装する想定
> （ライセンス事情・データ事情による。詳細は末尾）。

---

## 現在のアプローチ: 教師データ非依存・自己対戦 PPO

初期フェーズ（offline CQL + online per-move TD、`MORTAL_UPSTREAM_README.md` 以前の
旧構成）は、天鳳2009棋譜を教師とした CQL が「赤を持ったら鳴く」を分布外行動として
抑制してしまう天井に突き当たり、online の per-move TD（Q_chip 分離）でも安定した
解決に至らなかった。診断の結果、経済報酬の設計自体は誤りではなく、DQN の学習不安定は
MSE 損失がチップ由来の外れ値に支配される最適化アーチファクトであり、Huber損失
（δ=15）で解消することを確認。これを受け、**教師データ（CQL）を完全放棄し、
on-policy PPO 自己対戦に全面移行**した（`freeparlor/docs/design/ppo_migration_design.md`）。

- **報酬**: `reward = α·(素点Δ/1000) + γ·(GRP順位価値Δ) + β·(チップ枚数Δ × 5.0)`、
  α=γ=β=1（1チップ=5000点の実ルールに一致する真の経済定数）。チップ専用Qヘッドは廃止し、
  単一QをHuber-robustなMCリターンで学習（`reward_design_teacherfree.md`）。
  鳴き頻度を強制する明示項は無く、チップ経済 vs 守備・面前の機会費用から**内生的に
  出現するはず**という設計。
- **アーキテクチャ**: 共有Brain（ResNetエンコーダ）→ PolicyHead + ValueHead。
  1局=1エピソード、GAE(γ=1.0, λ=0.95)。PolicyHead は旧 dueling DQN の `a_head` から
  初期化（Boltzmann方策として厳密に等価、蒸留不要）。GRP（順位価値LSTM）は凍結流用。
  自己対戦基盤（server / trainer / client×3）は既存を再利用、相手プールは
  最新checkpoint 50% + 過去K=5チェックポイントから一様50%。
- **未解決だった論点**: 赤ドラ保持時の鳴き（`freeparlor/docs/design/
  reward_design_teacherfree.md` 時点で 20 positive / 28,672 局という希少事象）が
  経済報酬だけで内生的に出現するかどうか。これを検証するため、**探索の希少性への
  対処を3段階（Stage1〜3）に分けた実験ラダー**を設計・実施した。

---

## 探索ラダー: Stage1〜3（全段不成立で閉幕）

赤ドラ保持時の鳴きが自然に出現しない問題について、「機会費用（立直の方が本質的に得）」
仮説と「探索不足・学習の実装的な壁」仮説を切り分けるため、単一変数アブレーションで
3段階の介入を順に試した。

| Stage | 内容 | 判定 |
|---|---|---|
| **Stage1** | 純粋な自己対戦（介入なし、entropy探索のみ） | **不成立**。方策は「立直マキシマリズム」（π(立直) 0.42→0.92）をチップ戦略として発見し、その副作用として鳴みが沈んだ。純粋な経済報酬+entropy+自己対戦だけでは希少な赤鳴みを自力発見しない（`ppo_p3_stage1_result.md`） |
| **Stage2** | 配牌 rejection sampling で赤保持局面の遭遇率を訓練時のみ2.6倍に濃縮（eval は常に自然分布） | **不成立**（分岐2: 倍率0.185× < 2.0×閾値、有意な下降トレンド）。遭遇機会を増やしても鳴み判断改善は伸びず、機会費用ギャップはむしろ拡大（5.68→7.81）。濃縮分布で学習した方策を自然分布に配備すると有意な性能損失（**配備税**）が生じることも新たに確認（`ppo_p3_stage2_result.md`） |
| **Stage3** | 鳴き実行への anneal 付き per-decision ボーナス（b=5.0、step0-4000固定→4000-8000で0へ線形減衰→8000-16000は正典報酬のみの判定窓） | **不成立**（分岐2: 判定窓倍率1.066×だが強い下降トレンド、slope/SE=−21で Stage1 均衡へ収束）。鳴み局の収支は改善（−0.95→−0.33）したが損益分岐に届かず、機会費用ギャップ（~5.5–5.9チップ）が支配項のまま。anneal設計により配備税はゼロ（Stage2の教訓が有効だったことを確認）（`ppo_p3_stage3_result.md`） |

**結論**: 探索の遭遇率（Stage2）・報酬の学習誘導（Stage3）のいずれの介入でも
赤鳴みの経済的合理性は覆らず、**「機会費用が本質的に高い」という内在的仮説を支持**して
探索ラダーは全段不成立のまま閉幕。

---

## 現在進行中: DRCA プローブ（反実仮想アドバンテージの直接測定）

探索ラダー閉幕を受け、診断計測器として **DRCA（duplicate rollout counterfactual
advantage）プローブ**を設計・実装・実行中（`freeparlor/docs/design/
drca_probe_design.md`）。訓練への介入ではなく、同一seedの局面を「鳴く」腕と
「鳴かない」腕にfork-by-replayで分岐させ、Q(s,鳴く)−Q(s,鳴かない) を duplicate
rollout で直接測定する。目的は、Stage1-3の結果が (i) 内在的機会費用、
(ii) credit-assignment失敗、(iii) 競技力不足、(iv) 報酬設計そのものの符号ミス
のどれに起因するかを切り分けること。

2026-07-15時点、パイロット測定（50分岐点）は完了・監督側検証合格。規模確定
（K=8 / N=485）済みで、本測定第1枠（セット(a) × Stage1-16000 checkpoint）を
並走中。最新の進捗は `CLAUDE.md`「現在の状態」節を参照。

---

## アーキテクチャ（実装詳細）

- **ゲームエンジン**: libriichi（Rust）。フリー雀荘ルールのため改修
  （`agari_detail` で赤/裏/一発/役満を公開、チップ計算・配牌 rejection sampling
  （Stage2）を追加）。
- **学習**: PyTorch。192×40 ResNetエンコーダ + PolicyHead/ValueHead（PPO）。
  GAE(γ=1.0, λ=0.95)、PPO clipped surrogate + Huber(δ=15) value loss + entropy bonus。
- **自己対戦**: server / trainer / client×3 の3プロセス構成（GPU workload は
  常に1系統、学習とevalの同時実行は禁止）。

---

## 対象ルール

4人打ち・喰いタン・赤×3・25000持ち30000返し・ウマ10-20・オカあり、β=1
（1チップ=5000点）。

---

## 再現方法

### 環境
```bash
# WSL2（distro名 "mahjong"）/ RTX / CUDA / conda env "mortal"
conda activate mortal
```

### libriichi ビルド（Rust改修後）
`PYO3_PYTHON` の明示指定・`CARGO_TARGET_DIR` の未設定確認・import スモークが必須。
手順は `CLAUDE.md`「環境」節を参照（過去のビルド事故を受けて明文化）。
```bash
cargo build --release -p libriichi --lib
cp -f target/release/libriichi.so mortal/libriichi.so
PYTHONPATH=mortal python -c "from libriichi.stat import Stat"
```

### PPO 自己対戦学習
run 発進は `runs/` の spawn ランチャ（`freeparlor/scripts/run_ppo_*.sh`）経由。
発進前に必ず `freeparlor/scripts/verify_ppo_p1.py` の全検定PASSと disk 空き容量を
確認する（設計・run規約は `CLAUDE.md` 参照）。

### 評価
標準argmax eval バッテリー、grp_baseline 1v3対戦、Stage間メタ対決probeの3レンズを
`freeparlor/scripts/run_eval_battery_*.sh` 等で実行。

---

## 主要な文書

- `freeparlor/docs/design/ppo_migration_design.md` — PPO移行の設計正典
- `freeparlor/docs/design/reward_design_teacherfree.md` — 報酬設計の確定事項
- `freeparlor/docs/design/stage2_design.md` / `stage3_design.md` — 各Stageの設計・
  事前登録済み判定条件
- `freeparlor/docs/reports/ppo_p3_stage1_result.md` / `_stage2_result.md` /
  `_stage3_result.md` — 各Stageの判定結果
- `freeparlor/docs/design/drca_probe_design.md` — 現行DRCAプローブの設計・解釈条件
- `CLAUDE.md` — プロジェクト全体史・現在の状態・作業規律（最も詳細で最新）

---

## 今後

### 本リポジトリ内（調査の継続）
- DRCA プローブ本測定 → 判定（進行中）
- 探索ラダー閉幕後の方針設計（立直マキシマリズムの商用採否、経済定数変更、
  敵対的搾取者訓練の要否など）

### 商用版（別実装・本リポジトリの外）
商用フリー雀荘AIは、本リポジトリの延長ではなく**ゼロから実装する**予定。理由は2つ：

1. **ライセンス**: ベースの Mortal は AGPL-3.0-or-later（下記）。ネットワーク提供にも
   ソース公開義務が及ぶため、これを土台にしたクローズドソースの商用製品は作れない。
   本プロジェクトは「AGPLコードで**知見のみ**を得る検証」と位置づけ、商用版は
   コードを継承せず知見だけを用いて独立実装する。
2. **データ事情**: チップありルールの牌譜データは事実上存在しない（天鳳2009は
   チップ無し）。よって商用版は既存棋譜の模倣ではなく、**純粋な強化学習
   （自己対戦中心）**になる見込み。本プロジェクトの報酬設計・探索介入の知見が、
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
