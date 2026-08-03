# フリー雀荘特化 麻雀AI — 報酬設計の調査基盤

[Mortal](https://github.com/Equim-chan/Mortal)（天鳳＝着順特化の強化学習麻雀AI）を
ベースに、**チップ（祝儀）ありのフリー雀荘ルールで勝てる打牌**を学習させるには
報酬をどう設計すべきかを実験的に解明するプロジェクト。

> **現在地（2026-08-02）**: 探索ラダー（Stage1〜3）・anchor 系列・L1 系列はいずれも閉幕し、
> 現行軸は **候補2（敵対的搾取者）**。各系列の総括は下記の該当節と
> `freeparlor/docs/reports/*_summary_*.md` が正。進行中 run の状態は `CLAUDE.md`。

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
初期フェーズ（Phase1〜4）の記録は `freeparlor/docs/archive/dqn/`（当時の README は
同ディレクトリの `dqn_era_readme.md`）に保全してある。

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

## 診断: DRCA プローブ（反実仮想アドバンテージの直接測定）— 一部完了・中断

探索ラダー閉幕を受け、診断計測器として **DRCA（duplicate rollout counterfactual
advantage）プローブ**を設計・実装・測定した（`freeparlor/docs/design/
drca_probe_design.md`）。訓練への介入ではなく、同一seedの局面を「鳴く」腕と
「鳴かない」腕にfork-by-replayで分岐させ、Q(s,鳴く)−Q(s,鳴かない) を duplicate
rollout で直接測定する。目的は、Stage1-3の結果が (i) 内在的機会費用、
(ii) credit-assignment失敗、(iii) 競技力不足、(iv) 報酬設計そのものの符号ミス
のどれに起因するかを切り分けること。

事前登録した実効5枠のうち **2枠を完了**（各 N=485 分岐点 / K=8 rollout、
cluster-robust SE）:

| 枠 | 対象方策 | ΔQ̄（千点） | cluster SE | \|ΔQ̄\|/SE |
|---|---|---:|---:|---:|
| 第1枠 | Stage1-16000（学習済み） | **−3.26** | 0.72 | 4.53 |
| 第2枠 | init（PPO 未学習・基礎技能は無傷） | **−2.14** | 0.48 | 4.46 |

**赤保持時の鳴きの反実仮想価値は、基礎技能が無傷の init でも有意に負。** つまり
反鳴き均衡は「PPOが基礎を壊した結果のアーティファクト」ではなく、この経済（β=1・
立直ペイロード）の性質である、と挟み撃ちで確認できた。ただし分布は裾支配で、主判定の
推定対象は「無差別な鳴きの平均」であり「選択的な鳴きの価値」ではない
（`qualitative_expert_review_drca_frame1_20260722.md` §4）。

**残り3枠は未測定**（第3枠は進捗4.6%で中断・成果物は退避済み、a_s3mid / b_s3final は
未着手）。中核の問いが上記2枠で決着したこと、および優先軸が下記 anchor 系列へ移ったことに
よる打ち切り裁定（`drca_probe_design.md` §5a-1c）。これに伴い主 contrast 2・contrast 3 は
評価不能で、設計書 §4 の解釈シナリオ割り当てには到達していない。

---

## 閉幕: anchor 系列（アンカー付き PPO）— 基礎技能劣化への対策

探索ラダーと並行して、**PPO 自己対戦が Mortal 由来の基礎技能（牌理・降り）を
有意に劣化させている**ことが判明した（放銃劣化 z = +3.9〜+4.4、和了劣化 −2.1〜−3.6、
全 stage で有意。`fundamentals_significance_pass_20260725.md`）。チップ獲得でほぼ相殺して
損益分岐に見えていたが、内訳では基礎を支払っている。定性レビューでも「鳴きの入口は増えたが
鳴いた後のサブゲームが未熟」「降りの規律が崩壊気味」という像が一致した。

そこで**凍結した init（教師データ由来の基礎技能を持つ方策）をアンカーとして参照させる**
単一変数アブレーションを3 arm 実施した（`anchored_ppo_design.md`、判定条件は事前登録済み
= 判定窓 step 8000–16000 / 放銃差 z<2 / チップ +方向 ≥1SE / 1v3 両脚 n=800）。

| Arm | 介入 | 判定1（基礎維持） | 判定2（経済） |
|---|---|---|---|
| **C** | opponent pool へ凍結 init を `anchor_prob=0.25` で常駐 | ✗ | ✗（+0.45SE） |
| **K** | `ppo_loss` に凍結 init への masked full KL 項（`kl_beta=0.1`） | ✗（z=+2.51） | ○（+2.18SE） |
| **b04** | 同上・`kl_beta=0.4` | ✗（z=+2.21） | ○（+1.75SE） |

**3 arm すべてで判定1 は不成立**（`anchor_series_summary_20260731.md`）。得られた知見は2つ:

- **引き戻しは相手分布（pool）より損失側（KL）に置くほうが効く**（単一変数で確定）
- **中心的知見: 基礎技能は悪化したが EV は大きく上昇した**。チップ +0.4〜0.5枚/半荘
  （= +2.0〜2.5千点相当）で、放銃が有意に増え和了も減っているのに avg_rank は不変・
  収支は改善した。⇒ **チップありのルールでは基礎技能（放銃率・和了率）だけで
  AI の性能を説明できない**

---

## 閉幕: L1 系列（最適化衛生）— 劣化の原因は最適化にあるか

8 run 横断の診断（`ppo_optimization_health_20260725.md`）で、全 run に共通する
構造的事実が3つ見つかった: (1) `minibatch_size=512` は一度も効いていない
（1 optimizer step = 1半荘の full-batch × 4 epochs）、(2) **バッチ到着時点で既に
clip_fraction ≈ 0.20–0.33**（= 収集経験の約2割が恒常的に勾配に寄与していない）、
(3) 4 epochs は trust region 占有をほぼ動かさない。

因果は未検証だったので、**改善策ではなく対照実験**として2本を単一変数で走らせた
（判定条件は anchor 系列と同一計測器）。

| arm | 単一変数 | 判定1 放銃差 | 判定2 チップ | 合算 |
|---|---|---:|---:|---:|
| 参照 plain PPO | — | +2.876pp（z=+5.43） | +0.630（+2.63SE） | +2.809 |
| **O1** | `submit_every` 50→10 | ✗ +2.635pp（z=+4.91） | ○ +0.381（+1.62SE） | −2.128 |
| **O3** | `ppo_epochs` 4→1 | ✗ +1.620pp（z=+3.13） | ○ **+0.639（+2.77SE）** | **+5.237（+1.89SE）** |

**判定1 は両 arm とも不成立**だが、**O3 は放銃劣化が系列最小・チップが系列最大で、
合算（素点+順位点+チップ）が init を上回った**（全ストリームが + 方向は本プロジェクト初）。
事前登録した識別指標（init からの行動シフト量）により「単に学習が進んでいないだけ」は
否定済み。

**系列の中心的知見**（`l1_series_summary_20260802.md`）:

| | staleness（バッチ） | ×epochs = 経過 optimizer 更新 | clip@epoch1 |
|---|---:|---:|---:|
| 参照 | 91.5 | 366 | 0.2019 |
| O1 | 73.0（−20%） | 292（−20%） | 0.2046（**不変**） |
| O3 | 90.7（不変） | **91（−75%）** | **0.0521（−75%）** |

⇒ **clip を支配していたのは staleness ではなく「古い間にパラメータをどれだけ動かしたか」**。
診断の事実(2) の主因を特定した。

---

## 進行中: 候補2 — 敵対的搾取者（立直マキシマリズムは搾取可能か）

これまでの全 run は challenger を「mortal 的な相手のプール」と戦わせてきた。つまり
Stage1-16000 は**mortal 的な相手へのベストレスポンス**として育った方策であり、
**完成した Stage1-16000 自身へのベストレスポンスは一度も計算していない**。

検定する命題は1つ — **H(EX): 立直マキシマリズム均衡は搾取可能か**
（`adversarial_exploiter_design.md`、2026-08-02 凍結）。単一変数は
**訓練 rollout の相手分布を「自己対戦プール」から「凍結標的 ×3」へ替える**こと
（実装は既存の `OpponentPool` の外部 checkpoint 分岐を `anchor_prob=1.0` で使う。新規コードゼロ）。

判定は 2×2 の差分の差分で、「ただ強くなった」と「標的固有の搾取」を分ける:

| セル | challenger | baseline ×3 |
|---|---|---|
| A | 搾取者 | **標的**（Stage1-16000） |
| B | 搾取者 | init |
| C | init | **標的** |
| D | init | init（ミラー = 理論値 0） |

**DiD = (A − C) − (B − D)** の**順位点 EV**（ウマオカが非対称なので avg_rank ではなく
EV で判断する）が **≥ +2SE** なら搾取の存在証明。どちらに転んでも知見になる
（搾取あり = 均衡は自己対戦の産物 / 搾取なし = ベストレスポンスの不動点に近い頑健均衡）。

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

**西入（サドンデス）は採用しない**（フリー雀荘ルールでは通常行わない）。
libriichi は天鳳準拠で西入を実装しているため、`[env] enable_west_round` で
config フラグ化した（**既定 `true` = 天鳳準拠でビット不変**、新しい run は
`false` を明示する）。実装と run-validation は `parlor_rule_west_round_design.md` §7/§8。

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

`freeparlor/docs/INDEX.md` が全文書の索引（パス・日付・ステータス・要約）。主要なものは:

- `freeparlor/docs/design/ppo_migration_design.md` — PPO移行の設計正典
- `freeparlor/docs/design/reward_design_teacherfree.md` — 報酬設計の確定事項
- `freeparlor/docs/design/stage2_design.md` / `stage3_design.md` — 各Stageの設計・
  事前登録済み判定条件
- `freeparlor/docs/reports/ppo_p3_stage1_result.md` /
  `ppo_p3_stage2_result.md` / `ppo_p3_stage3_result.md` — 各Stageの判定結果
- `freeparlor/docs/design/drca_probe_design.md` — DRCAプローブの設計・解釈条件・打ち切り裁定
- `freeparlor/docs/reports/fundamentals_degradation_diagnosis_20260725.md` /
  `fundamentals_significance_pass_20260725.md` — 基礎技能劣化の診断と有意性
- `freeparlor/docs/design/anchored_ppo_design.md` — anchor 系列の設計（closed）と
  `freeparlor/docs/reports/anchor_series_summary_20260731.md` — **その総括**
- `freeparlor/docs/design/l1_o1_submit_every_design.md` / `l1_o3_ppo_epochs_design.md` —
  L1 系列の事前登録（凍結済み）と
  `freeparlor/docs/reports/l1_series_summary_20260802.md` — **その総括**
- `freeparlor/docs/design/adversarial_exploiter_design.md` — **現行軸**（候補2）の
  事前登録（凍結済み）
- `freeparlor/docs/design/parlor_rule_west_round_design.md` — 西入の除去（W1/S1）
- `freeparlor/docs/design/teacherfree_training_candidates.md` — 教師データ非依存の
  訓練方式候補（cold start / 均衡脱出の分離。DRAFT・非事前登録。下記「商用版」の土台）
- `freeparlor/docs/ops/qualitative_review_protocol.md` — レンズ4（人手の牌譜レビュー）の
  手順。**所見は必ず定量指標に突き合わせる**
- `freeparlor/docs/ops/project_history.md` — 2026-07-06以降の時系列経緯
- `CLAUDE.md` — 現在の状態・作業規律（進行中runの正）

---

## 今後

### 本リポジトリ内（調査の継続）
- **候補2（進行中）**: 搾取者 run 完走 → eval（基礎技能 n=800 + DiD 4セル n=1600）→
  レンズ4（搾取戦術が牌譜に読めるか）→ 判定
- **候補1（次）**: oracle 蒸留 / シミュレータ由来の自己教師あり補助タスク（問題 I）
- 未決の裁定: **`ppo_epochs=1` を以後の既定にするか**（O3 は同じ計算資源でより良い方策に
  見えるが判定1 は不成立で、逐次スクリーニング規律では既定化に 2 seed 目が要る）
- レンズ4 起票の計装3件（打点・安全度を含む打牌評価 / 七対子ルート選択 / 役牌暗刻落としの
  横断測定）— GPU 不要
- L1 の残り O2（複数半荘の batch 束ね。判定窓を消費半荘数で定義し直す必要あり）
- DRCA 残枠の再開可否（現在は保留・成果物は退避済み）

### 積み上がった知見（系列を跨いで効くもの）
1. **チップありのルールでは基礎技能だけで性能を説明できない**（anchor 総括）
2. **clip を支配するのは staleness ではなく経過 optimizer 更新数**（L1 総括）
3. **損傷もチップ獲得も最初の 2000 step でほぼ完了する**
   （`anchor_checkpoint_trajectory_20260728.md`）
4. **同一 config の run 間ばらつきが効果量と同程度**なので、新規アイデアは
   1 seed → 事前登録の判定を満たしたものだけ 2 seed 目、という逐次スクリーニングを敷いている
5. **eval の決定論は `EVAL_SEED_COUNT`（GPU バッチ形状）が同一であることを条件とする**
   （`parlor_rule_west_round_design.md` §7-3）

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
