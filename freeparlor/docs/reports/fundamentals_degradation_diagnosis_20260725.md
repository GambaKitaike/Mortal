# 基礎技能（牌理・降り）劣化の診断 — 「PPO が与えられた基礎を壊している」

**日付:** 2026-07-25
**種別:** 診断レポート（事前登録スタイル・コード変更なし・docs-only）
**起草:** 実装エージェント（Claude Code）。**採否・fix の方向づけは監督/Gamba 裁定事項**
**位置づけ:** 探索ラダー閉幕後の 0b 方針セッション（`../ops/policy_session_0b_frame.md`）の一次材料

---

## 0. 問い

Gamba の観察: **鳴きどうこう以前に、そもそも牌理・降りが理にかなっていない打牌が多い。**
Stage1/2/3 の探索ラダーが追っていた「鳴き頻度」とは別軸の、より基礎的な失着。

原因候補として素朴に挙がる3択を切り分ける:

1. **教師あり学習が足りない**（人間データ由来の基礎が入っていない）
2. **PPO の学習が足りない**（自己対戦をもっと回せば改善する）
3. **表現力が足りない**（学習パラメータ数が不足していて基礎を表現できない）

**本レポートの判定（§1–§5 の証拠から機械的に follow、以後 post-hoc 変更しない）:**
3択のいずれも素朴な読みは外している。正しい診断は
**「②に近いが符号が逆 — PPO は『足りない』のではなく、init が既に持っていた基礎を
自己対戦の過程で劣化させている」**。①（教師ベース）は既に入っており、③（容量）は
製品版 Mortal と同一で律速ではない。

---

## 1. ③表現力（容量）の棄却

ネットワークは **192ch × 40 residual block、約 10.8M パラメータ**:

- `mortal/config.toml:53-55` → `[resnet] conv_channels = 192 / num_blocks = 40`。
- `mortal/model.py` の `Brain`(ResNet 40 block) + `ActorCritic`（v4 は 1024-d φ 上の
  Linear ヘッド、`PolicyHead` 1024×46 / `ValueHead` 1024×1）。encoder ≈10.79M + ヘッド 48,175
  ≈ **10.8M**（起動時 `train_ppo.py:98-99` が `parameter_count` で実測ログ出力）。

これは**製品版 Mortal と同一アーキテクチャ**。同じ 192×40 で天鳳超人級の牌理・降りが
実現できている（本家実績）以上、**容量は律速ではない**。ここを増やす投資は筋が悪い。

→ **③棄却。**

## 2. ①教師ベースは既に init に入っている

PPO は乱数初期化ではなく、人間譜由来のオフライン RL 重みから warm-start している:

- 全 freeparlor config が同一 init を指す:
  `freeparlor/configs/ppo_stage3.toml:44` →
  `init_checkpoint = '.../phase4/beta1_huber_192x40/mortal.pth'`。
- `train_ppo.py:90-96`: encoder 全体（`mortal`）を init から `load_state_dict`、
  actor/critic は DQN ヘッドから warm-start（`load_ppo_from_mortal_checkpoint`）。
- `beta1_huber_192x40` の系譜（`mortal/train.py`）:
  - データ = **人間の天鳳ログ** `config.toml:33-34`
    （`[dataset] globs = ['.../data/tenhou/2009/**/*.mjson']`、`mortal/dataloader.py`）。
  - 目的 = MC リターン回帰（`train.py:266` `q_target_mc = gamma**steps * kyoku_rewards`）
    ＋ CQL 正則化 ＋ Huber 損失（`train.py:275-278`、delta=15）。
    ※リテラル move imitation（cross-entropy BC）ではなく value/advantage 回帰経由だが、
    **基礎は人間譜に接地している**。
  - 「huber」採用理由は `../design/reward_design_teacherfree.md`（MSE が chip 外れ値で崩壊、
    Huber で和了 14%→40%・avg_rank 1.246→1.006 を回復）。
- Stage1 の専門家レビュー自身が裏書き:
  `qualitative_expert_review_20260715.md:39` 「牌理の土台を人間牌譜由来の重み（本家 Mortal）
  から始める路線」。

→ **基礎は init 時点で存在している。①「教師あり不足」は素朴には成立しない**
（不足しているのではなく、後段で壊れている）。

> 命名の注意: リポジトリの `config.toml:5` は `beta1_192x40`（mse variant）を指すが、
> PPO の実 init は全 freeparlor config が指す**兄弟 run `beta1_huber_192x40`**。

## 3. 劣化の一次証拠 — 訓練後は init より弱い

「never good（init 自体も牌理に穴）」と「degraded（PPO で更に悪化）」の**両方**が出ている。

### (a) head-to-head（訓練済み1 vs init3、grp_baseline 1v3）

baseline = `beta1_huber_192x40`（=PPO の init）を3席に置き、訓練済み1席を対戦させる。
**全 stage で訓練後 checkpoint が init より弱い**（放銃増・和了減・着順悪化）:

| stage / label | avg_rank | agari | houjuu | 出典 |
|---|---:|---:|---:|---|
| init（ミラー較正脚） | **2.4750** | 20.67% | **12.10%** | stage1 §7 表1 (L139) |
| Stage1 step16000 | 2.5325 | 18.76% | 15.15% | stage1 §7 表1 (L140) |
| Stage2 step16000 | **2.7050** | 17.61% | 15.42% | stage2 §7 表 (L134) |
| Stage3 step16000 | 2.5800 | 18.80% | 15.18% | stage3 §7 表 (L121) |

放銃率が **12.10% → 約15.2%** に一貫悪化、和了率が **20.67% → 約18.8%** に低下、
平均着順も init 側（2.4750）に届く訓練 checkpoint が一つも無い。

### (b) 自己対戦バッテリーの放銃率推移（Stage3、`ppo_p3_stage3_result.md` §1）

init 12.29% → step2000 15.33% → **step4000 17.59%** → step8000 15.42% →
step12000 13.74% → step16000 13.33%。
鳴きボーナス期（b=5.0 固定の step0–4000）に放銃が急騰し、anneal 後も init 水準へ
完全には戻らない。**基礎の健全性が訓練で一時的に大きく崩れる**ことの直接痕跡。

### (c) メタ対決 Stage3-16000 vs Stage1-16000 ×3（`ppo_p3_stage3_result.md` §3/§7d）

avg_rank 2.6100、素点 −7.42±0.78（理論ミラー値 −5 から下方逸脱）、
**チップ −0.675±0.236 ≈ −2.9SE**、放銃 14.30%。監督結論（§7d L312-314）:
残留鳴きは立直メタで中立でなく**負債**。Stage2-16000 vs Stage1-16000 が
ミラーパリティだったのと対照的で、**Stage3 の基礎は Stage1 より悪い**。

### (d) 専門家レビュー（定性）

- **Stage1**（`qualitative_expert_review_20260715.md`）:
  L18「細かい牌理の未習得（1 の孤立牌より 1345 の形が強い、の形の価値評価ができない）」、
  L21「二立直に対し、共通現物を切ればテンパイ維持できる所でなぜか別牌を切ってベタオリ」。
  → **init+Stage1 の時点で既に牌理・押し引きに穴（never good）**。
- **Stage3**（`qualitative_expert_review_stage3_20260716.md`）:
  L31-33「**降りの規律が崩壊気味**。もはや『降りている』と言えない打牌が多い。途中まで
  現物を切っていたのに突然危険牌を切って放銃するパターンを頻繁に目視。Stage1 比での
  放銃率上昇と整合」。L49「(4) は Stage1 所見(2) から**悪化**した」。
  → **PPO で更に悪化（degraded）**。

### 配備税（参考、`ppo_p3_stage2_result.md` §7c）

Stage2（p_enrich=1.0 の赤濃縮分布で訓練）は自然分布配備で有意な性能損失
（avg_rank 2.4750→2.7050 ≈2.9SE）。これは**分布シフト固有**の損傷で Stage1/3 ではほぼ 0。
§3(a) の「訓練後 > init 放銃・着順悪化」は配備税とは別口の、分布シフトなしでも起きる劣化。

## 4. 原因仮説 — PPO 目的関数の3つの構造的欠陥

いずれも「設計上そうなっている」ことをコードで確認できる（バグではなく設計の帰結）。

### (主犯) アンカー不在 — 良い方策への引き戻しが無い

`ppo_loss`（`mortal/ppo.py:79-109`）は
`total = policy_loss(clip) + c_vf·value_loss − c_ent·entropy` のみ。
- **凍結 init への KL ペナルティ・trust-region・蒸留・BC が一切無い**
  （`kl`/`reference`/`anchor`/`distill`/`behavior_clone` は mortal/*.py に 0 件）。
  init は `train_ppo.py:90-96` で重み初期化に一度使われるだけで、以後どの損失にも登場しない。
- PPO clip（`ppo.py:96-99`）が制約するのは**1つ前の自分**（`logp_old`）に対してだけ。
  「毎歩は前の自分から近い」しか保証しないので、16000 step かけて**良い init から
  いくらでも遠くへドリフト可能**。これは RLHF で KL 項が担う「良い参照からの逸脱抑制」が
  丸ごと欠けた状態＝catastrophic forgetting / 報酬に都合よく基礎を捨てるドリフトの温床。

### (増幅1) 自己対戦均衡の退化

`freeparlor/configs/ppo_stage3.toml:180-184` → `[opponent_pool] enabled=true, past_k=5,
latest_prob=0.5`。相手は**自分の過去 checkpoint のみ**（全員が同じ立直マキシマリスト方向へ
収束）。雑な降り・悪い牌理を**咎める相手が居ない**ため、「正しく降りる/正しく形を作る」
ことへの勾配が消える。基礎を教えていた局面（強い相手の攻めに正しく対応する等）が
訓練分布から抜け落ちる。Stage1 監督ノートも同仮説を明記
（`qualitative_expert_review_20260715.md:31-32` 押し引きの現物規律への学習圧力が弱い）。

### (増幅2) スパース＆終局のみ報酬 + 攻撃側への牽引

- 報酬は**局終端 step のみ非ゼロ**（`mortal/ppo_dataloader.py:26-36`、episode=局）。
  牌理・降りの良し悪しへの**密な信号が無く**、結果経由の間接シグナルにしかならない
  ＝ドリフトの最初の犠牲になりやすい。
- 報酬合成（`mortal/reward_calculator.py`）でチップは重み ×5（`config.toml:50` chip_value=5.0、
  beta=1.0）。Stage3 は更に per-decision 鳴きボーナス（b=5.0）で**攻撃側へ露骨に牽引**。
  基礎規律より攻撃・チップ稼ぎに勾配が向く。

**序列:** 主犯は**アンカー不在**（基礎を守る構造がゼロ）。均衡退化と報酬設計が
それを増幅する。

## 5. なぜ「②もっと PPO を回す」が効かないか

Stage3 判定（`ppo_p3_stage3_result.md` §9）: 判定窓で **slope/SE = −21.03 の有意下降**、
四半期倍率 1.445×→0.690×、最終バケットは Stage1 平衡方向へ収束。
**回すほど基礎が沈む**方向にトレンドが出ている。よって「学習量が足りない（もっと回せ）」
という②の素朴解釈は反証されている。改善には**目的関数側の変更**が要る。

---

## 6. 候補実験の列挙（提案のみ・採否は未裁定）

いずれも「教師フリー（教師データ非依存）」というプロジェクト前提と程度の差でトレードオフする。
**採否・優先順位は監督/Gamba 裁定事項**。ここでは切り分け力とコストの整理のみ。

- **A. アンカー付き PPO** — 凍結 init への KL/蒸留ペナルティを損失に追加
  ＋ opponent pool に強い init を常駐（現状 pool を強相手で恒久固定）。
  init は既に読み込み済みで実装は最小、RLHF 標準手法。前提侵食が最も小さい
  （init 自体は既に人間譜由来なので「教師フリー」の建前は現状とほぼ不変）。
  §4 の主犯（アンカー不在）を直接叩く。単一変数アブレーションとして「アンカーの有無だけ」で
  基礎劣化が止まるかを切り分けられる。
- **B. 補助教師損失の再導入** — Tenhou ログの BC/蒸留損失を PPO に混合
  （`mortal/dataloader.py` は既存）。基礎を最も強く再教育できるが、
  **「教師データ非依存」前提を最も直接的に放棄**する。前提変更の裁定が必須。
- **C. 自己対戦分布の是正のみ** — 損失は一切変えず、退化した均衡（§4 増幅1）だけを壊す
  （強い init を pool 常駐＋相手多様化）。最も軽い介入で、**「均衡退化」単独で
  基礎劣化を説明できるか**を切り分ける対照実験になる。A のサブセット。

**推奨の下地**（裁定判断の材料として）: 前提侵食・実装コスト・主犯への直撃のバランスから
**A が第一候補**、C は「均衡単独犯かどうか」の安価な先行切り分けとして A の前段に置ける。
B は前提放棄を伴うため独立の方針判断。

## 7. 安価な確証プローブ（提案・別タスク）

§3 は既存 eval からの転記で既に強いが、**基礎を鳴きから分離**して測る確証が望ましい:

- init vs 各 checkpoint で **放銃率・現物遵守率・牌理近似指標**を argmax eval で比較
  （鳴き頻度を制御して基礎軸だけを見る）。
- 大半は既存資産で再現可: grp_baseline 1v3 ハーネス（§3a の出所）＋
  牌譜 HTML ビューア（`freeparlor/scripts/mjai_log_to_html.py`）。DRCA と違い新ハーネス最小。
- GPU 依存のため実行は DRCA 完走後の別タスク。

## 8. 0b 方針セッションへの接続

Stage3 監督ノート（`qualitative_expert_review_stage3_20260716.md:52-55`）:
「鳴きが増えたが下手」は**商品性の観点で最悪の象限**。本診断はこれを一段深め、
**基礎そのものが訓練で劣化している**ことを一次証拠で固定した。

`../ops/policy_session_0b_frame.md` への入力:
- **議題1（商用採否）**: 現行 PPO 系列は init（本家 Mortal）を配備性能で**下回る**。
  「立直マキシマリズムをそのまま商用」路線は基礎劣化のコストを負う。
- **議題3（敵対的搾取者訓練の要否）**: §4 増幅1（均衡退化）が基礎劣化の一因である以上、
  相手分布の設計（搾取者・多様化）は基礎規律の維持と同じ問題の裏表。
- 本診断は探索ラダー閉幕後の「PPO 目的関数の作り直し（§6 A/B/C）」を方針選択肢として
  卓上に載せる。

---

## 付録: 参照（変更していない）

- コード: `mortal/model.py`, `mortal/config.toml:33-55`, `mortal/train.py:266-303`,
  `mortal/dataloader.py`, `mortal/train_ppo.py:90-99`, `mortal/ppo.py:79-109`,
  `mortal/ppo_dataloader.py:26-36`, `mortal/reward_calculator.py`,
  `freeparlor/configs/ppo_stage3.toml:44,180-184`
- 文書: `ppo_p3_stage1_result.md` §6-7, `ppo_p3_stage2_result.md` §7c,
  `ppo_p3_stage3_result.md` §1/§3/§7c/§7d/§9,
  `qualitative_expert_review_20260715.md`, `qualitative_expert_review_stage3_20260716.md`,
  `../design/reward_design_teacherfree.md`, `../ops/policy_session_0b_frame.md`
