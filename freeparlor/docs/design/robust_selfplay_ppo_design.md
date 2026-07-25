# 壊れにくい自己学習 PPO — 設計ノート

**日付:** 2026-07-25
**ステータス:** **DRAFT・裁定非関与・実装未承認**。本書の各「推奨」は 0b セッションおよび
anchor Arm K 判定への入力であって決定ではない
**起草:** 実装エージェント（Claude Code）。設計判断・採否は Gamba
**発端:** 外部レビュー（ChatGPT）が「PPO をやめるのではなく、**壊れにくい自己学習 PPO**
として成立させる」という再定義を提示した。その5項目を本プロジェクトの実装状態に
突き合わせて整理したもの
**一次証拠:** `../reports/ppo_optimization_health_20260725.md`（L1 層の実測）、
`../reports/fundamentals_degradation_diagnosis_20260725.md` §3–§5、
`../reports/fundamentals_significance_pass_20260725.md` §1

> **本書は判定条件を書かない。** 事前登録すべき判定条件は、対象 arm を決めた後に
> `anchored_ppo_design.md` §6 の書式で別途凍結する（0b / Arm K 判定後）。

---

## 0. 問題の立て方

外部レビューの整理は本プロジェクトの現状と整合する:

- 目標 = 牌譜依存を減らす / 手段 = PPO 中心の自己学習 / 条件 = 基礎技能を壊さないこと
- 根拠: `fundamentals_degradation_diagnosis_20260725.md` §5 が
  「回すほど基礎が沈む」（Stage3 判定窓 slope/SE = −21.03）を実証し、
  改善には**目的関数側の変更**が要ると結論している

ただしレビューの5箇条書きをそのまま方針にすると3つの問題がある:

1. **層が混ざっている。** 対策の階層が異なり、実装コストも検証コストも桁が違う。
   混ぜると単一変数アブレーションの規律（`CLAUDE.md` ワークフロー規律）が壊れる。
2. **既にある層と空白の層が区別されていない。** anchor 系列が2層をカバーする一方、
   2層は設計・実装ともゼロ。
3. **「商用不可な牌譜への依存の最小化」は anchor 系列と逆行している。** §5 で扱う。

## 1. 壊れにくさの4層分解

| 層 | 内容 | 現状 |
|---|---|---|
| **L1 最適化の衛生** | batch 規模・advantage 正規化・勾配クリップ・off-policy staleness・target-KL | **診断 §4 が仮説に挙げていない層。実測で異常あり（§2）** |
| **L2 参照点の保持** | 凍結 init への KL / 差分のみ学習 / 部分凍結 | KL = Arm K で実装済み・**未発進**。「差分だけ学習」は**前例ゼロ**（§3） |
| **L3 環境（相手分布）** | pool 設計・多様性・staleness | anchor 常駐 = Arm C で実装済み・走行中。pool の staleness と無選抜性は未対処（§4） |
| **L4 運用（早く切る）** | run 中の劣化検知・eval ゲート・checkpoint 採択・監視 | **ほぼ空白 + 既存監視に実害のある穴**（§6） |

外部レビューの5項目の対応: 「既存方策を固定」= L2 の KL（実装済み）/
「差分だけ学習」= L2 の空白 / 「相手分布を崩しすぎない」= L3（実装済み）/
「早く壊れた実験を切る」= L4（空白）/「牌譜依存の最小化」= §5。
**L1 はレビューにも診断にも登場しない盲点。**

### 診断 §4 の仮説序列との関係

診断 §4 は「主犯 = アンカー不在、増幅 = 均衡退化 + スパース終局報酬」という序列を立て、
anchor 系列（C = L3 / K = L2）はその単一変数検定として設計上正しい。
本書は序列の書き換えを主張しない。主張するのは:

> **L1 に問題があると、「C も K も効かなかった」という結果が L1 由来である可能性を
> 排除できない。** L1 は診断 §4 に**追加**されるべき第4の候補である。

## 2. L1 — 最適化の衛生

一次証拠は `../reports/ppo_optimization_health_20260725.md`。要点のみ再掲（8 run 共通）:

- **1 optimizer step = 1 半荘の full-batch**。`minibatch_size=512` は全 run・全バッチで
  一度も効いていない（batch median 168–179、512 超 0.00%）。
  `collate_trajectory_batches` はデッドインポート。
- **バッチ到着時点（epoch 1）で既に clip_fraction ≈ 0.20–0.33**。
  陽性対照 `trainer_step=0`（client と trainer が同一パラメータ）では clip = 0.0000。
  定常状態の staleness は lag 1–2 = **50–100 optimizer step**（`submit_every=50`）。
- **4 epochs は trust region 占有をほとんど動かさない**（e1→e4 の差 −0.0008〜−0.0032）。
  → clip は「この更新の行き過ぎを抑える」機能を果たしておらず、
  **収集経験の約2割が恒常的に勾配に寄与していない**。
- 勾配クリップは `max_grad_norm = 0` で全 config OFF、LR スケジュールなし、
  target-KL early stop なし、value clipping / return 正規化なし、τ anneal なし
  （`ActorCritic.set_tau` は呼び出し元なし）。

### 2a. 候補（単一変数化しやすい順）

| 案 | 内容 | コスト | 備考 |
|---|---|---|---|
| **O1** | `submit_every` を下げる（例 50→10）。staleness の直接削減 | 極小（config 1 値） | **最有力の第一手**。訓練スループットへの影響（client の param 受信頻度）を先に計測すること |
| **O2** | 複数半荘を1バッチに束ねる（`collate_trajectory_batches` を実際に使う） | 小（trainer のループ改造） | batch を ~170 → ~1000 にすれば advantage 正規化の標本統計が安定する。ただし **1 optimizer step の意味が変わるので step 数基準の判定条件が全て読み替えになる**（判定窓 8000–16000 等）。導入時は step 換算を明記 |
| **O3** | `ppo_epochs` を 4 → 1 | 極小 | L1 実測では 4 epochs が trust region をほぼ動かしていない。計算 1/4 で挙動不変なら、浮いた計算を O2 に回せる。**「効果がない」ことの確認が先** |
| **O4** | `max_grad_norm` を有効化（例 1.0） | 極小 | 現在 OFF。~170 サンプルの高分散勾配に対する保険 |
| **O5** | target-KL による epoch 打ち切り | 小 | O3 が通れば不要（epoch が 1 なら打ち切る対象がない） |

**推奨する順序: O1 → O3 → O2。** O1 は 1 変数・極小コストで staleness を直接動かせる。
O3 は「4 epochs が効いていない」という L1 の観測を確認する安価な対照。
O2 は判定条件の step 換算を伴うので最後。

⚠ **O2 の注意（配備税の教訓の類推）**: batch を束ねると 1 step あたりの学習量が変わる。
`ppo_p3_stage2_result.md` §7c が示した「訓練条件を動かしたら配備条件との差が損失になる」
という教訓は分布の話だが、**step 数を基準にした判定条件は run 間で比較不能になる**という
同型の罠がある。O2 を入れる run は step ではなく**消費半荘数**で判定窓を定義すべき。

## 3. L2 — 「差分だけ学習する」

外部レビューの5項目のうち**唯一プロジェクトに前例がゼロ**の項目。
現状は `AdamW(mortal.parameters() + actor_critic.parameters())` の
**単一 param group・単一 LR**（`mortal/train_ppo.py:74-80`）で 10.8M パラメータ全体を動かす。
backbone 凍結オプションも LR 分離もない。

| 案 | 内容 | コスト | 備考 |
|---|---|---|---|
| **D1** | encoder 凍結・head only | 極小 | 表現の適応が一切できない。**下限確認用の対照**として価値がある |
| **D2** | param group 分離（encoder に 1/10 LR 等） | 極小 | 単一変数化しやすい。**安全な先行実験** |
| **D3** | **残差方策**: `logits = ref_logits + Δ(s)`、Δ に L2 罰則 | 小 | **Arm K の ref forward 配線をそのまま再利用できる**（`train_ppo.py:104-124` の ref モデル管理、`:374-383` の `ref_logits_all` 事前計算）。K の後なら追加コストはほぼ配線のみ。**本命** |
| **D4** | adapter / LoRA を residual tower に挿す | 中（Python のみ・Rust 不要） | 表面積が大きい。優先度低 |

**推奨: D3 が本命、D2 が先行実験。**

### 3a. D3（残差方策）の設計上の含意

- Arm K（KL 罰則）と D3 は**同じ問題への別の解**である。K は「離れることに罰金を課す」、
  D3 は「離れられる方向を構造的に制限する」。**両方入れると単一変数でなくなる**ので、
  K の判定が出るまで D3 に着手する意味は薄い。
- D3 は `kl_beta` と違い**β のような量級較正が不要**（Δ の L2 係数はあるが、
  Δ=0 が init そのものという初期条件が構造的に保証される）。
  anchor 設計書 §6a が K に用意した「過強→β/4、過弱→β×4」の機械的再走が
  D3 では不要になる可能性がある。これは D3 の利点。
- **共通の弱点**: どちらも参照点が牌譜由来の init である。§5 参照。

## 4. L3 — 相手分布（Arm C の残余）

Arm C（`anchor_prob=0.25`）で「凍結 init を pool に常駐」は実装済み。
未対処の残余が2つある。

### 4a. pool の staleness

pool の checkpoint は `save_every = 2000` step ごとに保存されたものを glob する
（`mortal/opponent_pool.py:41-45`）。一方 trainee は `submit_every = 50` step ごとに更新される。
**pool の「latest」は最大 2000 step 古い。** L1 の staleness（50–100 step、§2）とは
別口の、2桁大きい遅れ。

### 4b. pool の無選抜性

pool は `step_*.pth` を**無選抜で** glob する。基礎技能が劣化した checkpoint も
等確率で相手になる。`fundamentals_degradation_diagnosis_20260725.md` §3(b) が示した
Stage3 の放銃率推移（init 12.29% → step4000 **17.59%** → step16000 13.33%）を踏まえると、
**最も劣化した時期の checkpoint が pool に恒久的に残り続ける**。

候補（いずれも未設計・裁定前）:

| 案 | 内容 | 備考 |
|---|---|---|
| **P1** | `save_every` を下げて pool の粒度を細かくする | drain 容量とのトレードオフ（1 run ~230–370GB） |
| **P2** | eval 指標で pool 入りを選抜（eval-gated pool） | L4 の eval ゲートと同じ機構を要求する。GPU 1系統ルールとの整合が課題 |
| **P3** | PFSP / league 型の重み付け | 文書に前例なし。表面積が大きい。0b 議題3（敵対的搾取者）と接続する |

### 4c. 観測: 25% の pool 介入では学習方向が動いていない

`ppo_optimization_health_20260725.md` §8（sampled レンズ、訓練分布上、判定非関与）:
Arm C は step 3000–3499 で π(立直|可能) = 0.920 / π(鳴き|可能) = 0.133。
Stage1 の判定窓平均は 0.919 / 0.065。**収束の速さは Stage1 と同等。**

これは **Arm C の失敗ではない**（判定条件は放銃差 z<2 とチップ ≥1SE であり鳴き率は
判定指標ではない）。設計情報としての含意は「**L3 側の確率的介入 25% では
学習方向そのものは動かない**」であり、L2 側（K / D3）の必要性を相対的に上げる。

## 5. 商用パスとの衝突 — アンカー置換可能性の要件化

**外部レビューの「商用不可な牌譜への依存を最小化」は、現状の anchor 系列と論理的に
衝突している。** これは本書で最も重い論点であり、実装の話ではなく方針裁定の話である。

- init = `beta1_huber_192x40` は **天鳳2009ログ上の MC リターン回帰 + CQL + Huber(δ=15)**
  （`fundamentals_degradation_diagnosis_20260725.md` §2）。リテラル BC ではないが
  基礎は人間譜に接地している。
- `ppo_migration_design.md` §2.3 は「教師撤去は**学習ループからの撤去**であり
  prior としての影響は残る」と正しく限定している。しかし
  **anchor 系列は init を学習ループに戻す介入である**（Arm C = 相手分布に常駐、
  Arm K = 損失に常駐、いずれも 16000 step 全区間）。
  診断 §6 A も「init 自体が既に人間譜由来なので教師フリーの建前は現状とほぼ不変」と
  この方向を明示的に選んでいる。
- 天鳳の利用規約は**競合製品・商用利用不可**（`../archive/dqn/dqn_era_readme.md`）。
  Mortal 本体は **AGPL-3.0-or-later**（`README.md`）。

**帰結**: 壊れにくさを anchor で獲得しても、**壊れにくさの供給源そのものが製品に
持ち込めない**。したがって設計要件として次を立てるべきかが裁定事項になる:

> **要件候補 R1: 壊れにくさの機構は、最終的にアンカーを牌譜非依存な参照点に
> 置換できる形であること。**

### 5a. アンカー置換の候補（いずれも文書に前例なし・未設計）

| 案 | 内容 | 牌譜依存 | 備考 |
|---|---|---|---|
| **A1** | 解析的 auxiliary loss（現物遵守・シャンテン数など libriichi から計算可能な量） | **なし** | 「基礎規律」を牌譜でなくルールから供給する。診断 §7 が測定指標として挙げた現物遵守率・牌理近似指標と同じ量を**損失側にも使う**発想。libriichi 側の表面積が要る |
| **A2** | ルールベース teacher の蒸留 | **なし** | eval 経路の `enable_rule_based_agari_guard` が本家準拠で既に存在する。ただし guard は和了判定の補助であって打牌全体の teacher ではない。teacher の強さが上限になる |
| **A3** | self-play from scratch のラダーで得た中間 checkpoint をアンカーにする | **なし** | 牌譜完全非依存だが最も高コスト（`ppo_migration_design.md` §2.3 が「ゼロからの自己対戦はコスト非現実的」として init warm-start を選んだ経緯そのもの） |
| **A4** | 現状維持（init をアンカーとして使い続ける） | **あり** | 研究計測器としては正しい。製品には持ち込めない |

**本書は推奨を出さない。** これは「anchor は研究計測器か、製品アーキテクチャの一部か」
という方針判断に依存し、0b 議題5（`../ops/policy_session_0b_frame.md`）で扱う。

## 6. L4 — 早く壊れた実験を切る

### 6a. 既存監視の穴（defect・バックログ 11）

`CLAUDE.md` 監視期待値の第1項 `trajectory step count mismatch` = 0 について、
**この文字列を出力するコードは repo に存在しない**（`freeparlor/scripts/` と docs にのみ存在。
P2 期の修正で emitter が消えたと推測）。つまり**この監視項目は構造的に常に 0** である。

一方、現行 `mortal/client.py` が実際に出すデータ落ちシグナルは:

- `mortal/client.py:108` `'trajectory game key missing, skipping game ...'`（WARNING、
  **その局を丸ごと捨てる**）
- `mortal/client.py:184` `'trajectory orphan steps (...) for game_id=...'`（WARNING）

そして **run 中の監視 grep（`freeparlor/scripts/run_ppo_p3_stage1_inner.sh:250-259`）は
この2つを見ていない。** 発進前検定（`verify_ppo_p1.py:695-711`）は3種とも数えて 0 を
assert しているので、守られていないのは **run 中の継続監視だけ**。

**対応**: バックログ 11。`inner.sh` の監視 grep に2項目を追加し（FATAL / NOTICE の区分は
実施時に裁定）、死んだ `step count mismatch` の扱いを決め（emitter 復活か監視項目から降格か）、
**`CLAUDE.md` の記述も同一 commit で実装に合わせる**。バックログ4（launcher Cleanup の
run-validation）と同じ Arm K 発進 preflight で消化する。
**片側だけ直すと乖離が別方向にずれるので、記述と実装は必ず同時に直す。**

### 6b. 劣化検知・checkpoint 採択が PPO に無い

- run 中に `houjuu_rate` / `avg_rank` を見て止める機構は無い。停止条件は
  `max_steps` 到達・NaN（`train_ppo.py:506-508`）・watchdog 再起動過多のみ。
- **best checkpoint 保持が PPO に無い**: `control.best_state_file` を参照するのは
  `mortal/train.py:126,479-481`（DQN）のみ。`train_ppo.py:148-163` は
  `mortal.pth` を上書きし `step_%06d.pth` を無選抜で残すだけ。
- 結果として、基礎劣化の検出は**完走後の eval バッテリー + 人手判定にのみ依存**している。
  実際に 2026-07-16 の定性所見が 07-25 の診断まで9日遅れた
  （`../ops/qualitative_review_protocol.md` の制定動機）。

### 6c. 提案: run 中の基礎指標トリップワイヤ（観測のみ）

`mortal/client.py:64-188` `_finalize_ppo_trajectories` は**既に報酬計算のために
mjai ログを再読みしている**。ここで放銃 / 和了カウンタを取って `ppo_diag.jsonl` に
出すのは**ほぼ無料**。

- 測定は訓練 rollout 上なので eval と絶対値が違う（sampled π・guard OFF・相手は pool）。
  **run 内トレンドの警報としてのみ使う。**
- 信号の実在は確認済み: Stage3 自己対戦バッテリーで
  放銃 12.29% → 17.59%(step4000) → 13.33%(step16000)
  （`fundamentals_degradation_diagnosis_20260725.md` §3(b)）。

⚠ **禁則との整合（設計の必須条件）**:
- **観測のみ。** 行動上書きなし、報酬に触らない、**自動停止を入れない**。
  閾値の事前較正データが無い段階で自動停止を入れるのは、anchor 設計書 §5 が
  「学習応答ゲートを置かない」とした判断（Stage3 v1 の較正ミスの教訓）と同じ理由で不可。
- **eval 経路に漏らさない。** カウンタは trainee client 側のみ。
  介入パラメータではないので `dump_engine_config` への追加は不要だが、
  eval 経路で同じコードパスが走らないことを確認すること。
- **凍結中の run には入れない。** Arm C・Arm K いずれにも入れない
  （1 branch = 1 variable の規律。K の判定を汚さない）。実装は **K 判定後の別ブランチ**。

### 6d. eval-gated checkpoint（設計のみ・GPU 制約が本質的な障害）

「基礎指標が閾値を割ったら checkpoint を採択しない / 巻き戻す」機構は設計にも実装にも
存在しない（`../ops/supervisor_handbook.md` §4c の「checkpoint を物差しに使う前に
steps/best_perf/生成経緯を確認せよ」は人手の注意喚起であって機構ではない）。

**本質的な障害**: 正しい eval は 1v3 argmax・guard ON・自然分布で、GPU を要する。
`CLAUDE.md` の **GPU ワークロードは常に1系統**というルールにより、
訓練中の eval は打てない。したがって選択肢は:

- **G1**: 6c のトリップワイヤ（訓練分布上の proxy、GPU 追加不要）で警報だけ出す
- **G2**: run を区切って eval を挟む（GPU 直列。run 時間が伸びる）
- **G3**: GPU 増設を前提にする（本書の範囲外）

**推奨: G1 まで。** G2/G3 は 0b の運用リソース裁定に属する。

## 7. 実施順（凍結を壊さない）

```
[凍結・変更禁止] Arm C 本走 → eval バッテリー → レンズ4 定性レビュー → C 判定
[凍結・変更禁止] Arm K（実装済み・未発進）→ 発進 preflight → K 発進 → K 判定
                              ↓
       Arm K 発進 preflight に同乗: バックログ4（launcher Cleanup 検証）
                                  + バックログ11（§6a 監視穴埋め）
                              ↓
                        K 判定 → 0b（議題1–5）
                              ↓
       ここから先が本書の実装対象（別ブランチ・1変数ずつ）:
         L1: O1（submit_every）→ O3（ppo_epochs=1 の対照）→ O2（batch 束ね）
         L2: D2（param group 分離）→ D3（残差方策。K の ref 配線の上に建つ）
         L4: 6c トリップワイヤ（観測のみ）
         L3: P1/P2/P3 は 0b 議題3 と併せて裁定
         §5 R1（アンカー置換）: 0b 議題5 の裁定に従う
```

**変えないもの**: Arm C の config・コード、`anchored_ppo_design.md` §6/§6a の判定条件、
Arm K の実装、報酬正典3ストリーム（`reward_design_teacherfree.md`）。

## 8. 未決事項（本書では決めない）

1. **L1 の因果**: L1 の異常が基礎劣化の原因かは未検証
   （`ppo_optimization_health_20260725.md` §5）。O1/O3 の対照実験が要る。
2. **L1 と anchor 系列の順序**: L1 を先に直すと anchor 系列の結果が過去 run と
   比較不能になる。C/K 判定が出るまで L1 は触らない（§7）が、
   **L1 を直した後に C/K を再走する必要があるか**は未決。
3. **O2 導入時の判定条件の単位**: step ではなく消費半荘数へ移すべきか（§2a）。
4. **D3 と Arm K の関係**: K が成功した場合、D3 は上積みか代替かが未決（§3a）。
5. **§5 R1 の採否**: アンカー置換可能性を要件化するか。0b 議題5。
6. **診断 §4 が挙げていない残余仮説**: entropy 係数、value ヘッド初期化、
   行動空間の偏り。`ppo_diag.jsonl` に entropy / explained_variance が無いため
   既存データでは測れず、本書でも未検討。
