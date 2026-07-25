# CLAUDE.md — Mortal フリー雀荘 PPO プロジェクト

## プロジェクト概要

Mortal をフリー雀荘ルール（素点+ウマオカ+チップ、β=1）向けに再設計する調査基盤。
教師データ非依存・自己対戦 PPO。探索ラダー（Stage1〜3）は全段不成立で閉幕し、
DRCA プローブで反鳴き均衡が経済の性質であることを確認。現在は **anchor 系列
（アンカー付き PPO、基礎技能劣化への対策）が優先軸**で、Arm C 本走が凍結中。

**最初に読む文書（この順で）:**
1. `freeparlor/docs/design/ppo_migration_design.md` — PPO 移行の設計正典
2. `freeparlor/docs/design/anchored_ppo_design.md` — **現行 run（anchor 系列）の設計・
   事前登録済み判定条件**（凍結済み）
3. `freeparlor/docs/reports/fundamentals_degradation_diagnosis_20260725.md` +
   `fundamentals_significance_pass_20260725.md` — anchor 系列の動機（基礎技能劣化の診断）
4. `freeparlor/docs/reports/ppo_p3_stage1_result.md` — Stage1 判定結果（立直マキシマリズム）
5. `freeparlor/docs/design/reward_design_teacherfree.md` — 報酬設計の確定事項
6. `freeparlor/docs/ops/project_history.md` — 2026-07-06 以降の時系列経緯（旧「現在の状態」節）
7. `freeparlor/docs/ops/next_steps_2.md` — プロジェクト全体史（初期）
8. `freeparlor/docs/INDEX.md` — docs 全体の索引

## 環境

- 作業は `wsl -d mahjong` のみ。ユーザー `gamba`、リポジトリ `/home/gamba/mahjong/Mortal`
- `conda activate mortal`。学習は `runs/` の spawn ランチャ経由（CUDA fork 回避）
- libriichi 改修後は必ず以下の手順で（2026-07-08 ビルド事故を受けて明文化。
  `supervisor_handbook.md` §4c 参照）:
  1. `conda activate mortal` を明示的に確認済みであること
  2. `PYO3_PYTHON="$CONDA_PREFIX/bin/python"` を明示的にセット
  3. ビルド前に `CARGO_TARGET_DIR` が空（未設定）であることを確認
     （tmux セッション残留由来の汚染がビルド先とcp元の乖離を起こす）
  4. `cargo build --release -p libriichi --lib` →
     `cp -f target/release/libriichi.so mortal/libriichi.so`
  5. cp 後、import スモーク必須:
     `PYTHONPATH=mortal python -c "from libriichi.stat import Stat"`
  （※ 学習発進・eval バッテリーとも preflight_libriichi.sh が自動実行する。
  手動ビルドを信用しない）
- **GPU ワークロードは常に1系統**。学習と eval の同時実行禁止。eval バッテリーも直列実行
- メモリ: WSL 24GB 上限。学習 run は tmux 内で起動（切断耐性）

## ワークフロー規律（違反すると run が無効になる）

### タスク完了の定義
- **commit & push まで完了してタスク**。push 後に
  `git ls-remote origin | grep <branch>` を実行し、リモート先端がタスクの
  commit hash と一致する出力を貼って報告する。push されていない作業は未完了
- ブランチ: PPO 関連の基底は `ppo-migration`。main の DQN 経路は触らない
- **1 branch = 1 variable**: 実験変数を導入する変更は変数ごとにブランチを分ける。
  単一変数アブレーションの規律をブランチ構造で強制する

### 実行環境の分界線
- 実装エージェントのサンドボックス（Claude Code web 等、GPU なし・ローカル WSL 外）での
  検定PASSは参考値。正はローカル WSL（`mahjong` distro）での preflight 全パス。
  GPU 依存作業（run 発進、eval バッテリー）は常にローカル側で実施する
- サンドボックスで実行可能な検証の範囲（2026-07-07 実地確認）:
  - `cargo build --release -p libriichi --lib`: 実行可能（crates.io へのネットワーク
    アクセスあり、ビルド成功）。libriichi の Rust 変更はサンドボックスでビルド確認まで可
  - `python freeparlor/scripts/verify_ppo_p1.py`: 実行不可（numpy/torch 未導入、
    conda 環境なし）。検定PASSの確認は必ずローカル WSL 側で行う

### run の規約
- run dir は日時 suffix 必須（例 `stage1_20260705_053301`）。**再利用禁止**
- 中止した run は削除せず `aborted<N>_` として保全（証拠保存）
- 発進前 preflight: 残党チェック（`pkill` + `ss -tlnp | grep 5000`）+
  libriichi rebuild + 全検定（`freeparlor/scripts/verify_ppo_p1.py`）PASS。
  **検定の本数はスクリプトの実行結果（`ALL N CHECKS PASSED`）が正**。
  本書に本数をハードコードしない（2026-07-25 時点 20 本。19/20 本目が anchor 系列の
  Arm C / Arm K assert — `anchored_ppo_design.md` §7）
- 発進前 preflight は**実装完了報告を受け取った後にも**残党チェックを行う
  （報告に現れない孤児プロセスが GPU を掴んでいた事例あり — `project_history.md` 2026-07-25）
- 発進後: 開始報告（config 全文 + 監視項目 + step 100 到達 + alive clients 3/3）
- **凍結ルール**: 事前登録した run は判定窓が閉じるまでコード・config 変更禁止。
  例外はクラッシュとデータ整合性の破れのみ。「気になる挙動」は記録して続行

### 監視期待値（1件でも非ゼロなら報告）
- `trajectory step count mismatch` = 0（必須）
- `illegal_action_fallback_count` = 0（必須）
- `online chip resolution failed` = 0（必須）
- `loader size delta` = INFO（非致命・報告のみ）
- alive clients = 3/3、step 到達性（停滞は異常）

### 実装の禁則
- **サイレント修正・サイレントフォールバック禁止**。解決不能は例外で大声で落とすか、
  カウンタ+WARNING で可視化する（zeros 埋め・黙って skip は過去に run を3本殺した）
- **訓練 rollout への行動上書き禁止**（rule-based guard 等）。eval は本家準拠で guard ON
- 訓練 client は必ず π からの純サンプリング（greedy/top_p 混合禁止）
- 新規 Rust 表面積は最小に。本家挙動に戻せるならリバートを優先
- 検定・診断の self-play client は**本番 client と同一構成**（構成 dump diff==空）
- **訓練側の介入は eval 経路に漏らさない**。介入パラメータは engine の構成 dump に含め、
  eval 側で 0（無効）であることを検定で assert する。現行の対象は
  `p_enrich`（Stage2）/ `call_bonus_b`（Stage3）/ `anchor_prob`・`kl_beta`（anchor 系列）
  で、いずれも検定(17d) に同居（`stage2_design.md` §2 / `anchored_ppo_design.md` §7）

### 実験の規律
- 単一変数アブレーション優先。GPU を焼く前に設計文書を commit
- 判定条件は run 前に固定し、結果を見てから変更しない（post-hoc goalpost 禁止）
- 挙動の評価は2レンズ併記: argmax eval（配備挙動）と sampled action_mass（学習方向）。
  Stage2 以降は分布にも注意: 訓練測定は濃縮分布上、eval は常に自然分布
  （絶対値の run 跨ぎ比較は不可、倍率同士で比較 — `stage2_design.md` §4）
- 400 step 級スモークで挙動の結論を出さない（分散が支配する。配管検証のみ）
- **定性レビュー（レンズ4）を判定より前に通す**: run 完走 → eval バッテリー（レンズ1–3）→
  **Gamba による牌譜の人手レビュー** → 判定の起草、の順を守る。判定非関与だが、所見は必ず
  定量指標に突き合わせ、乖離があれば診断タスクを起票する（手順は
  `freeparlor/docs/ops/qualitative_review_protocol.md`）。GPU 不要のため後回しにしない
- **基本指標を「表に載せる」で終わらせない**: eval の各レンズで基本指標
  （agari / houjuu / fuuro / riichi / ryukyoku / avg_rank）を **init 基準との差分**として
  評価し、半荘クラスタ SE を付けて有意性まで出す（`analyze_fundamentals_1v3.py`）。
  劣化が有意なら判定文書の**結論部**に明示する。目的指標（チップ・鳴き）だけで損益を語らない。
  ※ Stage1–3 では基本指標は毎回 eval で出力され結果 md の表にも載っていたが、
  init との差分の有意性を取らなかったため、放銃劣化（z +3.9〜+4.4、全 stage 有意）の
  検出が最大2週間遅れた（データは 2026-07-08 時点で揃っていた）

## 現在の状態（2026-07-25 時点）

> **この節は現在進行中・未決の事項だけを載せる。** 完了済みの経緯（探索ラダー Stage1〜3、
> DRCA プローブ、過去のインシデント史）は `freeparlor/docs/ops/project_history.md` へ
> 不改変で移設した。正は各設計書・判定レポートと git log。
> **状態を変えるタスクを完了したら、この節の更新も同一 commit に含めること**
> （current でなくなった項目は削除ではなく `project_history.md` の末尾へ移す）。

### 現在地（要約）

- **進行中**: anchor 系列 **Arm C 本走**（`anchor_c_20260725_164756`、step 16000 まで凍結中）。
  基礎技能劣化への対策として凍結 init を opponent pool に常駐させる単一変数アブレーション
- **中断**: DRCA プローブ本測定（実効5枠のうち2枠のみ完了、§5a-1c で打ち切り裁定）
- **閉幕**: 探索ラダー Stage1〜3 は全段不成立（本質的機会費用仮説を支持）
- **実装エージェント**: Cursor Composer / Claude Code
  （いずれもローカル WSL の GPU・conda 環境・tmux に直接アクセス可）

### 進行中: anchor Arm C 本走（2026-07-25 17:03:10 JST 発進・凍結中）

- run dir `/home/gamba/mahjong/runs/ppo/anchor_c_20260725_164756`、tmux セッション
  `ppo_anchor_c_20260725_164756`、GPU = RTX 5060。init = beta1_huber_192x40
- config は `freeparlor/configs/ppo_anchor_c.toml`（プレースホルダを launcher が run パスへ
  in-place 解決。stage1 config との diff は run パス + `[opponent_pool]` の
  `anchor_prob=0.25` / `anchor_checkpoint`(=init と同一パス) のみ。`kl_beta` はキー不在＝
  設計された OFF）
- 発進前 preflight（launcher が自動実行）: 残党チェック・port5000 clear・libriichi rebuild +
  import smoke・**`verify_ppo_p1.py` 全20検定 PASS**（所要 ~15分）
- **機械ゲート（@step200）通過**: anchor 採択率 **0.2393**（67/280 draw、期待 0.25±0.05）、
  anchor が返す checkpoint は全て init で一致。opponent pool engine 構成 dump =
  `anchor_prob:0.25 / p_enrich:0.0 / call_bonus_b:0.0 / kl_beta:0.0 / eval_mode:False`
- step 219 時点で監視4項目（mismatch / illegal_action_fallback / chip 解決失敗 /
  trainer NaN）全て 0、alive clients 3/3
- **凍結宣言済み: step 16000 完走までコード・config 変更禁止**（例外はクラッシュと
  データ整合性の破れのみ）。**判定窓は step 8000–16000**（`anchored_ppo_design.md` §6:
  放銃差 z<2 + チップ +方向≥1SE、1v3 両脚 n=800）。完走・eval・判定は別タスク
- 発進試行1・2回目は preflight で FATAL 停止（非対話シェル由来で tmux に conda / cargo が
  未継承。訓練開始前・データ生成なし）。run dir は規約どおり
  `aborted1_anchor_c_20260725_164338` / `aborted2_anchor_c_20260725_164551` として保全

### 次: anchor Arm K（実装・CPU 検証完了・**未発進**）

- `ppo_loss` への masked full KL（`kl_beta=0.1`、pool 不変・anneal なし = 恒久レギュラライザ）。
  差し戻し修正まで完了済み（NaN 勾配・検定(20) 強化・§5-a1 ゲート改修・pool_draw 競合）
- **発進前 preflight に `verify_ppo_p1.py` 全20本 + 400-step 配管スモークを統合すること**
  （GPU 1系統ルールにより Arm C 走行中は実施不可のため持ち越し中）
- 判定後、K のみ最大1回の機械的再走が許容（過強→β/4、過弱→β×4。`anchored_ppo_design.md` §6a）

### 中断中: DRCA プローブ本測定（§5a-1c 打ち切り裁定・2026-07-25）

- 完了した実効枠は2つ: **第1枠**（セット(a)×Stage1-16000）ΔQ̄ = **−3.2635** 千点 /
  cluster SE 0.7208 / **4.53SE**、**第2枠**（セット(a)×init）ΔQ̄ = **−2.1353** /
  SE 0.4788 / **4.46SE**。中核の問い（反鳴き均衡は基礎劣化アーティファクトか経済の性質か）は
  この挟み撃ちで決着（= 経済の性質）
- **第3枠**（セット(a)×Stage3-16000）は進捗 4.6%（359/7760 rollout）で**中断**。
  成果物は `backups/drca/main_a_s3final_20260725_120218_20260725_163836.tar.gz` に退避、
  run dir 内 `run_resume.sh` で再開可能。残枠（同枠の残り・a_s3mid・b_s3final）は**未測定**
- 放棄した内容は `drca_probe_design.md` §5a-1c に明記（主 contrast 2 全損 → 解釈 C は
  判定せず、contrast 3 測定不能、§4 の A/B/C/D/N 割り当てに到達しない）。
  **最終報告では未測定枠の存在を必ず明示する**（選択的報告の回避）
- 再開可否は改めて裁定。現時点の優先度は anchor 系列より下

### 判断の土台（確定済み・詳細は `project_history.md` と各判定レポート）

- **探索ラダー全段不成立**: Stage1 純探索（立直マキシマリズムを発見、赤鳴きは沈む）→
  Stage2 分布介入（配備税を新たに発見）→ Stage3 報酬介入（anneal 内蔵で配備税ゼロ）の
  いずれでも赤鳴きの経済的合理性は覆らず、**本質的機会費用仮説を支持**して閉幕
- **反鳴き均衡は基礎劣化アーティファクトではない**（上記 DRCA 2枠）
- **基礎技能の劣化は有意**: 放銃劣化 z +3.9〜+4.4・和了劣化 −2.1〜−3.6（全 stage）、
  avg_rank 悪化が有意なのは Stage2 のみ（`fundamentals_significance_pass_20260725.md`）。
  これが anchor 系列の動機
- **運用の教訓**: 発進前の残党チェックは**実装完了報告の後にも必須**
  （2026-07-25、報告に現れない孤児 verify プロセスが GPU 1系統ルールを破っていた）。
  run 成果物の保持・清掃は `run_artifact_retention.md` の許可リスト方式で、毎回 Gamba 承認

## 残タスク（バックログ）

消化済み項目（1 / 1b / 2 / 6 / 7 / 8、および 0・9 の完了分）は
`freeparlor/docs/ops/project_history.md` の付録に原文で保全。以下は未決のみ。

0. **DRCA プローブ（診断、事前登録済み）**: 本測定は §5a-1c で**打ち切り・中断**
   （上記「現在の状態」節）。残枠 a_s3final の残り / a_s3mid / b_s3final は未測定。
   再開可否は anchor 系列の帰結と併せて改めて裁定する
0b. **探索ラダー閉幕後の方針設計セッション**（設計監督側）: 立直マキシマリズムの商用採否 /
   経済定数変更（新実験系）/ 敵対的搾取者訓練の要否を裁定。事前フレームは
   `freeparlor/docs/ops/policy_session_0b_frame.md`（DRAFT・裁定非関与）、設計ギャップの
   技術検討は `freeparlor/docs/design/product_gaps_design_notes.md`（DRAFT）。
   残り: **商品性要件の Gamba ヒアリング** → セッション実施
3. **Stage2b（解凍実験）の再評価**: 配備税の発見により「収束済み方策の分布シフト適応」の
   商用価値が上がった。実施判断は 0b と併せて検討
4. **launcher Cleanup 修正の run-validation**（実装・logic 検証は 2026-07-25 bc80aff で完了）:
   end-to-end 確認は次回の実訓練発進 preflight に持ち越し中 → **Arm K 発進時が該当**
5. **メタ系ハーネスのミラー較正 RUN**（実装・GPU 非依存検証は 2026-07-25 4d16538 で完了）:
   実測 RUN は GPU が空いたときに実施
9. **anchor 系列（現行の優先軸）**: 設計凍結済み（`anchored_ppo_design.md`）。実装・
   Arm C 発進まで消化済み。残り: **C 完走（step16000）→ C の eval バッテリー**
   （argmax 6ckpt + 1v3 n=800 + ミラー較正脚 + メタ対決）**→ レンズ4 定性レビュー**
   （`qualitative_review_protocol.md`、判定より前）**→ C 判定 → K の発進 preflight**
   （verify 全20本 + 400-step 配管スモーク）**→ K 発進 → K 判定 → 0b 接続**
10. **「鳴き後のツモ順ずれ」を抑える山操作 probe（DRCA-v2 候補・着手時期未定）**:
   鳴くと自摸順がずれ、固定された山との噛み合わせで両腕が乖離する。frame1 の極値レビューで
   これが裾の主機構と同定された（`qualitative_expert_review_drca_frame1_20260722.md` §4b の
   分類①③）。実施可否と時期の判断材料:
   - **既存枠への amendment は不可**。現行 DRCA の凍結設計は、この乖離を「鳴きの帰結の一部」
     として**意図的に含めている**（`drca_probe_design.md` の理論的位置づけ）。機構を変えると
     推定対象が変わり、測定済みの第1/2枠と比較不能になる。やるなら別 probe として新規に事前登録
   - **集計 ΔQ̄ には不要**。ツモ順ずれは腕内の共通ショックで K=8 では平均消えしないが、
     N=485 の分岐点間では消える。効くのは**分岐点単位の推定**＝定性ドリルダウンや
     「取りこぼし仮説（選択的な鳴きの価値）」のような、ケース単位の問いを立てるとき
   - **コストと解釈の制約**: libriichi の山操作が要り、現行 DRCA の「Rust 表面積ゼロ」方針を破る。
     また自摸順を揃えた盤面は物理的に非麻雀なので、得られるのは「鳴きの価値のうち手牌変換
     由来の成分」という**分解**であって、配備上の意思決定価値そのものではない
   - **着手条件**: DRCA を再開し、かつケース単位の問いを立てるとき。anchor 系列の帰結が出るまで保留

## 役割分担

設計判断・レビュー・仮説の裁定は Claude（chat 側・監督）が担当。このリポジトリでの
あなた（実装エージェント = Cursor Composer / Claude Code 等。現在どれかは
「現在の状態」節を参照）の役割は実装・検証・commit・push。
設計変更が必要だと感じたら、実装せずに提案として報告する。
