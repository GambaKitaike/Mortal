# freeparlor/docs/ 索引

初版: 2026-07-10（design / reports / ops / archive/dqn への再編時）。
**最終更新: 2026-07-28**（anchor Arm C/K の判定・レンズ4・checkpoint 軌跡・進路事前登録・
セッション申し送りを追加。L1 最適化衛生の診断と「壊れにくい自己学習PPO」設計ノートを追加）。
日付は文書本文が名乗る日付（= 内容の基準日）。ステータスは判断根拠が明確なもののみ厳密で、
曖昧なものは本文参照を推奨。

- **active** = 現在も参照される生きた文書 / **frozen** = 事前登録により変更禁止（run 進行中）
- **closed** = 役目を終えた（判定完了・タスク消化済み） / **DRAFT** = 未凍結・裁定前

## design/ — 設計・pre-registration（10本）

| パス | 日付 | ステータス | 要約 |
|---|---|---|---|
| `design/anchored_ppo_design.md` | 2026-07-25 | **frozen（進行中）** | **現行軸**。アンカー付きPPO（基礎劣化対策）の単一変数2 arm — C（opponent pool へ凍結init を anchor_prob=0.25 で常駐）/ K（`ppo_loss` に masked full KL、kl_beta=0.1）。凍結commit 847dc8d が事前登録。判定条件は §6、K の再走規定は §6a、発進ゲートは §5-a1 amendment 済み。 |
| `design/early_damage_probe_design.md` | 2026-07-28 | **事前登録（診断・判定非関与・未発進）** | step 0–2000 の内部形状を測る短 run（2000 step ≈ 3.1h）。軌跡測定が残した唯一の宿題（この区間に checkpoint が無い）を埋める。**単一変数 = 観測専用の `diag_save_every`**。`save_every` を下げる素朴案は `OpponentPool` の glob 対象を変えて2変数になるため不可（§2a）。判定条件は置かない。発進可否は Gamba 裁定。 |
| `design/ppo_migration_design.md` | 2026-07-02 | active | PPO移行の設計正典。教師データ非依存本線の実装設計（critic scale・希少性探索の分岐を含む）。 |
| `design/reward_design_teacherfree.md` | 2026-07-01 | active | 教師データ非依存の報酬設計（あ）確定版。reward_audit を受けた本線設計。 |
| `design/drca_probe_design.md` | 2026-07-12 DRAFT → 07-13 凍結 | frozen（測定は打ち切り） | DRCAプローブ（duplicate rollout による鳴き反実仮想アドバンテージの直接測定）の設計・解釈条件。§5a が事前登録、§5a-1a 規模確定（K=8/N=485）、§5a-1b 48h条項（実効5枠へ削減）、**§5a-1c 打ち切り裁定（2026-07-25、第3枠中断・残枠未測定・主contrast2は評価不能）**。 |
| `design/stage3_design.md` | 2026-07-11 | closed | Stage3（anneal付きper-decision鳴きボーナス）の設計・事前登録済み判定条件。判定完了（2026-07-13、分岐2成立 → `reports/ppo_p3_stage3_result.md`）。発進ゲートは v1→v2 amendment 済み（§3）。 |
| `design/stage2_design.md` | 2026-07-06 | closed | Stage2（配牌rejection samplingによる赤濃縮）の設計・事前登録済み判定条件。判定完了（2026-07-11、分岐2成立 → `reports/ppo_p3_stage2_result.md`）。 |
| `design/product_gaps_design_notes.md` | 2026-07-24 | DRAFT | 商品化設計ギャップ G1–G3 の技術検討メモ（`ops/policy_session_0b_frame.md` §5.3 の材料）。裁定非関与・実装未承認。 |
| `design/teacherfree_training_candidates.md` | 2026-07-25（07-26 取り込み） | DRAFT | 教師データ非依存の訓練方式 候補棚卸し。**問題 I（cold start = 天鳳教師フェーズの代替）と 問題 II（均衡脱出）を分離**して候補5本を整理。(1) シミュレータ由来の自己教師あり補助タスク（問題 I の最有力・実体は oracle 蒸留。**`robust_selfplay_ppo_design.md` §5a の A1 と実質同一**）/ (2) リーグ訓練・敵対的搾取者（最も安い・0b 議題3）/ (3) R-NaD 系（**不採用寄り** — 理論保証が4人戦で消える）/ (4) 信念状態つき探索（本丸・最も高い）/ (5) HITL。**事前登録ではない・判定条件を含まない**。 |
| `design/robust_selfplay_ppo_design.md` | 2026-07-25（07-28 改訂） | DRAFT | 「壊れにくい自己学習PPO」の設計ノート。壊れにくさを4層（L1最適化衛生 / L2参照点 / L3相手分布 / L4運用）に分解し、anchor系列がL2/L3をカバーする一方 **L1とL4が空白**であることを確定。L1の一次証拠は `reports/ppo_optimization_health_20260725.md`。L2「差分だけ学習」の案D1–D4（推奨D3=残差方策、ArmK の ref配線を再利用）、L4 トリップワイヤ、§5 アンカー置換可能性（0b議題5）。**§5b/§5c（07-28 追記）: anchor 系列は商用化で無駄にならない（機構の知見と配管は持ち越せる／アンカーは config パス指定で差し替え可能）が、K の部分的保護は「強い人間譜由来の参照点」で得たものであり、牌譜非依存アンカーで同等以上が出るかは別の実験＝16k run 1本の未計上コスト**。**裁定非関与・実装未承認・判定条件は書かない**。 |
| `design/reward_audit_teacherfree.md` | 2026-07-01 | closed | `RewardCalculator.calc_delta_blend` の棚卸し（教師データ非依存化に向けた監査、reward_design の前段）。 |

## reports/ — 判定結果・run記録・診断（19本）

> **移動時の制約（2026-07-25 追記）:** `ppo_p1_verify_log.txt` は `verify_ppo_p1.py` が
> 出力先をハードコードしているため移動不可。`fundamentals_degradation_diagnosis_20260725.md` と
> `drca_frame2_a_init_aggregate_20260725.md` は**凍結済みの** `design/anchored_ppo_design.md` が
> `../reports/` の相対パスで根拠文書として参照しているため、anchor 系列の判定が終わるまで移動不可。
> それ以外の相互参照はすべてバッククォート付きファイル名のみで、移動してもリンクは壊れない。

### 現行軸（基礎劣化 → anchor 系列）

| パス | 日付 | ステータス | 要約 |
|---|---|---|---|
| `reports/fundamentals_degradation_diagnosis_20260725.md` | 2026-07-25 | active | 「PPOが与えられた基礎（牌理・降り）を壊している」の診断。anchor系列の動機。 |
| `reports/fundamentals_significance_pass_20260725.md` | 2026-07-25 | active | 上記 §3(a) への半荘クラスタSE付与。放銃劣化は全stage有意（z +3.9〜+4.4）、和了劣化も有意（−2.1〜−3.6）、avg_rank悪化が有意なのはStage2のみ。 |
| `reports/ppo_optimization_health_20260725.md` | 2026-07-25 | active | **判定非関与の診断**。診断 §4 が仮説に挙げていないL1層（最適化衛生）の横断実測（8 run）。`minibatch_size=512` は全run全バッチで不発（1 optimizer step = 1半荘の full-batch、median 168–179）/ バッチ到着時点で既に clip_fraction ≈0.20–0.33（陽性対照 step0 は 0.0000、定常staleness は 50–100 step）/ 4 epochs が trust region をほぼ動かさない（e1→e4 = −0.0008〜−0.0032）。因果は主張しない。 |
| `reports/anchor_arm_c_result.md` | 2026-07-27 | closed | **anchor Arm C 判定**（opponent pool へ凍結 init を常駐）。事前登録条件 §6 への機械的照合の結果 **象限 IV（判定1✗ 判定2✗）= 不成立**。放銃 11.85%→15.45%（z=+6.57）。進路の正は `ops/anchor_c_route_decision_20260726.md`。 |
| `reports/anchor_arm_k_result.md` | 2026-07-28 | closed | **anchor Arm K 判定**（`ppo_loss` に凍結 init への masked full KL）。**象限 III（判定1✗ 判定2○）= 買ったが払った**。放銃 11.85%→13.16%（z=+2.51、**C の劣化幅の約1/3**）、チップ +0.508（+2.18SE）。**C と結果が分かれ、引き戻しは pool より損失側が効くことを単一変数で確定**。§6a の kl_beta×4 再走は許容だが未実施。 |
| `reports/anchor_checkpoint_trajectory_20260728.md` | 2026-07-28 | active | **判定非関与の探索的診断**。C/K の中間 checkpoint を判定と同一条件（n=800）で 1v3 測定。**損傷もチップ獲得も最初の 2000 step（全体の 12.5%）でほぼ完了**しており、残り 14000 step は「維持（K）か喪失（C）か」の期間。**KL アンカーの効能は初期劣化の防止ではなくドリフトの停止**。判定1・判定2 を両立する中間 checkpoint は存在しない。 |
| `reports/policy_quality_metrics_20260728.md` | 2026-07-28 | active | **判定非関与の探索的診断**。レンズ4 起票の3指標（鳴きの質 / 牌効率 / 方策の一貫性）を判定と同一の 1v3 牌譜 n=800 で測定。**鳴き判断は両腕とも双方向に劣化**（テンパイ機会の見送り K z=+7.21 / C z=+13.17、取った鳴きのテンパイ率 K z=−2.01 / C z=−3.96）。一方 **牌効率と一貫性は Arm C だけが壊れ Arm K は n.s.**（向聴を外した率 C z=+22.27 / K −0.65、切り順逆転率 C z=+5.69 / K +1.38）＝ 判定と独立な測定が C/K の分岐を再現した。土台の向聴計算器は libriichi と **2,259,173 決定点で不一致ゼロ**。 |
| `reports/qualitative_review_anchor_c_20260727.md` | 2026-07-27 | exploratory | Arm C のレンズ4（Gamba・1半荘、自己対戦 step16000）。**平均和了打点 +1137.8点(z=+7.72) は全層 n.s. の構成シフト由来**（立直和了 54.3%→92.7% / ダマ 21.3%→0.8%）＝ Simpson 型の合成効果で、**成果として引用してはならない**。加カンが実質消滅。 |
| `reports/qualitative_review_anchor_k_20260728.md` | 2026-07-28 | exploratory | Arm K のレンズ4（Gamba・1半荘、1v3）。「かなり良い。**論外な打ち方が全く無かった**」＝ C の「見るに堪えない」と対照的で定量と同方向。残る所見は七対子決め打ち・打点構築の見送り・鳴き機会の取りこぼし。打点上昇は C 同様に構成シフト由来で全層 n.s.。 |

### DRCA プローブ

| パス | 日付 | ステータス | 要約 |
|---|---|---|---|
| `reports/drca_frame2_a_init_aggregate_20260725.md` | 2026-07-25 | closed | 第2枠（セット(a)×init）単枠集計: ΔQ̄=−2.1353 / SE 0.4788 / 4.46SE。集計キー衝突（champion slot間の3-tuple併合）の特定と4-tuple化修正、第1枠回帰確認を含む。 |
| `reports/qualitative_expert_review_drca_frame1_20260722.md` | 2026-07-22 | exploratory | 第1枠 ΔQ̂ 極値24件のGamba全件目視評価 + 監督側訂正§4（裾支配・主判定の推定対象は「無差別な鳴きの平均」・同一山ゆえ結果論はK=8で消えない）。 |
| `reports/drca_pilot_qualitative_notes.md` | 2026-07-16 | exploratory | パイロット50分岐点の定性ドリルダウン（負の極値=手壊しコスト型 / 正の極値=好機・防御的速度鳴き）+ 牌譜HTMLビューア生成。判定非関与。 |

### 探索ラダー（Stage1〜3、閉幕）

| パス | 日付 | ステータス | 要約 |
|---|---|---|---|
| `reports/ppo_p3_stage1_result.md` | 2026-07-06 | closed | Stage1判定結果（立直マキシマリズム、事前固定条件成立→Stage2移行確定）。§6 argmax evalバッテリー・§7 grp_baseline 1v3 を含む。 |
| `reports/ppo_p3_stage2_result.md` | 2026-07-10（判定 07-11） | closed | Stage2 evalバッテリー+判定結果（分岐2成立、機会費用仮説支持・**配備税**の発見 §7c → Stage3解封）。 |
| `reports/ppo_p3_stage3_result.md` | 2026-07-13 | closed | Stage3 evalバッテリー+判定結果（分岐2成立、slope/SE=−21で減衰。正典反鳴き勾配の再現性 §7a・anneal内蔵で配備税ゼロ §7c → **探索ラダー閉幕** §9）。 |
| `reports/ppo_p3_stage1.md` | 2026-07-04 | closed | Stage1本走のrun状態・インシデント史。 |
| `reports/ppo_p3_pause_resume.md` | 2026-07-05 | closed | run #7のpause/resume記録。 |
| `reports/qualitative_expert_review_20260715.md` | 2026-07-15 | exploratory | Stage1-16000 argmax自己対戦の専門家（Gamba）定性レビュー。本プロジェクト初の「絶対的な強さ」評価。 |
| `reports/qualitative_expert_review_stage3_20260716.md` | 2026-07-16 | exploratory | Stage3-16000 argmax の定性レビュー（鳴きレパートリー増だが鳴き後が未熟・カン判断異常・降りの規律崩壊）。上記の対。 |

### 生成物

| パス | 日付 | ステータス | 要約 |
|---|---|---|---|
| `reports/ppo_p1_verify_log.txt` | 随時更新 | 生成物 | `verify_ppo_p1.py` の最新実行ログ（発進preflightのたびに上書きされる）。**移動禁止** — 出力先が `verify_ppo_p1.py` にハードコードされている。 |

## archive/ppo_p1p2/ — PPO 配管期の完結文書（5本、全て closed）

2026-07-11（DRCA 実装期）以降は参照されていない、P1（配管）・P2（スモーク／プローブ）期の記録。
本走（Stage1〜3）の判定には使わない。

| パス | 日付 | 要約 |
|---|---|---|
| `archive/ppo_p1p2/ppo_p1_plumbing.md` | 2026-07-02 | PPO P1配管の実装サマリ。 |
| `archive/ppo_p1p2/ppo_p2_smoke.md` | 2026-07-02 | PPO P2スモーク結果（OOM対策後更新）。 |
| `archive/ppo_p1p2/ppo_p2_diag.md` | 2026-07-02 | PPO P2 OOM対策後の診断再走結果。 |
| `archive/ppo_p1p2/ppo_p2b_lr_probe.md` | 2026-07-03 | PPO P2b lrプローブ（fuuro崩壊のlr要因検証）。 |
| `archive/ppo_p1p2/ppo_p2c_advantage_decomp.md` | 2026-07-03 | PPO P2c 宣言行動（鳴き・立直）のadvantage分解計装。 |

## ops/ — 運用文書（12本）

| パス | 日付 | ステータス | 要約 |
|---|---|---|---|
| `ops/supervisor_handbook.md` | 2026-07-10（07-25 改訂） | active | 設計監督Claude向け引き継ぎハンドブック（出力規約・ワークフロー鉄則 §3・確定知見・殺した仮説リスト §4）。設計相談セッションの冒頭で必読。§5 は現況の二重管理をやめ CLAUDE.md へのポインタに変更済み。 |
| `ops/qualitative_review_protocol.md` | 2026-07-25 | active | **定性レビュー（レンズ4）の実施プロトコル**。run 完走 → eval バッテリー → **レンズ4 → 判定**の順を規定。判定非関与だが、所見を定量指標に突き合わせ乖離があれば診断タスクを起票することを必須化（07-16 の所見が 07-25 の診断まで9日遅れた事故の再発防止）。 |
| `ops/project_history.md` | 2026-07-25 | active（履歴） | **CLAUDE.md「現在の状態」節から移設した時系列経緯**（2026-07-06 Stage1判定 〜 2026-07-25 anchor Arm C 発進）+ バックログ原文。不改変保全・追記のみ。 |
| `ops/run_artifact_retention.md` | 2026-07-16 | active | run成果物の保持・清掃運用（成果物クラス別ポリシー・イベント駆動トリガ・許可リスト方式の削除手順）。2026-07-09 ディスク枯渇インシデントの再発防止。 |
| `ops/policy_session_0b_frame.md` | 2026-07-16（07-24 §5 / 07-25 §6 追記） | DRAFT | 探索ラダー閉幕後の方針設計セッション（バックログ0b）の事前フレーム。議題×DRCA帰結の分岐シナリオと不足材料リスト。裁定非関与。**議題は 4+1 に拡張（2026-07-25、§6 = 議題5 ライセンス・データ権利の分界。牌譜由来 init への依存を製品にどう持ち込むか + 診断 §6 B の前提変更）**。 |
| `ops/session_handover_20260728.md` | 2026-07-28 | active | **次セッション（やるべきことの洗い出しと優先順位付け）のための材料整理**。anchor 系列の完了内容と未決事項。裁定は行わない。**優先順位付けの正はこれ**。 |
| `ops/anchor_c_route_decision_20260726.md` | 2026-07-26 | closed | **Arm C 判定後の進路の事前登録**（結果を見る前に確定 = post-hoc goalpost 禁止の実践）。判定条件は変更せず、4象限それぞれで次に何をするかの資源配分のみ決める。※ **C=IV かつ K=III の組み合わせは明示的に扱っていない**ため次の軸の選択は 0b の裁定事項。 |
| `ops/next_steps_2.md` | 2026-06-29 | active（歴史） | プロジェクト全体史・引き継ぎメモ（最終目標・初期の現状まとめ）。 |
| `ops/anchor_impl_task_20260725.md` | 2026-07-25 | closed | anchor系列 Arm C/K の実装タスクプロンプト（実装エージェント宛）。実装完了により消化。 |
| `ops/anchor_impl_rework_20260725.md` | 2026-07-25 | closed | Arm K 差し戻しプロンプト（NaN勾配・検定(20)強化・§5-a1ゲート改修・pool_draw競合）。修正完了により消化。 |
| `ops/prompts_for_20260713.md` | 2026-07-08 | closed | 2026-07-13投入用プロンプト集（①Stage1残タスク ②Stage2実装 ③Stage2発進）。①〜③すべて消化済み。歴史文書として不改変保全。 |
| `ops/next_steps.md` | 2026-06-23 | closed | 旧引き継ぎメモ。`next_steps_2.md` に事実上置換済み。 |

## archive/dqn/ — オフラインDQN時代の完結文書（36本、全て closed）

日付は全て 2026-06-22〜2026-07-02。教師データ由来のオフラインDQN+CQL経路（現行PPO本線への移行前）の調査・診断・phase結果。`main` ブランチのDQN経路にのみ関連し、現行 `ppo-migration` の判断には使わない。

| パス | 日付 | 要約 |
|---|---|---|
| `archive/dqn/dqn_era_readme.md` | 2026-06（2026-07-25 移設） | **当時の `freeparlor/README.md`**（Phase1〜4のポートフォリオ向けまとめ）。移設時にアーカイブ注記の追加と相対リンク修正のみ実施、本文は不改変。 |
| `archive/dqn/phase1_result.md` | 2026-06-23 | Phase1 Result: Reproducible 64×10 Run。 |
| `archive/dqn/phase1_stats_192x40.md` | 2026-06-23 | Phase1 Playstyle Stats: 192×40 Self-Play。 |
| `archive/dqn/libriichi_agari_survey.md` | 2026-06-23 | libriichi和了情報調査（チップ報酬β向け）。 |
| `archive/dqn/phase2_result.md` | 2026-06-23 | Phase2 Result: Free-Parlor Reward（64×10 Connectivity）。 |
| `archive/dqn/phase3_sweep.md` | 2026-06-23 | Phase3 Result: α:γ Ratio Sweep（192×40）。 |
| `archive/dqn/phase4_aka_conditional.md` | 2026-06-23 | Phase4: 赤ドラ条件別打牌分析。 |
| `archive/dqn/phase4_chip.md` | 2026-06-23 | Phase4 Result: Chip Reward β。 |
| `archive/dqn/phase4c_human_aka_conditional.md` | 2026-06-23 | Phase4c: 人間データの赤ドラ条件別打牌分析（仮説C検証）。 |
| `archive/dqn/phase4d_aka_opp_probe.md` | 2026-06-24 | Phase4d: 赤取りこぼし損失プローブ（実装・サニティ）。 |
| `archive/dqn/phase4d_chip_realize.md` | 2026-06-25 | Phase4d: 赤保持→チップ実現。 |
| `archive/dqn/phase4d_sweep_results.md` | 2026-06-25 | Phase4d: lambda_oppスイープ結果。 |
| `archive/dqn/online_r_chip_layer1.md` | 2026-06-25 | Online TDチップ報酬 層1: arena horaへのchip_delta埋め込み。 |
| `archive/dqn/online_r_chip_layer2.md` | 2026-06-25 | Online TDチップ報酬 層2: dataloader TDトランジション。 |
| `archive/dqn/online_r_chip_layer3.md` | 2026-06-25 | Online TDチップ報酬 層3: Q_chipヘッド+target net+n-step TD。 |
| `archive/dqn/online_replay_buffer.md` | 2026-06-25 | Onlineリプレイバッファ（データ生成・drain・ログ形式）。 |
| `archive/dqn/online_throughput_test.md` | 2026-06-25 | Online 3プロセス疎通+スループット計測。 |
| `archive/dqn/online_throughput_parallel.md` | 2026-06-25 | Online生成律速緩和（client並列化再計測）。 |
| `archive/dqn/online_main_report.md` | 2026-06-26 | Q_chip Online本番学習のセットアップ・中間報告。 |
| `archive/dqn/online_main_progress.md` | 2026-06-26 | Q_chip Online本番学習の進捗報告（step〜24800時点）。 |
| `archive/dqn/online_diag_a_lambda_opp_zero.md` | 2026-06-27 | 診断A: lambda_opp=0のonline影響切り分け。 |
| `archive/dqn/online_diag_b_cql_weak.md` | 2026-06-27 | 診断B: online弱CQL導入の副露影響切り分け。 |
| `archive/dqn/online_diag_fuuro_summary.md` | 2026-06-27 | online無差別鳴き診断A/B総合報告。 |
| `archive/dqn/online_fuuro50_aka_selectivity.md` | 2026-06-27 | 副露率50%の正体（赤選択性の切り分け集計）。 |
| `archive/dqn/online_cql_min_q_weight_sweep.md` | 2026-06-29 | online CQL min_q_weightスイープ×step6000。 |
| `archive/dqn/kyoku_length_dist.md` | 2026-06-29 | 局長（trainee move数/局）分布。 |
| `archive/dqn/aka_call_hora_count.md` | 2026-06-30 | 赤を活かした鳴き和了（target正例）の希少性調査。 |
| `archive/dqn/call_channel_diag.md` | 2026-06-30 | 鳴き和了チャネル診断。 |
| `archive/dqn/online_chip_nstep40_mc.md` | 2026-06-30 | online chip_n_step=40純MCアブレーション。 |
| `archive/dqn/mqw03_call_channel.md` | 2026-06-30 | mqw03副露和了率（Part A）。 |
| `archive/dqn/beta1_huber_verify.md` | 2026-07-01 | β=1 Huber損失検証（スケール仮説）。 |
| `archive/dqn/beta1_pnl_salvage.md` | 2026-07-01 | β=1収支/avg_rankサルベージ調査。 |
| `archive/dqn/fuuro_jump_archaeology.md` | 2026-07-01 | 副露率ジャンプ考古学（β=1激減後の変更点調査）。 |
| `archive/dqn/mqw03_collapse_diag.md` | 2026-07-01 | mqw03副露和了崩落診断（step-wise B1+B2）。 |
| `archive/dqn/mqw03_cql_qshift.md` | 2026-07-01 | mqw03 CQL鳴きQ押し下げ検証。 |
| `archive/dqn/beta1_huber_192x40_verify.md` | 2026-07-02 | β=1 Huber 192×40の交絡排除検証。 |

## 分類上の注記

- 初版（2026-07-10）の分類はユーザー提示の分類表（design 4本 / reports 9本 / ops 4本、明示列挙）に従い、archive/dqn は各カテゴリに明示列挙されなかった残り全部というルールで機械的に確定した。以後の追加分は同じ基準で振り分けている。
- 本数は実カウントが正（2026-07-25 時点: design 8 / reports 13 / ops 9 / archive/dqn 36 / archive/ppo_p1p2 5、`configs/archive` 25）。
- reports/ の小見出し（現行軸 / DRCA / 探索ラダー / 生成物）は読者の導線のための便宜的な区分で、ディレクトリ構造は分けていない。時代が閉じた群のみ `archive/<era>/` へ実際に退避する（dqn → ppo_p1p2 の順で運用中）。
- 更新規律: 文書を追加・ステータス変更したら本索引も同一 commit で更新する。
