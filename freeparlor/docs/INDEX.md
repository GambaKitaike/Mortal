# freeparlor/docs/ 索引

生成日: 2026-07-10。フォルダ再編（design / reports / ops / archive/dqn）に伴う索引。
ステータスは判断根拠が明確なもののみ厳密。曖昧なものは本文参照を推奨。

## design/ — 設計・pre-registration（4本）

| パス | 日付 | ステータス | 要約 |
|---|---|---|---|
| `design/ppo_migration_design.md` | 2026-07-04 | active | PPO移行の設計正典。教師データ非依存本線の実装設計（critic scale・希少性探索の分岐を含む）。 |
| `design/stage2_design.md` | 2026-07-09 | active | Stage2（配牌rejection samplingによる赤濃縮）の設計・事前登録済み判定条件。現行runの判定窓が閉じるまで凍結対象。 |
| `design/reward_design_teacherfree.md` | 2026-07-02 | active | 教師データ非依存の報酬設計（あ）確定版。reward_audit を受けた本線設計。 |
| `design/reward_audit_teacherfree.md` | 2026-07-02 | closed | `RewardCalculator.calc_delta_blend` の棚卸し（教師データ非依存化に向けた監査、reward_design の前段）。 |

## reports/ — PPO時代の確定レポート・run記録（9本）

| パス | 日付 | ステータス | 要約 |
|---|---|---|---|
| `reports/ppo_p1_plumbing.md` | 2026-07-02 | closed | PPO P1配管の実装サマリ。 |
| `reports/ppo_p1_verify_log.txt` | 2026-07-09 | closed | PPO P1 sanity verification の生ログ（`verify_ppo_p1.py` 出力）。 |
| `reports/ppo_p2_smoke.md` | 2026-07-02 | closed | PPO P2スモーク結果（OOM対策後更新）。 |
| `reports/ppo_p2_diag.md` | 2026-07-03 | closed | PPO P2 OOM対策後の診断再走結果。 |
| `reports/ppo_p2b_lr_probe.md` | 2026-07-03 | closed | PPO P2b lrプローブ（fuuro崩壊のlr要因検証）。 |
| `reports/ppo_p2c_advantage_decomp.md` | 2026-07-03 | closed | PPO P2c 宣言行動（鳴き・立直）のadvantage分解計装。 |
| `reports/ppo_p3_pause_resume.md` | 2026-07-06 | closed | PPO P3 run #7のpause/resume記録。 |
| `reports/ppo_p3_stage1.md` | 2026-07-07 | closed | PPO P3 Stage1本走のrun状態・インシデント史。 |
| `reports/ppo_p3_stage1_result.md` | 2026-07-08 | closed | PPO P3 Stage1判定結果（立直マキシマリズム、事前固定条件成立→Stage2移行確定）。 |

## ops/ — 運用文書（4本）

| パス | 日付 | ステータス | 要約 |
|---|---|---|---|
| `ops/supervisor_handbook.md` | 2026-07-10 | active | 設計監督Claude向け引き継ぎハンドブック（出力規約・確定知見・殺した仮説リスト §4）。 |
| `ops/next_steps_2.md` | 2026-07-03 | active | プロジェクト全体史・引き継ぎメモ（最終目標・現状まとめ）。CLAUDE.mdの「最初に読む文書」6番目。 |
| `ops/prompts_for_20260713.md` | 2026-07-08 | active | 2026-07-13投入用プロンプト集。プロンプト①完了、②③保留中につき「active」（歴史文書として本文は不改変保全の注記あり）。 |
| `ops/next_steps.md` | 2026-06-24 | closed | 旧引き継ぎメモ（2026-06-23時点）。`next_steps_2.md` に事実上置換済み。 |

## archive/dqn/ — オフラインDQN時代の完結文書（35本、全て closed）

日付は全て 2026-06-22〜2026-07-02。教師データ由来のオフラインDQN+CQL経路（現行PPO本線への移行前）の調査・診断・phase結果。`main` ブランチのDQN経路にのみ関連し、現行 `ppo-migration` の判断には使わない。

| パス | 日付 | 要約 |
|---|---|---|
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

- 分類はユーザー提示の分類表（design 4本 / reports 9本 / ops 4本、明示列挙）に従った。archive/dqn は各カテゴリに明示列挙されなかった残り全部というルールで機械的に確定。
- 提示された本数目安（archive/dqn 37本、configs/archive 27本）と実カウント（35本・25本）に差異あり。ヘッダ確認は本索引作成時に全ファイル実施済みで、実カウント側が正。
