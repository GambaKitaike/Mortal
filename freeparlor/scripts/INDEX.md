# freeparlor/scripts/ 索引

初版: 2026-07-10。**最終更新: 2026-07-25**（DRCA・anchor・eval 系の追加を反映、分類を用途別に再構成）。
実カウント: **73本**（`.py` 44 / `.sh` 29、`__pycache__` を除く）。
ディレクトリは**フラット構成を維持**（下記「移動禁止の理由」参照）。

## 移動禁止の理由（2026-07-25 追記）

このディレクトリを**サブディレクトリ化してはいけない**。run dir 側に生成された運用スクリプトが
ここのパスを**絶対パスで**参照しており、リポジトリ外（および退避 tar の中）にも同じ参照が残っている:

- `runs/drca/main_*/run_resume.sh` → `freeparlor/scripts/drca_run_probe.py`
  （中断中の DRCA 第3枠の再開手段。退避 tar
  `backups/drca/main_a_s3final_*.tar.gz` の中身も同じ）
- `runs/drca/*/run_pilot.sh` / `run_main_s1final.sh` → 同上（過去 run の再現手段）
- 学習 run の tmux は `run_ppo_p3_stage1_inner.sh` を絶対パスで exec し、
  その中から `preflight_libriichi.sh` / `verify_ppo_p1.py` を `$REPO/freeparlor/scripts/` で呼ぶ

分類は本索引の見出しで表現し、ファイル配置は変えない方針。

## 既知事項: docs/ 出力パスの陳腐化（DQN 期の5本のみ・実害なし）

`freeparlor/docs/` の分類フォルダ化（design/reports/ops/archive/dqn）以前のフラット構成を
ハードコードしたままのスクリプトが5本ある。いずれも**DQN 期のクローズ済み診断**で現行 PPO 本線では
実行されないため、実害なし・無改変で放置している:

- `count_aka_call_hora.py` → `docs/aka_call_hora_count.md`（実体は `docs/archive/dqn/`）
- `analyze_kyoku_length.py` → `docs/kyoku_length_dist.md`（同上）
- `call_channel_diag.py` → `docs/call_channel_diag.md`（同上）
- `mqw03_collapse_diag.py` → `docs/mqw03_collapse_diag.md`（同上）
- `generate_phase4d_results.py` → `docs/phase4d_sweep_results.md`（同上）

**`verify_ppo_p1.py` は修正済み**（2026-07-11、出力先 `docs/reports/ppo_p1_verify_log.txt`）。
これは preflight で毎回走る現行スクリプトのため、初版索引で「要注意」としていた項目は解消済み。

---

## run/ — 学習 run の発進・再開（16本）

現行軸は anchor 系列。すべて `runs/` の spawn ランチャを exec する構成で、
preflight（残党チェック・libriichi rebuild・検定全 PASS・ディスク空き）は launcher が自動実行する。

| ファイル | 内容 |
|---|---|
| `run_ppo_anchor_c.sh` | **現行**。anchor Arm C（opponent pool へ凍結 init を常駐）の発進 |
| `run_ppo_anchor_k.sh` | anchor Arm K（`ppo_loss` に masked full KL）の発進。未使用（K 未発進） |
| `run_ppo_p3_stage1_inner.sh` | **全 stage / resume 発進が exec する共有 inner**（server/trainer/client の起動・watchdog・monitor・cleanup） |
| `run_ppo_p3_stage1.sh` | Stage1 発進 |
| `run_ppo_p3_resume.sh` / `run_ppo_p3_recover.sh` | Stage1 の resume / 障害復帰 |
| `run_ppo_p3_orchestrator.sh` | Stage1 期の orchestrator（inner のインライン複製を含む・共通化候補） |
| `run_ppo_stage2.sh` / `run_ppo_stage2_resume.sh` | Stage2 発進 / resume |
| `run_ppo_stage3.sh` | Stage3 発進。ディスク検査（env `DISK_MIN_GB`）を preflight に導入した最初の launcher（以後 anchor 系列・`run_drca_main_frame.sh` も踏襲） |
| `run_ppo_p2_smoke.sh` / `run_ppo_p2b_lr_probe.sh` / `run_ppo_p2c_advantage_decomp.sh` | P2 期のスモーク・プローブ |
| `run_ppo_p3_mismatch_repro.sh` / `run_p2_mismatch_forensic.sh` | mismatch 再現・フォレンジック |
| `preflight_libriichi.sh` | libriichi rebuild + cp + import スモーク（学習発進・eval バッテリーの双方から呼ばれる） |

## gate/ — 発進ゲート判定（3本）

| ファイル | 内容 |
|---|---|
| `check_anchor_launch_gate.py` | anchor 系列の機械ゲート @step200（Arm C: anchor 採択率 / Arm K: §5-a1 の3条件） |
| `check_stage3_launch_gate.py` | Stage3 の二段ゲート v2（`--gate mechanical` @200 / `--gate learning_response` @2000） |
| `check_stage2_launch_gate.py` | Stage2 の機械ゲート @step500（鳴き可能局面中の赤保持割合） |

## eval/ — 評価（17本）

3レンズ（argmax eval バッテリー / grp_baseline 1v3 / メタ対決）+ ミラー較正脚。
**すべての eval 経路で訓練側介入が無効（`p_enrich=0` / `call_bonus_b=0` / `anchor_prob`・`kl_beta`）
であることを構成 dump で assert する。**

| ファイル | 内容 |
|---|---|
| `eval_ppo_smoke_sanity.py` | argmax 自己対戦（標準バッテリーの実体） |
| `eval_grp_baseline_1v3.py` | grp_baseline との 1v3 対戦 |
| `eval_meta_stage1_vs_stage2.py` | メタ対決 probe（Stage3 対決・ミラー較正でも再利用） |
| `run_eval_battery_stage1.sh` / `run_eval_battery_stage2.sh` / `run_eval_battery_stage3.sh` | 標準 argmax eval バッテリー（6 checkpoint） |
| `run_eval_grp_baseline_1v3.sh` / `run_eval_grp_baseline_1v3_stage2.sh` / `run_eval_grp_baseline_1v3_stage3.sh` | 1v3 対戦バッテリー |
| `run_eval_meta_stage1_vs_stage2.sh` / `run_eval_meta_stage1_vs_stage3.sh` | メタ対決 |
| `run_eval_meta_mirror.sh` | ミラー較正脚（X vs 3X で理論ミラー値からのゼロ点を実測。実 RUN は未実施） |
| `run_eval_ppo_control.sh` / `run_eval_ppo_smoke_sanity.sh` / `run_ppo_p3_eval_checkpoint.sh` | 個別 eval 実行 |
| `analyze_freeparlor_pnl_1v3.py` | 1v3 の素点/順位点/チップ 3ストリーム収支集計（`--mirror-calibration` 内蔵） |
| `analyze_fundamentals_1v3.py` | 1v3 の基礎指標を **init 基準の差分 + 半荘クラスタ SE + z** で比較。基本表（agari/houjuu/avg_rank、**判定が読む表**・計算経路は凍結）+ 拡張表（fuuro/riichi/ryukyoku/平均和了打点/平均放銃打点/順位分布/順位点/素点/チップ per 局）。打点・順位分布は libriichi `Stat.from_log` をログ単位で呼び、件数の一致を毎半荘 assert（2実装の突き合わせ）。`--basic-only` で拡張を無効化 |

## drca/ — DRCA プローブ（6本）

fork-by-replay による鳴き反実仮想アドバンテージの測定ハーネス。libriichi・学習コードは無変更。
**上記「移動禁止の理由」の主対象。**

| ファイル | 内容 |
|---|---|
| `drca_common.py` | 共有ヘルパ（記録用パススルー engine・sidecar 台本の読み書き・(role, seq) 照合） |
| `drca_collect_branchpoints.py` | 分岐点の採取（実 react_batch クエリ列を sidecar に記録） |
| `drca_run_probe.py` | 台本再生による両腕ロールアウト（`--parallel` / `--resume` 対応） |
| `drca_aggregate.py` | 集計（cluster-robust SE・exploratory 層別・`--expect-branch-points` 安全弁） |
| `run_drca_main_frame.sh` | 本測定の枠別 launcher（`a_init` / `a_s3final` / `a_s3mid` / `b_s3final`） |
| `drca_pilot_qualitative_drilldown.py` | パイロットの分岐点別 ΔQ̂ 極値抽出（read-only・exploratory） |

## aggregate/ — 判定集計・診断集計（12本）

| ファイル | 内容 |
|---|---|
| `aggregate_stage2_judgment.py` / `aggregate_stage2_secondary.py` | Stage2 の事前登録判定 / 副次集計 |
| `aggregate_stage3_judgment.py` / `aggregate_stage3_secondary.py` | Stage3 の事前登録判定 / 副次集計 |
| `summarize_p3_stage1.py` / `summarize_ppo_diag.py` | Stage1 判定集計 / ppo_diag.jsonl の汎用要約（**P2 期専用**。既定パスがハードコードで扱う event は `batch_lag`/`ppo_epoch` のみ。commit 済み `archive/ppo_p1p2/ppo_p2_diag.md` の再現器なので**出力書式を変えないこと**） |
| `analyze_ppo_optimization_health.py` | **L1（最適化衛生）の横断診断**。`runs/ppo/*/logs/ppo_diag.jsonl` を横断集計して batch_size 分布と `minibatch_size` 越え率 / clip_fraction を **param snapshot age と lag で条件付けた**表（`trainer_step=0` を陽性対照に取る）/ epoch 別 clip・ratio_std / advantage_decomp のカテゴリ別サンプル数 / action_mass トレンド / **欠損 event の明示**を出す。read-only・GPU 不要。`--run` で run 指定、`--root` で走査元、`-o` で保存。回帰テストの基準は完走済み `stage3_20260712_033403`（期待値は `docs/reports/ppo_optimization_health_20260725.md` §7）|
| `summarize_p2b_action_mass.py` / `summarize_p2c_advantage_decomp.py` | P2b/P2c の集計 |
| `collect_ppo_p2_metrics.py` | P2 期のメトリクス収集 |
| `parse_p2_mismatch.py` / `parse_p2_mismatch_forensic.py` | mismatch ログ解析 |

## verify/ — 検定（6本）

| ファイル | 内容 |
|---|---|
| `verify_ppo_p1.py` | **現行の検定スイート（発進 preflight で毎回全 PASS が必須）**。本数はスクリプト出力（`ALL N CHECKS PASSED`）が正。冒頭に `.so` 鮮度チェックのプリフライトゲートあり |
| `verify_agari_detail.py` / `verify_arena_chip_delta.py` / `verify_layer3_chip.py` / `verify_td_transitions.py` / `verify_lambda_opp_zero.py` | DQN 期の個別検定（現行 PPO 本線では未使用） |

## tools/ — 補助ツール（2本）

| ファイル | 内容 |
|---|---|
| `mjai_log_to_html.py` | mjai 牌譜（json.gz/jsonl）→ log-viewer HTML 生成。定性レビュー用 |
| `measure_pool_vram.py` | opponent pool の VRAM 実測 |

## dqn-era/ — オフライン DQN + CQL 期（11本、現行では不使用）

`analyze_aka_conditional.py` / `analyze_aka_conditional_human.py` / `analyze_chip_realize.py` /
`analyze_kyoku_length.py` / `call_channel_diag.py` / `count_aka_call_hora.py` /
`mqw03_collapse_diag.py` / `generate_phase4d_results.py` / `preprocess_chips.py` /
`collect_cql_sweep_metrics.py` / `collect_cql_sweep_one.py`

対応する結果文書は `freeparlor/docs/archive/dqn/`。上記「docs/ 出力パスの陳腐化」の5本はここに含まれる。
