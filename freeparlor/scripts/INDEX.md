# freeparlor/scripts/ 索引

生成日: 2026-07-10。ファイル移動・改変は無し（本索引のみ新規追加）。ディレクトリはフラット構成を維持。
実カウント: 47本（`__pycache__` を除く）。

## 既知事項: docs/ 出力パスの陳腐化

`freeparlor/docs/` の分類フォルダ化（design/reports/ops/archive/dqn）に伴い、以下のスクリプトが
ハードコードしている出力先パスが古いフラット構成のままになっている（**スクリプト自体は無改変** —
本タスクのスコープ外）:

- `count_aka_call_hora.py` → `docs/aka_call_hora_count.md`（実体は `docs/archive/dqn/` に移動済み）
- `analyze_kyoku_length.py` → `docs/kyoku_length_dist.md`（同上）
- `call_channel_diag.py` → `docs/call_channel_diag.md`（同上）
- `mqw03_collapse_diag.py` → `docs/mqw03_collapse_diag.md`（同上）
- `generate_phase4d_results.py` → `docs/phase4d_sweep_results.md`（同上。上記4本と合わせ計5本が archive/dqn 対象のクローズ済み診断）
- `verify_ppo_p1.py` → `docs/ppo_p1_verify_log.txt`（**要注意**: これは archive 済みの診断ではなく、
  preflight で毎回実行される現行検証スクリプト。実体は `docs/reports/` に移動済みのため、
  次回実行時は `docs/reports/ppo_p1_verify_log.txt` を更新する代わりに `docs/ppo_p1_verify_log.txt`
  へ新規ファイルを作ってしまう）

対応（別タスク）: 各スクリプトの `MD_PATH`/`OUT_MD`/`out_path` 定義を新パスに更新するか、
出力後に `git mv` する運用にするか要判断。

## run/ — run発進・orchestrator（17本）

| ファイル |
|---|
| `run_ppo_p3_stage1.sh` |
| `run_ppo_p3_stage1_inner.sh` |
| `run_ppo_p3_orchestrator.sh` |
| `run_ppo_p3_resume.sh` |
| `run_ppo_p3_recover.sh` |
| `run_ppo_p3_mismatch_repro.sh` |
| `run_ppo_p2_smoke.sh` |
| `run_ppo_p2b_lr_probe.sh` |
| `run_ppo_p2c_advantage_decomp.sh` |
| `run_ppo_stage2.sh` |
| `run_ppo_stage2_resume.sh` |
| `run_p2_mismatch_forensic.sh` |
| `run_ppo_p3_eval_checkpoint.sh` |
| `run_eval_battery_stage1.sh` |
| `run_eval_grp_baseline_1v3.sh` |
| `run_eval_ppo_control.sh` |
| `run_eval_ppo_smoke_sanity.sh` |

## eval/ — 評価実行（2本）

| ファイル |
|---|
| `eval_grp_baseline_1v3.py` |
| `eval_ppo_smoke_sanity.py` |

## analyze/ — 集計・診断・データ生成（17本）

| ファイル |
|---|
| `analyze_aka_conditional.py` |
| `analyze_aka_conditional_human.py` |
| `analyze_chip_realize.py` |
| `analyze_freeparlor_pnl_1v3.py` |
| `analyze_kyoku_length.py` |
| `call_channel_diag.py` |
| `check_stage2_launch_gate.py` |
| `collect_cql_sweep_metrics.py` |
| `collect_cql_sweep_one.py` |
| `collect_ppo_p2_metrics.py` |
| `count_aka_call_hora.py` |
| `generate_phase4d_results.py` |
| `measure_pool_vram.py` |
| `mqw03_collapse_diag.py` |
| `parse_p2_mismatch.py` |
| `parse_p2_mismatch_forensic.py` |
| `preprocess_chips.py` |

## verify/ — 検定（6本）

| ファイル |
|---|
| `verify_ppo_p1.py` |
| `verify_agari_detail.py` |
| `verify_arena_chip_delta.py` |
| `verify_lambda_opp_zero.py` |
| `verify_layer3_chip.py` |
| `verify_td_transitions.py` |

## summarize/ — レポート整形（4本）

| ファイル |
|---|
| `summarize_p2b_action_mass.py` |
| `summarize_p2c_advantage_decomp.py` |
| `summarize_p3_stage1.py` |
| `summarize_ppo_diag.py` |

## misc/ — その他インフラ（1本）

| ファイル |
|---|
| `preflight_libriichi.sh` |
