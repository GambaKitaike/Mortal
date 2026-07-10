# PPO P3 Stage2 — eval バッテリー結果 (2026-07-10)

**run:** stage2_20260709_092541 (step 0–6000、ディスク枯渇クラッシュにより中断) +
stage2_20260709_194510_resume (step_006000.pth から resume、step 6000–16000)
**本 md の位置づけ:** `stage2_design.md` §5 の eval バッテリー(3レンズ)結果のみを収める。
判定窓 (step 8000–16000、`stage2_design.md` §4) の判定そのものは設計監督側が別途記入する
（本 md は判定節を含まない）。
**実行:** 実装エージェント（Claude Code CLI、ローカル WSL）が 2026-07-10 に eval のみ実行。
学習は行っていない（凍結は step 16000 完走により解除済み）。

---

## 0. 完走確認・保全チェック

### checkpoints

**resume run (`stage2_20260709_194510_resume/checkpoints/`)** — 判定窓の主対象:

| file | size | mtime |
|---|---:|---|
| step_000000.pth | 43,699,891 | 2026-07-09 19:45 (init を複製) |
| step_002000.pth | 130,741,580 | 2026-07-09 19:45 (旧run step2000 を複製) |
| step_004000.pth | 130,741,580 | 2026-07-09 19:45 (旧run step4000 を複製) |
| step_006000.pth | 130,741,580 | 2026-07-09 19:45 (旧run step6000 を複製、resume 起点) |
| step_008000.pth | 130,746,124 | 2026-07-09 22:50 |
| step_010000.pth | 130,746,124 | 2026-07-10 01:54 |
| step_012000.pth | 130,746,124 | 2026-07-10 05:06 |
| step_014000.pth | 130,746,124 | 2026-07-10 08:25 |
| step_016000.pth | 130,746,124 | 2026-07-10 11:31 |

**旧run (`stage2_20260709_092541/checkpoints/`)** — ディスク枯渇クラッシュ以前の保全記録
（step2000/step4000 は本 eval バッテリーのレンズ1でこちらから直接使用）:

| file | size | mtime |
|---|---:|---|
| step_000000.pth | 43,699,891 | 2026-07-09 09:38 |
| step_002000.pth | 130,741,580 | 2026-07-09 11:16 |
| step_004000.pth | 130,741,580 | 2026-07-09 13:16 |
| step_006000.pth | 130,741,580 | 2026-07-09 15:43 |

### ppo_diag.jsonl / tb

| run | ppo_diag.jsonl | tb (events.out.tfevents...) |
|---|---:|---:|
| resume (0–16000) | 22,759,491 bytes | 1,958 bytes |
| 旧run (0–6000、破棄枝含む) | 17,053,524 bytes | 保全済み（`logs/`） |

### 完走・監視項目（tmux `ppo_stage2_20260709_194510_resume` 末尾ログより）

```
reached step 16000
ppo step 16000: epoch1_clip=0.1173 epoch4_clip=0.1071 ev=-0.1398
mismatch: 0
fallback: 0
loader_delta: 5780   (INFO、非致命の既知クラス)
chip errors: 0
=== Done (eval at checkpoints: run_eval separately) ===
Cleanup...
Terminated
exit=0
```

`trainer.log` 全文（resume run 通し）を `warning|error|nan` (大小無視) で走査した結果は
0 件。`server.log` も `error|traceback|exception` 0 件。alive_clients は run 全体を通して
一貫して `6/3` 表記（3 論理クライアント × プロセス2本 = 6 プロセスをカウントしている仕様。
CLAUDE.md の「3/3」という書式と数え方が異なるだけで、値自体は run 中不変であり異常ではない）。

**必須監視3項目（mismatch/fallback/chip errors）は run 全体を通じて 0 を維持。
trainer NaN 0。完走確認: PASS。**

### インシデント発見: eval 直前に検出した残党プロセス（本タスク範囲内で対処済み）

lens1 eval バッテリーの初回起動が `port 5000 in use` で abort した。調査の結果、
resume run の `run_server.py`（PID 328537）と `run_client.py` ×3（PID 328622/328666/328711、
各々に conda wrapper 子プロセス付き、計6プロセス）が、トレーナーが 2026-07-10 11:31 に
正常終了（exit=0、ログ上は `Cleanup...` → `Terminated` まで到達）した後も生き残っていた。
`client0.log`/`server.log` の最終書き込みは同日 13:50 で、以降 21:43 の発見時点まで
約8時間、ログ増加なしのまま GPU を掴み続けていた（74% CPU、port 5000 LISTEN 保持）。
`run_ppo_stage2_resume.sh` 内の `Cleanup...` ステップがトレーナー以外の子プロセス
（server/client）の終了を捕捉できていない疑いがある。

**対処:** `pkill -f` はコマンドライン文字列に自分自身の呼び出し文字列が含まれ自爆する
事故が1回発生（シェルが SIGTERM で落ちた）。以降は該当 PID を直接 `kill -TERM` して解消。
port 5000 clear・GPU idle を確認の上で eval を再実行した。

**恒久対策は未実施**（run 完走後の別タスク候補: launcher の Cleanup ステップが
server/client を確実に道連れにするよう修正するか、preflight 側の残党チェックを
eval 系スクリプトにも一貫して要求する運用を明文化するか）。

---

## 1. レンズ1: 標準 argmax eval（配備レンズ、自然分布）

**実行条件:** RUN_DIR=stage2_20260709_194510_resume、checkpoint 2000/4000 は旧run
(stage2_20260709_092541) から、8000/12000/16000 は resume run から。init=
beta1_huber_192x40/mortal.pth。seeds [10000, 10100)、各 checkpoint 100 半荘直列、
argmax (eval_mode=True)、guard ON、p_enrich=0（自然分布）。
スクリプト: `run_eval_battery_stage2.sh` / `eval_ppo_smoke_sanity.py`（新規、Stage1 の
`run_eval_battery_stage1.sh` を checkpoint 参照先のみ変更して踏襲）。

| label | fuuro | riichi | agari | houjuu | ryukyoku | avg_rank |
|---|---:|---:|---:|---:|---:|---:|
| init | 17.22% | 23.57% | 20.59% | 12.29% | 18.13% | 2.5000 |
| step2000 | 5.49% | 35.15% | 19.79% | 11.94% | 21.05% | 2.5000 |
| step4000 | 10.94% | 32.81% | 20.01% | 12.95% | 20.46% | 2.5000 |
| step8000 | 7.94% | 37.15% | 20.69% | 13.59% | 18.36% | 2.5000 |
| step12000 | 9.78% | 35.76% | 20.34% | 13.03% | 18.95% | 2.5000 |
| step16000 | 6.50% | 35.48% | 19.58% | 12.02% | 22.18% | 2.5000 |

（avg_rank は全行 2.5000 = 同一モデル4面打ちの構造的必然。init 行は Stage1 の
`ppo_p3_stage1_result.md` §6 init 行と完全一致 — 同一 checkpoint・同一 seed による
既知の整合性チェックとして記録）

**構成 dump（この eval 経路の p_enrich 確認）:** `eval_ppo_smoke_sanity.py` に本タスクで
追加した診断 engine dump より、全 checkpoint で
`{'eval_mode': True, 'enable_rule_based_agari_guard': True, ..., 'p_enrich': 0.0}` を
確認・assert 済み（例: init `eval engine config dump: {... 'p_enrich': 0.0}`）。

---

## 2. レンズ2: grp_baseline (DQN) 1v3 対戦

**実行条件:** challenger=argmax/guard ON、baseline=beta1_huber_192x40/mortal.pth
（Stage1 と同一固定 baseline。grp_baseline.pth は配管 fixture のため使用禁止 — 既定の
運用踏襲）、seeds [10000, 10100)、4半荘/seed の座席均等ローテ、400半荘/checkpoint。
スクリプト: `run_eval_grp_baseline_1v3_stage2.sh`（Stage1 の
`run_eval_grp_baseline_1v3.sh` を RUN_DIR のみ変更して踏襲）。

### 表1: argmax 打牌統計 + avg_rank（challenger視点）

| label | avg_rank | fuuro | riichi | agari | houjuu | ryukyoku |
|---|---:|---:|---:|---:|---:|---:|
| init | 2.4750 | 17.20% | 23.81% | 20.67% | 12.10% | 18.35% |
| step16000 | 2.7050 | 5.22% | 34.63% | 17.61% | 15.42% | 19.01% |

（init 行は Stage1 の同表と完全一致 — 同一 checkpoint・同一 baseline・同一 seed による
既知の整合性チェック。rankings_1st_2nd_3rd_4th: init=[105,99,97,99]、
step16000=[80,83,112,125]）

### 表2: フリー雀荘収支（per 半荘、mean±SE、`analyze_freeparlor_pnl_1v3.py`）

| stream | init | step16000 |
|---|---:|---:|
| 素点 | −4.656±0.730 | −7.643±0.777 |
| 順位点 | +0.600±1.160 | −3.975±1.111 |
| チップ（枚） | +0.115±0.225 | −0.103±0.225 |
| 合算 | −3.481±2.719 | −12.131±2.749 |
| チップ/局 | +0.0110±0.0221 | −0.0100±0.0232 |

n_hanchan=400（両checkpoint）、n_kyoku: init=4175、step16000=4118。init 行は Stage1 の
同表と完全一致（較正チェック、ミラーマッチ理論値: 素点−5/順位点0/チップ0 に整合）。

**構成 dump:** challenger/baseline とも `dump_engine_config` で p_enrich=0.0 を確認
（baseline は DQN の MortalEngine で `getattr` フォールバックにより 0.0）。
`challenger engine config: {..., 'eval_mode': True, ..., 'p_enrich': 0.0}` /
`baseline engine config: {..., 'p_enrich': 0.0}`。

---

## 3. レンズ3: メタ対決 probe（Stage2 step16000 vs Stage1 step16000 ×3）

**実行条件:** challenger=Stage2 step16000（stage2_20260709_194510_resume、argmax/guard ON）、
baseline=Stage1 step16000（stage1_20260706_020120_resume/checkpoints/step_016000.pth、
argmax/guard ON）×3席。座席ローテ・seed範囲・半荘数はレンズ2と同一仕様
（seeds [10000,10100)、400半荘）。既存 `eval_grp_baseline_1v3.py` の
`build_challenger_engine` に `name` 引数を追加し（後方互換のデフォルト値付き）、
baseline 側も同一関数で PPOEngine として構築する新規スクリプト
`eval_meta_stage1_vs_stage2.py` / `run_eval_meta_stage1_vs_stage2.sh` で実行。

### 表1: argmax 打牌統計 + avg_rank（Stage2 step16000 視点）

| avg_rank | fuuro | riichi | agari | houjuu | ryukyoku |
|---:|---:|---:|---:|---:|---:|
| 2.4475 | 6.38% | 35.14% | 19.97% | 12.19% | 20.39% |

rankings_1st_2nd_3rd_4th=[108, 100, 97, 95]

### 表2: フリー雀荘収支（Stage2 step16000 視点、per 半荘、mean±SE）

| stream | 値 |
|---|---:|
| 素点 | −4.851±0.824 |
| 順位点 | +1.125±1.162 |
| チップ（枚） | −0.105±0.243 |
| 合算 | −4.251±2.883 |
| チップ/局 | −0.0108±0.0256 |

n_hanchan=400、n_kyoku=3890。

**構成 dump:** challenger/baseline とも同一関数 (`build_challenger_engine`) 経由の
PPOEngine で `eval_mode=True`（argmax）・`enable_rule_based_agari_guard=True`（guard ON）・
`p_enrich=0.0` をスクリプト内で assert 済み。ログ実測:
`challenger engine config: {..., 'p_enrich': 0.0}` /
`baseline engine config: {..., 'p_enrich': 0.0}`（両者とも eval_mode=True で同一設定）。

---

## 4. p_enrich=0 確認まとめ（全レンズ横断）

| レンズ | 経路 | p_enrich dump |
|---|---|---:|
| 1 (標準argmax) | `eval_ppo_smoke_sanity.py`（診断engine、本タスクで追加） | 0.0（全6 checkpoint で assert） |
| 2 (grp_baseline 1v3) | `eval_grp_baseline_1v3.py`（challenger=PPOEngine明示0.0既定 / baseline=DQN MortalEngine、getattr fallback 0.0） | 0.0（両ラベルで確認） |
| 3 (メタ対決probe) | `eval_meta_stage1_vs_stage2.py`（challenger/baseline とも PPOEngine明示0.0既定） | 0.0（assert済み） |

いずれも訓練 client 専用の p_enrich 介入（`stage2_design.md` §2）は不使用であり、
全 eval は自然配牌分布上で実行されている。

---

## 5. sanity / 留意

- 本 md は eval バッテリー(3レンズ)の実行結果のみを収める。判定窓 (step 8000–16000) に
  対する判定 1/2/3（`stage2_design.md` §4）の適用は設計監督側の別タスク
- レンズ1 の初回起動時に検出した残党プロセス（server/client、8時間放置）は本タスク内で
  安全に対処済み（§0 参照）。恒久対策（launcher Cleanup ステップの修正）は別タスク候補
- レンズ2/3 は同一 seed・同一座席ローテ仕様のため、init 行・較正チェックは Stage1
  結果 (`ppo_p3_stage1_result.md`) と直接比較可能。レンズ3 は新規ハーネスのため
  Stage1 に対応する数値は存在しない（本 run が初回実行）
- 新規追加ファイル: `freeparlor/scripts/run_eval_battery_stage2.sh`、
  `freeparlor/scripts/run_eval_grp_baseline_1v3_stage2.sh`、
  `freeparlor/scripts/eval_meta_stage1_vs_stage2.py`、
  `freeparlor/scripts/run_eval_meta_stage1_vs_stage2.sh`。既存ファイルへの変更:
  `eval_ppo_smoke_sanity.py`（診断engine dump+assert 追加）、
  `eval_grp_baseline_1v3.py`（`build_challenger_engine` に `name` 引数追加、
  既定値 'challenger' で後方互換）。学習コード・config・凍結対象には一切触れていない
