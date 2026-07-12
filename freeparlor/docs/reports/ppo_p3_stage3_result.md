# PPO P3 Stage3 — eval バッテリー結果 (2026-07-13)

**run:** stage3_20260712_033403（init から step 16000 まで単一 run・クラッシュなし完走）
**本 md の位置づけ:** `stage3_design.md` §6 の eval バッテリー(3レンズ)結果のみを収める。
判定窓 (step 8000–16000、`stage3_design.md` §4) の判定条件 1/2/3 への照合は
設計監督側の別タスク（本 md には判定節を含めない）。
**実行:** 実装エージェント（Claude Code CLI、ローカル WSL）が 2026-07-13 に eval のみ実行。
学習は行っていない（凍結は step 16000 完走により解除済み）。GPU 1系統・3レンズ直列。

---

## 0. 完走確認・保全チェック

### checkpoints (`stage3_20260712_033403/checkpoints/`)

| file | size | mtime |
|---|---:|---|
| step_000000.pth | 43,699,891 | 2026-07-12 03:49 |
| step_002000.pth | 130,741,644 | 2026-07-12 05:39 |
| step_004000.pth | 130,741,644 | 2026-07-12 07:44 |
| step_006000.pth | 130,741,644 | 2026-07-12 10:23 |
| step_008000.pth | 130,741,644 | 2026-07-12 13:22 |
| step_010000.pth | 130,741,644 | 2026-07-12 16:37 |
| step_012000.pth | 130,741,644 | 2026-07-12 19:57 |
| step_014000.pth | 130,741,644 | 2026-07-12 23:09 |
| step_016000.pth | 130,746,188 | 2026-07-13 02:18 |

`step_016000.pth` は `steps=16000`・`actor_critic` state 込みでロード可能なことを確認済み
（sha256 `2c84a341...951df7`、本タスクで再確認）。ppo_diag.jsonl 144,008 行、tb/logs 保全済み。

### 完走・監視項目（`logs/trainer.log` 末尾より）

```
2026-07-13 02:17:33 ppo step 16000: epoch1_clip=0.1922 epoch4_clip=0.1961 ev=0.0640
2026-07-13 02:17:35 saved numbered checkpoint: .../checkpoints/step_016000.pth
reached step 16000
mismatch: 0
fallback: 0
loader_delta: 12091   (INFO、非致命の既知クラス)
chip errors: 0
=== Done (eval at checkpoints: run_eval separately) ===
Cleanup...
Terminated
exit=0
```

`trainer.log` 全文を `warning|error|nan`（大小無視）で走査した結果は 0 件。`server.log` も
`error|traceback|exception` 0 件。anneal スケジュール違反（call_bonus_b が固定 5.0 区間・
線形区間・0 区間の境界を外れて適用された記録）は検定(18)の schedule 境界値チェックと
run 内 call_bonus イベントログの双方で確認されておらず、違反 0 件。

**必須監視4項目（mismatch/fallback/chip errors/trainer NaN）は run 全体を通じて 0。完走確認: PASS。**

### インシデント発見: watchdog×Cleanup 競合による良性の二重起動（本タスク範囲内で確認・対処不要）

`trainer.log` を精査したところ、step 16000 到達時に以下の二重起動シーケンスが記録されている:

1. 02:17:33–02:17:35 通常の学習ループで step 16000 に到達、checkpoint 保存、
   `param has been submitted`
2. 02:17:37 `saved numbered checkpoint` 再掲後、`Terminated`（trainer プロセスの正常終了）
3. 02:17:59 別プロセスが `loaded checkpoint: steps=16,000` → 直後に
   `reached max_steps=16,000, stopping` → checkpoint 再保存

原因は `run_ppo_p3_stage1_inner.sh` の `trainer_watchdog()`: trainer が exit code 0 で
正常終了しても、server プロセスがまだ生きている間は「異常終了」と区別せず即座に
`start_trainer` を再実行する（`steps >= MAX_STEPS` の判定は外側の監視ループ側にしかなく、
watchdog 自身は関知しない）。再起動されたプロセスは checkpoint をロードした直後に
`max_steps` 到達を検知して即終了するため、**追加の学習ステップは一切実行されない**
（2回目の起動は「読み込み→即終了→再保存」のみ）。step 16000 の checkpoint は両方の
起動で同一の学習状態を保存したものであり、学習内容への影響はない。

**対応:** 本タスクの範囲では対処不要（eval に影響しないベニグンな既知パターン）。
恒久対策（watchdog に `MAX_STEPS` 到達後の再起動抑制を追加する）はバックログ#4
「launcher Cleanup ステップの修正」に統合される新規証拠として記録する。

---

## 1. レンズ1: 標準argmax eval（自己対戦、自然分布・b=0固定 eval 経路）

**実行条件:** RUN_DIR=stage3_20260712_033403、checkpoint 2000/4000/8000/12000/16000
+ init（beta1_huber_192x40/mortal.pth）。seeds [10000, 10100)、各 checkpoint 100 半荘直列、
argmax (eval_mode=True)、guard ON、p_enrich=0・call_bonus_b=0（eval 経路の既定）。
スクリプト: `run_eval_battery_stage3.sh`（Stage2 版の RUN_DIR のみ変更・新規）/
`eval_ppo_smoke_sanity.py`（本タスクで call_bonus_b=0 assert を追加）。

| label | fuuro | riichi | agari | houjuu | ryukyoku | avg_rank |
|---|---:|---:|---:|---:|---:|---:|
| init | 17.22% | 23.57% | 20.59% | 12.29% | 18.13% | 2.5000 |
| step2000 | 62.86% | 15.92% | 22.37% | 15.33% | 11.19% | 2.5000 |
| step4000 | 77.84% | 12.58% | 23.20% | 17.59% | 9.14% | 2.5000 |
| step8000 | 38.89% | 23.69% | 22.45% | 15.42% | 11.39% | 2.5000 |
| step12000 | 26.66% | 30.86% | 21.02% | 13.74% | 16.36% | 2.5000 |
| step16000 | 14.66% | 33.24% | 20.34% | 13.33% | 19.14% | 2.5000 |

（avg_rank は全行 2.5000 = 同一モデル4面打ちの構造的必然。**init 行は Stage1/2 の
init 行と完全一致** — 同一 checkpoint・同一 seed による既知の整合性チェック。
step2000/4000 の副露率スパイク（62.86%/77.84%）は b=5.0 固定期の強制探索として
`stage3_design.md` §5 に事前登録済みの想定挙動であり異常ではない。step8000 以降
anneal 完了に伴い副露率が単調に低下し、step16000 で 14.66% まで沈静化）

**構成 dump（p_enrich / call_bonus_b 確認）:** 全6 checkpoint で
`{'eval_mode': True, 'enable_rule_based_agari_guard': True, ..., 'p_enrich': 0.0, 'call_bonus_b': 0.0}`
を確認（例: init `eval engine config dump: {..., 'p_enrich': 0.0, 'call_bonus_b': 0.0}`）。
6 checkpoint すべてで両値 0.0 を assert 済み（`eval_ppo_smoke_sanity.py` に
`call_bonus_b` assert を本タスクで追加）。

---

## 2. レンズ2: grp_baseline (DQN) 1v3 対戦（配備レンズ）

**実行条件:** challenger=argmax/guard ON、baseline=beta1_huber_192x40/mortal.pth
（Stage1/2 と同一固定 baseline。grp_baseline.pth は配管 fixture のため使用禁止）、
seeds [10000, 10100)、4半荘/seed の座席均等ローテ、400半荘/checkpoint。
スクリプト: `run_eval_grp_baseline_1v3_stage3.sh`（Stage2 版の RUN_DIR のみ変更・新規）。

### 表1: argmax 打牌統計 + avg_rank（challenger視点）

| label | avg_rank | fuuro | riichi | agari | houjuu | ryukyoku |
|---|---:|---:|---:|---:|---:|---:|
| init | 2.4750 | 17.20% | 23.81% | 20.67% | 12.10% | 18.35% |
| step16000 | 2.5800 | 13.08% | 33.93% | 18.80% | 15.18% | 18.97% |

（**init 行は Stage1/2 の同表と完全一致** — 既知の整合性チェック。
rankings_1st_2nd_3rd_4th: init=[105,99,97,99]、step16000=[97,94,89,120]）

### 表2: フリー雀荘収支（per 半荘、mean±SE、`analyze_freeparlor_pnl_1v3.py`）

| stream | init | step16000 |
|---|---:|---:|
| 素点 | −4.656±0.730 | −5.7575±0.8028 |
| 順位点 | +0.600±1.160 | −1.1750±1.1624 |
| チップ（枚） | +0.115±0.225 | +0.2550±0.2430 |
| 合算 | −3.481±2.719 | −5.6575±2.8973 |
| チップ/局 | +0.0110±0.0221 | +0.0246±0.0228 |

n_hanchan=400（両checkpoint）、n_kyoku: init=4175、step16000=4144。**init 行は Stage1/2
の同表と完全一致**（較正チェック、ミラーマッチ理論値: 素点−5/順位点0/チップ0 に整合）。

### 配備税チェック（事前登録済み、Stage2 7c と同一手法・2SE 基準）

| stream | 差分 (step16000 − init) | SE(差) | diff/SE | 有意性 (\|diff/SE\|≥2) |
|---|---:|---:|---:|---|
| 素点 | −1.1015 | 1.0854 | −1.015 | n.s. |
| 順位点 | −1.7750 | 1.6420 | −1.081 | n.s. |
| チップ | +0.1400 | 0.3312 | +0.423 | n.s. |
| 合算 | −2.1765 | 3.9734 | −0.548 | n.s. |

avg_rank 2.4750→2.5800（差+0.105、Stage2 の 2.4750→2.7050 差+0.2300 より小さい）。
判定文言はここには書かない（判定は設計監督側の別タスク）。

**構成 dump:** challenger/baseline とも `dump_engine_config` で p_enrich=0.0・
call_bonus_b=0.0 を確認（baseline は DQN の MortalEngine で `getattr` フォールバックにより
両値とも 0.0）。`challenger engine config: {..., 'p_enrich': 0.0, 'call_bonus_b': 0.0}` /
`baseline engine config: {..., 'p_enrich': 0.0, 'call_bonus_b': 0.0}`。

---

## 3. レンズ3: メタ対決 probe（Stage3 step16000 vs Stage1 step16000 ×3）

**実行条件:** challenger=Stage3 step16000（stage3_20260712_033403、argmax/guard ON）、
baseline=Stage1 step16000（stage1_20260706_020120_resume/checkpoints/step_016000.pth、
argmax/guard ON）×3席。座席ローテ・seed範囲・半荘数はレンズ2と同一仕様
（seeds [10000,10100)、400半荘）。既存 `eval_meta_stage1_vs_stage2.py` ハーネスを
チェックポイントパスのみ差し替えて流用（スクリプト本体は無変更、call_bonus_b assert
のみ本タスクで追加）、新規ラッパー `run_eval_meta_stage1_vs_stage3.sh` で実行。

### 表1: argmax 打牌統計 + avg_rank（Stage3 step16000 視点）

| avg_rank | fuuro | riichi | agari | houjuu | ryukyoku |
|---:|---:|---:|---:|---:|---:|
| 2.6100 | 14.64% | 33.58% | 19.31% | 14.30% | 20.29% |

rankings_1st_2nd_3rd_4th=[86, 91, 116, 107]

### 表2: フリー雀荘収支（Stage3 step16000 視点、per 半荘、mean±SE）

| stream | 値 |
|---|---:|
| 素点 | −7.4175±0.7783 |
| 順位点 | −2.3750±1.1148 |
| チップ（枚） | −0.6750±0.2362 |
| 合算 | −13.1675±2.7721 |
| チップ/局 | −0.0697±0.0256 |

n_hanchan=400、n_kyoku=3874。

Stage2 の同レンズ（Stage2-16000 vs Stage1-16000 ×3）はミラーマッチ理論値圏内
（素点 −4.851±0.824 ≈ −5）だったのに対し、本 run（Stage3-16000 vs Stage1-16000 ×3）は
全ストリームで理論値から明確に外れている（素点 −7.42 ≈ 理論値−5から−2SE超、
チップ −0.675±0.236 ≈ −2.9SE）。解釈はここに書かない（判定は設計監督側の別タスク）。

**構成 dump:** challenger/baseline とも同一関数 (`build_challenger_engine`) 経由の
PPOEngine で `eval_mode=True`（argmax）・`enable_rule_based_agari_guard=True`（guard ON）・
`p_enrich=0.0`・`call_bonus_b=0.0` をスクリプト内で assert 済み。ログ実測:
`challenger engine config: {..., 'p_enrich': 0.0, 'call_bonus_b': 0.0}` /
`baseline engine config: {..., 'p_enrich': 0.0, 'call_bonus_b': 0.0}`（両者とも eval_mode=True で同一設定）。

---

## 4. p_enrich=0 / call_bonus_b=0 確認まとめ（全レンズ横断）

| レンズ | 経路 | p_enrich dump | call_bonus_b dump |
|---|---|---:|---:|
| 1 (標準argmax、6 checkpoint) | `eval_ppo_smoke_sanity.py` | 0.0（全6 checkpoint で assert） | 0.0（全6 checkpoint で assert、本タスクで追加） |
| 2 (grp_baseline 1v3、2 checkpoint) | `eval_grp_baseline_1v3.py` | 0.0（challenger明示既定 / baseline getattr fallback） | 0.0（challenger明示既定・本タスクで assert 追加 / baseline getattr fallback） |
| 3 (メタ対決probe、1試合) | `eval_meta_stage1_vs_stage2.py`（ハーネス流用） | 0.0（両者 assert 済み） | 0.0（両者 assert 済み、本タスクで追加） |

いずれも訓練 client 専用の p_enrich（Stage2）・call_bonus_b（Stage3）介入は不使用であり、
全 eval は自然配牌分布・b=0 の正典報酬上で実行されている。

---

## 5. sanity / 留意

- 本 md は eval バッテリー(3レンズ)の実行結果のみを収める。判定窓 (step 8000–16000) に
  対する判定 1/2/3（`stage3_design.md` §4）の適用は設計監督側の別タスク
- §0 で確認した watchdog×Cleanup 競合による良性の二重起動は、学習内容・checkpoint
  内容に影響を与えていない（steps=16000 一貫、二重目の起動は即終了のみ）ことを
  本タスクで確認済み。恒久対策はバックログ#4 に統合
- レンズ2/3 は同一 seed・同一座席ローテ仕様のため、init 行・較正チェックは Stage1/2
  結果と直接比較可能（本 md §1・§2 で両方とも完全一致を確認）。レンズ3 の Stage1 比較
  対象数値は Stage2 の同レンズで先例あり（`ppo_p3_stage2_result.md` §3）
- 新規追加ファイル: `freeparlor/scripts/run_eval_battery_stage3.sh`、
  `freeparlor/scripts/run_eval_grp_baseline_1v3_stage3.sh`、
  `freeparlor/scripts/run_eval_meta_stage1_vs_stage3.sh`。既存ファイルへの変更:
  `eval_ppo_smoke_sanity.py`（`call_bonus_b` assert 追加）、
  `eval_grp_baseline_1v3.py`（challenger dump に `p_enrich`/`call_bonus_b` assert 追加）、
  `eval_meta_stage1_vs_stage2.py`（`call_bonus_b` assert 追加）。学習コード・config・
  訓練 launcher・検定ロジック（`verify_ppo_p1.py`）には一切触れていない
- GPU 1系統・3レンズ直列を遵守（各レンズ実行前に port 5000 clear・GPU idle を確認済み）
