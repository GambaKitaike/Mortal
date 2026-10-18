# 診断A — lambda_opp=0 の online 影響切り分け

**日付:** 2026-06-26  
**run:** `/home/gamba/mahjong/runs/online_diag_a/`  
**状態:** step2000 test_play まで完了（本報告時点）。診断B 起動のため run は停止済み。

---

## 目的

online 自己対戦で副露率が warmstart（17%）から step2000 時点で既に 61% まで跳升している。  
`lambda_opp=0.3`（phase4d の取りこぼし罰）が online TD 報酬に入り、鳴き圧を増幅している疑いを切り分ける。

**判定基準（事前）:**

| 結果 | 解釈 |
|---|---|
| lambda_opp=0 でも step2000 副露 ≈ 61% | 主因は lambda_opp **以外**（Q_main online 学習 / cql_loss=0 等） |
| lambda_opp=0 で副露が跳ねない（≈17–35% 帯） | 主因は lambda_opp。online では外すべき |

---

## 実験条件

| 項目 | online_main（参照） | 診断A |
|---|---|---|
| warm-start | phase4d lo=0.3 | 同一 |
| lambda_opp | **0.3** | **0.0** ← 唯一の変更 |
| cql_loss | 0（online 既定） | 0（変更なし） |
| min_q_weight | 5（未使用） | 5（未使用） |
| beta_sel | 0→0.3（step2000–4000 ramp） | 同一 |
| 構成 | server×1, trainer×1, client×3 | 同一 |
| test_play | self-play 3000 半荘 | 同一 |
| 目標 step | — | **2000**（test_every=2000） |

**変更していないもの:** Q_main 経路、chip_weight、beta_sel スケジュール、batch_size、モデル 192×40。

config: `/home/gamba/mahjong/runs/online_diag_a/config.toml`

---

## タイムライン

| 時刻 | イベント |
|---|---|
| 21:09 | 起動（warmstart step0 復元） |
| 21:24 | Step0 ループ成立、学習開始 |
| 22:21 | total steps: 2,000（test_play 開始） |
| **22:49** | **step2000 test_play 完了** |
| 22:49 | `run_online_diag.sh` が DONE 記録（副露 83.81%） |
| 23:54 – 01:00 | **意図せず学習継続**（cleanup が train.py 子プロセスを残した）。step4000/6000 test_play も実行 |

---

## 結果 ★

### step2000 test_play（診断A の主結果）

| 指標 | 診断A (lambda_opp=0) | online_main (lambda_opp=0.3) | phase4d lo=0.3 warmstart |
|---:|---:|---:|---:|
| **副露率** | **83.81%** | **61.25%** | **17.59%** |
| 鳴き和了率（赤保持局） | — ※ | 7.24% | 3.03% |
| 立直率 | **7.95%** | — | 23.49% |
| 和了率 | 21.13% | — | 20.63% |
| 放銃率 | 15.76% | — | 13.26% |
| 流局率 | 16.36% | — | 18.35% |
| beta_sel | **≈0.0** | **0.0** | — |

※ 診断A step2000 では aka_held_call_win_rate の TB 抽出は未実施。trainer.log の behavior 行のみ。

出典（診断A）:

```
2026-06-26 22:49:12  test_play behavior: agari=21.13% houjuu=15.76% fuuro=83.81% riichi=7.95% ryukyoku=16.36%
```

出典（online_main step2000）: `freeparlor/docs/online_fuuro50_aka_selectivity.md`（TB `test_play_behavior_fuuro`）

---

## 判定

| 事前仮説 | 結果 |
|---|---|
| lambda_opp=0 なら 61% に跳ね**ない** → 主因 lambda_opp | **棄却** |
| lambda_opp=0 でも跳ねる → 主因は別 | **支持** |

**数値:** lambda_opp=0 でも step2000 副露 **83.81%**（online_main 61.25% **より高い**）。  
warmstart 17% からの跳升は lambda_opp **無し**でも発生。

---

## 異常値メモ（step2000 の 83.81%）

step2000 の行動プロファイルが online_main / 意図せず継続した後段 test_play と大きく乖離:

| step | lambda_opp | beta_sel | 副露率 | 立直率 |
|---:|---|---:|---:|---:|
| 2000（主結果） | 0.0 | ≈0 | **83.81%** | **7.95%** |
| 4000（継続分・参考） | 0.0 | 0.3 | 61.15% | 20.77% |
| 6000（継続分・参考） | 0.0 | 0.3 | 65.46% | 18.80% |
| online_main 2000 | 0.3 | 0.0 | 61.25% | — |

- step2000 の **立直 7.95%** は warmstart（23%）・online_main 帯と比べ極端に低い。
- 83.81% が test_play 実装・ckpt タイミングの一過性異常か、lambda_opp=0 固有の学習崩れかは **未確定**。
- ただし「lambda_opp=0 なら副露が抑制される」方向の結果は **出ていない**（17% 帯には戻らない）。

---

## インフラ上の問題

| 問題 | 内容 | 対処 |
|---|---|---|
| 停止漏れ | `run_online_diag.sh` cleanup が `train.py` 子プロセスを kill できず step6000+ まで継続 | cleanup に `pkill mortal/train.py` 追加（診断B 前に修正済み） |
| step 判定 | 初版は test_play 後の step 行で DONE 判定 → タイミングずれ | test_play 直前 step で判定するよう修正 |

---

## 次ステップ

| 項目 | 状態 |
|---|---|
| 診断B | **完了**（副露 13.61%）→ `online_diag_b_cql_weak.md` |
| 総合報告 | `online_diag_fuuro_summary.md` |
| 本番 online 再開 | **未実施**（原因特定優先） |
| step2000 / 83.81% の再現確認 | 未実施（必要なら test_play 再生成） |

---

## 集計メタ

| 項目 | パス |
|---|---|
| config | `/home/gamba/mahjong/runs/online_diag_a/config.toml` |
| trainer ログ | `/home/gamba/mahjong/runs/online_diag_a/logs/trainer.log` |
| step2000 結果 | `/home/gamba/mahjong/runs/online_diag_a/result_step2000.txt` |
| 起動スクリプト | `/home/gamba/mahjong/runs/run_online_diag.sh` |
| 参照（online_main 時系列） | `freeparlor/docs/online_fuuro50_aka_selectivity.md` |
| 参照（本番設定） | `freeparlor/docs/online_main_report.md` |

**実施内容:** 診断A のみ。lambda_opp と cql は同時変更していない。
