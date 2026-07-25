# 基礎劣化の有意性パス — 診断レポート §3(a) への cluster-robust SE 付与

**日付:** 2026-07-25
**種別:** 確証プローブ（`fundamentals_degradation_diagnosis_20260725.md` §7 の CPU 実行可能部分）。
既存 1v3 eval 牌譜の再走査のみ・GPU 不要・学習コード/eval 成果物無変更
**スクリプト:** `freeparlor/scripts/analyze_fundamentals_1v3.py`（read-only。イベント解釈・
最終スコア再構成は `analyze_freeparlor_pnl_1v3.py` を import — ロジック複製禁止準拠）

---

## 1. 結果（各 400 半荘、diff = step16000 − init、z = diff/SE）

| stage | agari 差 | z | houjuu 差 | z | avg_rank 差 | z |
|---|---:|---:|---:|---:|---:|---:|
| Stage1-16000 | −1.91pp | **−2.13** | +3.06pp | **+3.93** | +0.058 | +0.71 (n.s.) |
| Stage2-16000 | −3.07pp | **−3.55** | +3.32pp | **+4.39** | +0.230 | **+2.90** |
| Stage3-16000 | −1.87pp | **−2.09** | +3.08pp | **+4.12** | +0.105 | +1.30 (n.s.) |

**読み（2SE 基準）:**
- **放銃率の劣化は全 stage で強く有意**（z +3.9〜+4.4）。和了率の低下も全 stage 有意
  （Stage1/3 は 2SE ぎりぎり、Stage2 は明確）。診断レポートの中核主張
  「PPO は init の基礎（守備・和了）を劣化させる」は統計的に確定。
- **avg_rank の悪化が有意なのは Stage2 のみ**（= 既知の配備税）。Stage1/3 の
  着順悪化は n.s. — 診断 §3(a) の「平均着順も init に届かない」は方向の記述としては
  正しいが、有意な主張として使えるのは放銃・和了の劣化。0b 資料にはこの精緻化を反映すべき
  （Stage1 の合算収支が n.s. で +方向だった事実 — stage1 result §7 表2 — と合わせ、
  正確な像は「基礎を有意に支払い、チップ獲得で相殺してほぼ損益分岐」）。

SE は半荘クラスタの ratio-estimator（率）/ 半荘平均 SE（着順）、差は独立2標本近似。
両脚は同一 seed 集合 [10000,10100) で配牌を共有するため正の相関があり、この近似は
**保守的**（有意な結果はペア化するとさらに強くなる方向）。committed な配備税チェック
（`ppo_p3_stage3_result.md`）と同一の近似。

## 2. 検証

- **パーサ検証:** pooled 率・avg_rank が commit 済み各 stage result の Stat 由来値と
  完全一致（init 20.67%/12.10%/2.4750、Stage1 18.76%/15.15%/2.5325、
  Stage2 17.61%/15.42%/2.7050、Stage3 18.80%/15.18%/2.5800）。
- **init 脚の決定論一致:** stage2 eval 時と stage3 eval 時の game_logs_init は
  全指標 diff = 0.000（同一 seed・同一 checkpoint の決定論的同一測定であることを実証）。

## 3. 発見: stage1 の init 脚牌譜が消失している

`stage1_20260706_020120_resume/logs/eval_grp_baseline/game_logs_init/` が**空**
（step16000 脚 400 件は健在。brokenfixture 保全 dir には init 脚があるが無効 run）。
eval 牌譜は `run_artifact_retention.md` で恒久保全クラスのため、消失経路は不明
（`eval_grp_baseline_1v3.py:180-182` は再実行時に同 label dir を rmtree する仕様 —
中断された再実行が候補）。**実害なし**: init 脚は決定論的同一測定が stage2/3 の run dir に
2部現存（§2 で一致実証済み）。本書の Stage1 行はこれを init 参照に使用した。
