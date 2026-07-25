# DRCA 本測定・第2枠（セット(a)×init）単枠集計 + 集計キー衝突の修正

**日付:** 2026-07-25
**種別:** 単枠集計（第1枠の前例踏襲）+ ハーネス修正1件。**判定は全枠揃い後・監督側**
（`../design/drca_probe_design.md` §5a の凍結解釈条件は不変更）
**run:** `/home/gamba/mahjong/runs/drca/main_a_init_20260721_184910`（probe 7760/7760 完走、
divergence/assert/illegal_fallback 0。集計は CPU のみ、進行中の第3枠に非干渉）

---

## 1. 単枠結果（PRIMARY、`aggregate_final.json` に保存済み）

| 指標 | 値 |
|---|---:|
| n_branch_points | 485 / 485（`--expect-branch-points 485` PASS） |
| **ΔQ̄（n加重平均）** | **−2.1353 千点** |
| cluster-robust SE | 0.4788 |
| \|ΔQ̄\|/SE | **4.459** |
| 符号検定 | 214+ / 270− / 1◦（two-sided exact p=0.0123） |

exploratory 層別（§5a-5、判定非使用）: chi −3.14±0.65 / pon −0.83±0.70(n.s.) /
**kan +6.41±3.52（n=12）**、聴牌時 −7.05±2.43、early −2.73 / late −0.62。

**参照値（第1枠 Stage1-16000）:** ΔQ̄ = −3.2635±0.7208（4.53SE）。両枠の差 ≈ +1.13
（SE(差) ≈ 0.87、~1.3SE、n.s.。ただし採取 seed 母集団が異なるため厳密比較不可 —
第1枠は pilot 20260713–14 + 追加 20260715 系、第2枠以降は §5a-1b launcher の
seed 基点 20260713 で枠間同一母集団）。

**解釈メモ（判定ではない）:** init は基礎技能が無傷の人間譜由来方策
（`fundamentals_degradation_diagnosis_20260725.md` §3a で全 PPO checkpoint に勝つ側）だが、
その init でも赤保持鳴き機会の平均反実仮想価値は有意に負。反鳴き均衡が
「PPO 方策の基礎劣化アーティファクト」である可能性は、これで上界が引けた
（符号は健全な牌理の下でも変わらない）。正式な解釈割り当ては §4/§5a-2 に基づき全枠後。

## 2. 発見: 集計グループ化キーの衝突（安全弁が正しく発火）

初回集計で `--expect-branch-points 485` が FATAL 発火（484/484）。調査の結果、
**採取の重複ではなく集計器のキー不足**と特定:

- bp.jsonl の2エントリ — `(game_key=20260720_1146241857_c, champion, seq=129)` が同一 —
  は**別々の実分岐点**（東7局 seat1 チー機会・聴牌 vs 東8局 seat0 ポン/カン機会・3向聴）。
- 原因は既知の構造（2度目の差し戻し修正 9fea7a7 で文書化済み）: champion 役は
  1 game 3 slot、各 slot の seq が独立 0 起算のため `(game_key, 'champion', seq)` は
  slot 間で衝突し得る。sidecar 消費側は digest 照合で解決済みだったが、
  **`drca_aggregate.py` の `branch_key` と `drca_run_probe.py` の `branch_identity`
  （--resume 用）だけが 3-tuple のまま**だった。
- 影響: 集計では2分岐点が1クラスタに併合（n加重平均 ΔQ̄ は**不変**、SE 0.4783→0.4788・
  符号検定 213/270/1→214/270/1 の微差のみ）。--resume では衝突ペアが「2K+2K=不完全」と
  誤判定され破棄・再走される非効率（データ破損なし。第2枠は resume 未使用のため実害なし）。

**修正（本 commit）:** 両関数のキーに `seat` を追加した 4-tuple へ
（slot⇔物理席が 1:1・slot 内 seq 一意なので証明可能に一意。旧スキーマ行は
`get('seat', -1)` でソート可能性を維持）。凍結解釈条件・分岐点定義・rollout データは
一切不変更 — 識別子の正確化のみ。

## 3. 検証

- **第1枠回帰:** pilot 800行に role/seq を bp.jsonl から補完（join キー
  game_key×seat×kyoku×turn×shanten、一意性 assert）+ 3 shard を新キーで再集計 →
  **commit 済み `aggregate_final.json` と完全一致**（485点、ΔQ̄ −3.2635、SE 0.7208、
  符号 219/255/11 p=0.1078）。衝突なしデータへの無影響を実証。
- **第2枠:** 新キーで 485/485、安全弁 PASS、`aggregate_final.json` 保存。
- **第3枠（進行中 `main_a_s3final_20260725_120218`）:** bp.jsonl 485行/485ユニーク =
  衝突なし。進行中 probe プロセスは本修正の影響を受けない（ForkState は元々
  (role,seq) バケット + digest 照合で衝突安全。ファイル編集は実行中プロセスに非干渉）。

## 4. 残作業（監督側）

- 全5枠…改め実効枠（§5a-1b 適用後）揃い後の正式判定。第2枠の数値は本書 §1 を転記可。
- 第1枠 aggregate_final.json は旧 3-tuple キー産だが衝突なしのため数値有効
  （§3 回帰で確認済み）。再生成は不要。
