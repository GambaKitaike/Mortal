# freeparlor/ — フリー雀荘ルール調査基盤の作業ディレクトリ

本家 [Mortal](https://github.com/Equim-chan/Mortal) からの**差分のうち、学習コード本体
（`mortal/`・`libriichi/`）に属さないもの**を全てここに置く。設計文書・事前登録された判定条件・
run config・発進ランチャ・検定/集計/評価スクリプトが対象。

プロジェクト全体の趣旨・現在地はリポジトリルートの [`README.md`](../README.md)、
作業規律と進行中 run の状態は [`CLAUDE.md`](../CLAUDE.md) を参照。

---

## ディレクトリ構成

| パス | 内容 |
|---|---|
| `docs/design/` | 設計正典・**事前登録された判定条件**（8本）。凍結 commit が事前登録の実体で、以後の変更は禁止（例外は明示的 amendment のみ） |
| `docs/reports/` | 判定結果・run 記録・診断レポート・定性レビュー（18本） |
| `docs/ops/` | 運用文書（ハンドブック・成果物保持運用・実装タスクプロンプト・引き継ぎメモ、8本） |
| `docs/archive/dqn/` | オフライン DQN + CQL 時代（2026-06〜07-02）の完結文書（36本）。現行 PPO 本線の判断には使わない |
| `docs/INDEX.md` | 上記 docs の索引（パス・日付・ステータス・要約） |
| `configs/` | 現行 run の config（10本）。`archive/` は DQN 時代の phase config（25本） |
| `scripts/` | 発進ランチャ（`.sh` 29本）+ 検定・集計・評価・診断（`.py` 43本）。`scripts/INDEX.md` に分類索引 |
| `experiments/` | 未使用（プレースホルダ） |

学習の run 成果物（checkpoints / logs / drain / 牌譜）はリポジトリ外の
`/home/gamba/mahjong/runs/` に出る。保持・清掃のポリシーは
[`docs/ops/run_artifact_retention.md`](docs/ops/run_artifact_retention.md)。

---

## 最初に読む文書

1. [`docs/design/ppo_migration_design.md`](docs/design/ppo_migration_design.md) — PPO 移行の設計正典
2. [`docs/design/reward_design_teacherfree.md`](docs/design/reward_design_teacherfree.md) — 報酬設計の確定事項
3. [`docs/reports/ppo_p3_stage1_result.md`](docs/reports/ppo_p3_stage1_result.md) —
   探索ラダー Stage1 判定（立直マキシマリズムの発見）
4. [`docs/design/anchored_ppo_design.md`](docs/design/anchored_ppo_design.md) —
   現行の優先軸（アンカー付き PPO、凍結済み）
5. [`docs/ops/supervisor_handbook.md`](docs/ops/supervisor_handbook.md) —
   確定知見・殺した仮説リスト・出力規約

---

## 実行の作法（要点のみ）

- **GPU ワークロードは常に1系統。** 学習と eval の同時実行は禁止、eval バッテリーも直列。
- run 発進は `scripts/run_ppo_*.sh` 経由（`runs/` の spawn ランチャを exec する）。
  発進前 preflight（残党チェック・libriichi rebuild・`scripts/verify_ppo_p1.py` 全検定 PASS・
  ディスク空き容量）はランチャが自動実行する。
- run dir は日時 suffix 必須・**再利用禁止**。中止した run は削除せず `abortedN_` として保全。
- 事前登録した run は判定窓が閉じるまでコード・config 変更禁止（凍結）。
  例外はクラッシュとデータ整合性の破れのみ。

詳細と背景（過去に run を殺した事故の記録を含む）は `CLAUDE.md`「ワークフロー規律」節。
