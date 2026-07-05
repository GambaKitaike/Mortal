# CLAUDE.md — Mortal フリー雀荘 PPO プロジェクト

## プロジェクト概要

Mortal をフリー雀荘ルール（素点+ウマオカ+チップ、β=1）向けに再設計する調査基盤。
現在は教師データ非依存・自己対戦 PPO への移行フェーズ（Stage1 本走中）。

**最初に読む文書（この順で）:**
1. `freeparlor/docs/ppo_migration_design.md` — PPO 移行の設計正典
2. `freeparlor/docs/ppo_p3_stage1.md` — 現在の run 状態・インシデント史
3. `freeparlor/docs/reward_design_teacherfree.md` — 報酬設計の確定事項
4. `freeparlor/docs/next_steps_2.md` — プロジェクト全体史

## 環境

- 作業は `wsl -d mahjong` のみ。ユーザー `gamba`、リポジトリ `/home/gamba/mahjong/Mortal`
- `conda activate mortal`。学習は `runs/` の spawn ランチャ経由（CUDA fork 回避）
- libriichi 改修後は必ず: `cargo build --release -p libriichi --lib` →
  `cp -f target/release/libriichi.so mortal/libriichi.so`
  （※ preflight スクリプトが毎発進時に自動 rebuild する。手動ビルドを信用しない）
- **GPU ワークロードは常に1系統**。学習と eval の同時実行禁止
- メモリ: WSL 24GB 上限。学習 run は tmux 内で起動（切断耐性）

## ワークフロー規律（違反すると run が無効になる）

### タスク完了の定義
- **commit & push まで完了してタスク**。push 後に
  `git ls-remote origin | grep <branch>` を実行し、リモート先端がタスクの
  commit hash と一致する出力を貼って報告する。push されていない作業は未完了
- ブランチ: PPO 関連は `ppo-migration`。main の DQN 経路は触らない

### run の規約
- run dir は日時 suffix 必須（例 `stage1_20260705_053301`）。**再利用禁止**
- 中止した run は削除せず `aborted<N>_` として保全（証拠保存）
- 発進前 preflight: 残党チェック（`pkill` + `ss -tlnp | grep 5000`）+
  libriichi rebuild + 全検定（`freeparlor/scripts/verify_ppo_p1.py`、現在16本）PASS
- 発進後: 開始報告（config 全文 + 監視項目 + step 100 到達 + alive clients 3/3）
- **凍結ルール**: 事前登録した run は判定窓が閉じるまでコード・config 変更禁止。
  例外はクラッシュとデータ整合性の破れのみ。「気になる挙動」は記録して続行

### 監視期待値（1件でも非ゼロなら報告）
- `trajectory step count mismatch` = 0（必須）
- `illegal_action_fallback_count` = 0（必須）
- `online chip resolution failed` = 0（必須）
- `loader size delta` = INFO（非致命・報告のみ）
- alive clients = 3/3、step 到達性（停滞は異常）

### 実装の禁則
- **サイレント修正・サイレントフォールバック禁止**。解決不能は例外で大声で落とすか、
  カウンタ+WARNING で可視化する（zeros 埋め・黙って skip は過去に run を3本殺した）
- **訓練 rollout への行動上書き禁止**（rule-based guard 等）。eval は本家準拠で guard ON
- 訓練 client は必ず π からの純サンプリング（greedy/top_p 混合禁止）
- 新規 Rust 表面積は最小に。本家挙動に戻せるならリバートを優先
- 検定・診断の self-play client は**本番 client と同一構成**（構成 dump diff==空）

### 実験の規律
- 単一変数アブレーション優先。GPU を焼く前に設計文書を commit
- 判定条件は run 前に固定し、結果を見てから変更しない（post-hoc goalpost 禁止）
- 挙動の評価は2レンズ併記: argmax eval（配備挙動）と sampled action_mass（学習方向）
- 400 step 級スモークで挙動の結論を出さない（分散が支配する。配管検証のみ）

## 現在の状態（2026-07-06 時点）

- run #7b **走行中**（`stage1_20260706_020120_resume`、step 10001 から再開）
- **凍結中**: step 16000 完走まで コード・config 変更禁止（2026-07-06 02:52 JST 凍結宣言）
- 再開詳細: `freeparlor/docs/ppo_p3_pause_resume.md` / `ppo_p3_stage1.md` §1g
- 完走後: eval バッテリー + 判定窓集計（別タスクとして指示される）
- 判定窓 8000–16000 は run7a(8000–10000) + run7b(10000–16000) を global step で連結

## 役割分担

設計判断・レビュー・仮説の裁定は Claude（chat 側）が担当。このリポジトリでの
あなた（Claude Code）の役割は実装・検証・commit・push。設計変更が必要だと
感じたら、実装せずに提案として報告する。
