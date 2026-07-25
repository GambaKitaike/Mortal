# 実装タスクプロンプト: anchor 系列（アンカー付き PPO）Arm C + Arm K

**日付:** 2026-07-25
**宛先:** 実装エージェント（Cursor Composer / Sonnet）
**正となる仕様:** `freeparlor/docs/design/anchored_ppo_design.md`（**凍結済み・変更禁止**。
本プロンプトと食い違ったら設計書が正。設計変更が必要だと感じたら実装せず提案として報告）
**規律:** `CLAUDE.md` のワークフロー規律全部（commit&push+ls-remote 貼付で完了、
サイレントフォールバック禁止、訓練 rollout への行動上書き禁止、検定 self-play は
本番同構成、Rust 表面積ゼロ）

## 0. 絶対条件

- **libriichi（Rust）無変更。** 変更対象は `mortal/*.py`・`freeparlor/configs/`・
  `freeparlor/scripts/` のみ
- **両介入ともデフォルト OFF = 既存経路ビット不変**（`anchor_prob=0.0`/キー不在、
  `kl_beta=0.0`/キー不在で、既存 run と bit-identical な挙動。検定で担保する）
- **DRCA 第3枠が GPU で走行中**（`main_a_s3final_20260725_120218`、~07-29 まで）。
  GPU 1系統ルールにより、**GPU を使う検証（verify_ppo_p1.py フル実行・スモーク）は
  第3枠完走後の発進 preflight に統合**する。実装段階では GPU 非依存検証
  （bash -n・合成テンソルでの単体検証・OFF ビット同一性の CPU 実行)のみ行うこと。
  新設する検定 (19)(20) は **device='cpu' で単体実行可能な設計**にする
- ブランチは `ppo-migration`。commit は Arm C と Arm K で分けてよい（推奨）

## 1. Arm C — opponent pool への凍結 init 常駐

1. `mortal/opponent_pool.py` `OpponentPool.sample()`（L36-46）に anchor 分岐:
   `random.random() < self.anchor_prob` なら `self.anchor_checkpoint` を返す、
   それ以外は従来ロジックへフォールスルー。コンストラクタに
   `anchor_prob: float = 0.0` / `anchor_checkpoint: str|Path|None = None` を追加
2. `mortal/config.py`（L42 付近の opponent_pool デフォルト）に
   `anchor_prob: 0.0` / `anchor_checkpoint: ''` を追加。
   `mortal/player.py`（L245 付近）で config → OpponentPool へ配線
3. **loud FAIL**: `anchor_prob > 0` かつ anchor_checkpoint が空/非実在なら
   起動時に例外で落とす（黙って従来ロジックに退化させない）
4. **draw ログ**: anchor/latest/past のどれを引いたかを機械可読に記録する
   （発進ゲート @step200 で「anchor 採択率 = 0.25±0.05」を検証するため。
   既存のログ基盤に合わせて形式は任せるが、run dir の成果物から
   採択率を再計算できること・形式をゲートスクリプトの docstring に明記すること）
5. 構成 dump（`ppo_engine.dump_engine_config`）に `anchor_prob`/`anchor_checkpoint`
   を追加（p_enrich/call_bonus_b と同じ getattr デフォルトパターン。
   eval 経路は属性未設定 = 0.0/'' になること）

## 2. Arm K — 凍結 init への masked full KL 項

1. `mortal/ppo.py` `ppo_loss`（L79-109）にオプション引数
   `kl_beta: float = 0.0` / `ref_logits: Tensor|None = None` を追加:
   - `kl_beta == 0.0` または `ref_logits is None` → **一切の追加計算をせず**
     返り値 dict も現行とキー・値ともに完全一致（OFF ビット不変）
   - ON 時: 合法手 mask 上の full KL
     `KL = Σ_legal p_θ(a)·(log p_θ(a) − log p_ref(a))` をバッチ平均し、
     `total += kl_beta * kl_ref`、dict に `'kl_ref'` を追加。
     masked softmax/log-prob は既存ヘルパを再利用（数値安定化含め複製しない）
2. `mortal/train_ppo.py`:
   - `kl_beta > 0` のときのみ参照モデルをロード（Brain+ActorCritic を
     `kl_ref_checkpoint`（デフォルト空 = `init_checkpoint` 流用）から
     `load_ppo_from_mortal_checkpoint` 経路で構築 = **PPO step 0 の方策と同一**）。
     全パラメータ `requires_grad_(False)` + `.eval()`。`kl_beta == 0.0` では
     **ロード自体をしない**
   - ref forward はロールアウトバッチ受領時に **epoch ループの外で1回だけ**
     no_grad で計算（minibatch サイズにチャンクして OOM 回避）、
     他の `*_all` テンソルと同様に保持し minibatch でスライスして `ppo_loss` へ
   - diag: 毎バッチ `kl_anchor` イベント（`kl_beta` / `kl_ref_mean` /
     `kl_term_total`）を分離ログ（Stage3 `call_bonus` イベントの実装パターン踏襲）
3. config: `[ppo] kl_beta = 0.0`（デフォルト）/ `kl_ref_checkpoint = ''`。
   構成 dump に `kl_beta` を追加（同上 getattr パターン）
4. **loud FAIL**: `kl_beta > 0` かつ ref checkpoint 非実在なら起動時に例外

## 3. 検定の拡張（`freeparlor/scripts/verify_ppo_p1.py`、設計書 §4）

- **(19) anchor pool**: (a) `anchor_prob=0.0`/キー不在で `sample()` の返り値系列が
  従来実装とシード固定で完全一致（OFF 恒等性）、(b) `anchor_prob=1.0` で常に
  anchor_checkpoint、(c) 中間値で採択率がバイナリアル妥当域
- **(20) KL 項**: (a) `kl_beta=0.0`/キー不在で loss dict が既存実装とビット一致 +
  ref 未ロードの確認、(b) 合成 logits/mask での masked KL の手計算一致
  （境界: mask 1本のみ・ref=π で KL=0）、(c) β>0 で ref パラメータに grad が
  流れない（`requires_grad`/`.grad is None` assert）
- 既存 **17d 同居分**: eval 構成 dump assert に `anchor_prob==0.0`・`kl_beta==0.0`
  を追加。既存 eval スクリプト（`eval_ppo_smoke_sanity.py`/`eval_grp_baseline_1v3.py`/
  `eval_meta_stage1_vs_stage2.py` 等）の assert 群にも同2項目を追加
  （call_bonus_b assert 追加時と同じパターン）
- (19)(20) は CPU 単体実行可能にすること（§0）。検定本数のハードコード禁止
  （`ALL N CHECKS PASSED` の N はスクリプトが自分で数える現行方式を維持）

## 4. config / launcher

- `freeparlor/configs/ppo_anchor_c.toml`: Stage1 config との diff が
  **run パス（`anchor_c_PENDING_LAUNCH` プレースホルダ）+ [opponent_pool] の
  anchor 2キーのみ**であることを diff で確認・報告
- `freeparlor/configs/ppo_anchor_k.toml`: 同じく diff が **run パス
  （`anchor_k_PENDING_LAUNCH`）+ [ppo] kl_beta/kl_ref_checkpoint のみ**
- `freeparlor/scripts/run_ppo_anchor_c.sh` / `run_ppo_anchor_k.sh`:
  `run_ppo_stage3.sh` 踏襲（DISK_MIN_GB preflight・プレースホルダ in-place 解決・
  tmux・preflight_libriichi + verify 全検定）
- 発進ゲートスクリプト `freeparlor/scripts/check_anchor_launch_gate.py`:
  機械ゲートのみ（設計書 §5）。`--arm c` = draw ログから anchor 採択率
  0.25±0.05、`--arm k` = ppo_diag の kl_anchor イベントが全バッチ
  kl_beta=0.1・有限値・NaN なし。@step200

## 5. 検証と報告（監督が3段検証する前提で）

1. GPU 非依存検証を全部実施・証拠を貼る: (19)(20) の CPU 実行ログ、
   OFF ビット同一性（既存検定 (1)–(18) のロジック無変更確認を含む）、
   bash -n、config diff の実出力
2. **変更していないものを明示列挙**: libriichi、client.py rollout 経路、
   報酬3ストリーム、DRCA ハーネス4本、既存検定ロジック、進行中の第3枠 run
3. commit & push 後に `git ls-remote origin | grep ppo-migration` の出力を貼付
4. CLAUDE.md「現在の状態」の更新を同一 commit に含める
5. 400-step スモーク（K の配管検査）と verify フル実行は**やらない**
   （第3枠完走後の発進 preflight で監督立ち会いのもと実施 — §0）
