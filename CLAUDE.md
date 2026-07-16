# CLAUDE.md — Mortal フリー雀荘 PPO プロジェクト

## プロジェクト概要

Mortal をフリー雀荘ルール（素点+ウマオカ+チップ、β=1）向けに再設計する調査基盤。
教師データ非依存・自己対戦 PPO。Stage1（純探索）は判定完了、現在 Stage2
（配牌 rejection sampling による赤濃縮）の実装準備フェーズ。

**最初に読む文書（この順で）:**
1. `freeparlor/docs/design/ppo_migration_design.md` — PPO 移行の設計正典
2. `freeparlor/docs/design/stage2_design.md` — 現行 run の設計・事前登録済み判定条件
3. `freeparlor/docs/reports/ppo_p3_stage1_result.md` — Stage1 判定結果（立直マキシマリズム）
4. `freeparlor/docs/reports/ppo_p3_stage1.md` — Stage1 の run 状態・インシデント史
5. `freeparlor/docs/design/reward_design_teacherfree.md` — 報酬設計の確定事項
6. `freeparlor/docs/ops/next_steps_2.md` — プロジェクト全体史

## 環境

- 作業は `wsl -d mahjong` のみ。ユーザー `gamba`、リポジトリ `/home/gamba/mahjong/Mortal`
- `conda activate mortal`。学習は `runs/` の spawn ランチャ経由（CUDA fork 回避）
- libriichi 改修後は必ず以下の手順で（2026-07-08 ビルド事故を受けて明文化。
  `supervisor_handbook.md` §4c 参照）:
  1. `conda activate mortal` を明示的に確認済みであること
  2. `PYO3_PYTHON="$CONDA_PREFIX/bin/python"` を明示的にセット
  3. ビルド前に `CARGO_TARGET_DIR` が空（未設定）であることを確認
     （tmux セッション残留由来の汚染がビルド先とcp元の乖離を起こす）
  4. `cargo build --release -p libriichi --lib` →
     `cp -f target/release/libriichi.so mortal/libriichi.so`
  5. cp 後、import スモーク必須:
     `PYTHONPATH=mortal python -c "from libriichi.stat import Stat"`
  （※ 学習発進・eval バッテリーとも preflight_libriichi.sh が自動実行する。
  手動ビルドを信用しない）
- **GPU ワークロードは常に1系統**。学習と eval の同時実行禁止。eval バッテリーも直列実行
- メモリ: WSL 24GB 上限。学習 run は tmux 内で起動（切断耐性）

## ワークフロー規律（違反すると run が無効になる）

### タスク完了の定義
- **commit & push まで完了してタスク**。push 後に
  `git ls-remote origin | grep <branch>` を実行し、リモート先端がタスクの
  commit hash と一致する出力を貼って報告する。push されていない作業は未完了
- ブランチ: PPO 関連の基底は `ppo-migration`。main の DQN 経路は触らない
- **1 branch = 1 variable**: 実験変数を導入する変更は変数ごとにブランチを分ける。
  単一変数アブレーションの規律をブランチ構造で強制する

### 実行環境の分界線
- 実装エージェントのサンドボックス（Claude Code web 等、GPU なし・ローカル WSL 外）での
  検定PASSは参考値。正はローカル WSL（`mahjong` distro）での preflight 全パス。
  GPU 依存作業（run 発進、eval バッテリー）は常にローカル側で実施する
- サンドボックスで実行可能な検証の範囲（2026-07-07 実地確認）:
  - `cargo build --release -p libriichi --lib`: 実行可能（crates.io へのネットワーク
    アクセスあり、ビルド成功）。libriichi の Rust 変更はサンドボックスでビルド確認まで可
  - `python freeparlor/scripts/verify_ppo_p1.py`: 実行不可（numpy/torch 未導入、
    conda 環境なし）。検定PASSの確認は必ずローカル WSL 側で行う

### run の規約
- run dir は日時 suffix 必須（例 `stage1_20260705_053301`）。**再利用禁止**
- 中止した run は削除せず `aborted<N>_` として保全（証拠保存）
- 発進前 preflight: 残党チェック（`pkill` + `ss -tlnp | grep 5000`）+
  libriichi rebuild + 全検定（`freeparlor/scripts/verify_ppo_p1.py`）PASS。
  **検定の本数はスクリプトの実行結果（`ALL N CHECKS PASSED`）が正**。
  本書に本数をハードコードしない（2026-07-08 時点 17 本。p_enrich assert が
  17 本目 — `stage2_design.md` §3）
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
- Stage2 固有: p_enrich は**訓練 client のみ**。eval 経路は常時 p_enrich=0
  （構成 dump に p_enrich を含め、eval 側 0 を検定で assert — `stage2_design.md` §2）

### 実験の規律
- 単一変数アブレーション優先。GPU を焼く前に設計文書を commit
- 判定条件は run 前に固定し、結果を見てから変更しない（post-hoc goalpost 禁止）
- 挙動の評価は2レンズ併記: argmax eval（配備挙動）と sampled action_mass（学習方向）。
  Stage2 以降は分布にも注意: 訓練測定は濃縮分布上、eval は常に自然分布
  （絶対値の run 跨ぎ比較は不可、倍率同士で比較 — `stage2_design.md` §4）
- 400 step 級スモークで挙動の結論を出さない（分散が支配する。配管検証のみ）

## 現在の状態（2026-07-15 時点）

> **この節は陳腐化する前提。** 正は `stage2_design.md`・`ppo_p3_stage1_result.md`・
> git log。状態を変えるタスクを完了したら、この節の更新も同一 commit に含めること。

- **Stage1 判定完了**（2026-07-06）: 事前固定条件（赤保持鳴き試行率 2 倍未満 かつ
  上昇トレンド無し）が両方成立（0.236×、slope/SE=−6.14）→ Stage2 移行確定
- **Stage2 設計 commit 済み**（`stage2_design.md`、判定条件は事前登録済み・変更禁止）
- **実装エージェント**: Cursor Composer（2026-07-15 復帰。ローカル WSL の GPU・conda 環境・tmux に直接アクセス可）
- **Stage1 argmax eval バッテリー（標準6本）完了**（2026-07-08。結果は
  `ppo_p3_stage1_result.md` §6）
- **Stage1 eval 全完了**（2026-07-08。自己対戦バッテリー §6 + grp_baseline 1v3
  対戦 §7。両方とも `ppo_p3_stage1_result.md` 参照）
- **Stage2 実装完了**（2026-07-08、実装・検定のみ・未発進）: `board.rs` に
  配牌 rejection sampling（`Board::init_from_seed_enriched`）を追加。
  p_enrich を `[ppo]` config → 訓練 client の `PPOEngine`（`build_production_trainee_engine`）
  → `OneVsThree::py_vs_py`（challenger からの getattr）→ `BatchGame`/`Game` →
  `Board` まで配線。eval/opponent 経路は属性未設定＝デフォルト 0.0 で常時無効
  （`ppo_engine.dump_engine_config` に `p_enrich` を含め検定で 0 を assert）。
  検定を 17 本目まで拡張（(a) p_enrich=0.0 で決定論的ゴールデンハッシュ完全一致、
  (b) p_enrich=1.0 で赤保持率100%、(c) 自然率実測 25.44%（172/676局）を
  `stage2_design.md` §2 に記録、(d) eval 構成 dump で p_enrich=0 を assert）。
  `freeparlor/configs/ppo_stage2.toml` 作成済み（run パスは
  `stage2_PENDING_LAUNCH` プレースホルダ、発進時に launcher で置換）。
  ローカル WSL preflight（rebuild + import smoke）+ `verify_ppo_p1.py` 全 17 本
  PASS 確認済み（正式判定）
- **検定(14)修正済み（2026-07-09）**: `check_reward_placement_e2e`
  が trainee 無判断で終わる局（相手の九種九牌等の稀事象クラス）を記録欠落バグと誤判定していた
  問題を修正。当該局は json.gz 再走査で trainee 席の反応イベント 0 件を検証した上で許容し、
  局番号・終局イベント種別をログ出力するよう変更（`supervisor_handbook.md` §4c に新バグクラス
  追記）。**ローカル WSL での `verify_ppo_p1.py` 全 17 検定 PASS 確認済み（2026-07-09）**
- **Stage2 本走 発進・凍結中（2026-07-09 09:25:41 JST 発進）**: run dir
  `/home/gamba/mahjong/runs/ppo/stage2_20260709_092541`、tmux セッション
  `ppo_stage2_20260709_092541`。init = beta1_huber_192x40（相手プールも
  `checkpoints/step_000000.pth` のみ・mortal 重み init と bit-identical を確認済み。
  Stage1 checkpoint 混入なし）。config は `freeparlor/configs/ppo_stage2.toml`
  （プレースホルダを新規 launcher `run_ppo_stage2.sh` が発進時に run パスへ
  in-place 解決。差分は p_enrich=1.0 と run パスのみ — Stage1 config との diff で確認済み）。
  発進前 preflight（残党チェック・libriichi rebuild・`verify_ppo_p1.py` 全17検定
  PASS）はローカル WSL で改めて実施済み。**発進ゲート（step 500、鳴き可能局面中の
  赤保持割合 ≥54%）は 91.93%（n_call_possible_aka_held=15570/n_call_possible=16936、
  `check_stage2_launch_gate.py` で算出）で通過** — Stage1 実測 36% の 2.5 倍超。
  監視項目（mismatch/fallback/chip_err/trainer NaN）は step 1565 時点まで全て 0、
  alive clients 3/3。**凍結宣言済み: step 16000 完走までコード・config 変更禁止**
  （例外はクラッシュとデータ整合性の破れのみ）。**判定窓は step 8000–16000**
  （`stage2_design.md` §4）。完走・判定集計は別タスク
- **インシデント: ディスク枯渇クラッシュ（2026-07-09、~15:43〜19:34 JST の間）**:
  `/`（1TB）が過去の Stage1 本走・中止 run 群（921GB）+ 本 run 自身の `drain/`
  （6.5h で167GB、Stage1 と同じ既知の蓄積パターンであり Stage2/p_enrich 固有の
  新規バグではない — `drain/` はステップごとにサブディレクトリが作られ run 中は
  一切クリーンアップされない設計。`server.py` は起動時にのみ rmtree する）で
  100%枯渇。server が `OSError: ENOSPC` → trainer が `drain()` 中に
  `UnexpectedEOF` でクラッシュ → tmux サーバー自体も道連れで消滅（`/tmp` も
  同一 `/` 上）。**データ整合性は無傷**: 最終保存 checkpoint `step_006000.pth`
  （15:43 保存）は正常。ユーザー承認を得た上で (a) `ppo_p3_aborted*`/`smoke_p2*`
  系の中止・スモーク run を削除（339GB解放、`stage1_20260705_053301`・
  `stage1_20260706_020120_resume` は判定根拠として温存）、(b) クラッシュ run
  自身の `drain/`（判定に不要な既消費生データ、167GB）のみ削除・checkpoints/
  logs/config は保全、で計504GB復旧（残り10000 step分の推定必要量~230GBに
  対し十分な余裕）。新規 `run_ppo_stage2_resume.sh`（`run_ppo_p3_resume.sh` 踏襲）
  で `step_006000.pth` から新 run dir
  `/home/gamba/mahjong/runs/ppo/stage2_20260709_194510_resume`（tmux
  `ppo_stage2_20260709_194510_resume`）へ resume 発進（19:45:17 JST）。
  p_enrich=1.0 等の単一変数は不変・run パスのみ変更。resume 後 preflight
  （rebuild + 検定17本 PASS）実施済み、step 6960 時点で監視項目
  （mismatch/fallback/chip_err/trainer NaN）全て 0、alive clients 3/3、
  disk 使用率 50%（482GB空き）で健全継続中。**恒久対策は未実施**（run 完走後の
  別タスクとして: (1) 完走 run の `drain/` を判定集計後に削除する運用の明文化、
  (2) 古い aborted/smoke run の定期清掃、のいずれかが必要 — 次回も同じ蓄積で
  枯渇し得る）
- run 7a/7b の checkpoints・ppo_diag.jsonl・tb は保全済み（プロンプト①で実在確認）
- **Stage2 完走確認 + eval バッテリー(3レンズ)完了**（2026-07-10、コード変更なし・
  eval実行のみ）: resume run は step 16000 で正常終了（exit=0、mismatch/fallback/
  chip errors/trainer NaN 全て 0）、checkpoints 8000–16000・ppo_diag.jsonl・tb 保全
  確認済み。標準argmax eval（6 checkpoint）、grp_baseline 1v3（init/step16000）、
  Stage2 vs Stage1 メタ対決 probe（新規ハーネス）を実行。全 eval 経路で p_enrich=0
  を assert 確認。結果は `ppo_p3_stage2_result.md` に eval 節のみ commit
  （判定窓 step 8000–16000 の判定は未実施、設計監督側の別タスク）。
  eval 直前に旧 run の残党プロセス（server/client、trainer 正常終了後も8時間
  放置されていた）を検出・対処（詳細は同 md §0）。新規 eval スクリプト4本
  追加、既存 `eval_ppo_smoke_sanity.py`/`eval_grp_baseline_1v3.py` に診断用の
  p_enrich=0 dump/assert を追加（学習コード・config は無変更）
- **Stage2 判定完了（2026-07-11、設計監督側起草・コード変更なし）**: 事前固定条件
  （`stage2_design.md` §4）への照合により**分岐2 成立**（倍率 0.185× < 2.0×、
  slope/SE=−2.56 で上昇トレンド無し）→ **機会費用仮説を支持、Stage2（配牌赤濃縮に
  よる分布介入）は失敗**。遭遇機会は設計通り 2.6 倍に増加したが試行率は Stage1
  （0.236×）より深く沈んだ（遭遇不足の単独犯説は棄却）。濃縮された赤供給は鳴きより
  立直への誘因として強く働き機会費用ギャップが拡大（7a）、加えて濃縮分布育ちの方策は
  自然分布への配備で有意な性能損失を出す「配備税」を新たに確認（7c、Stage1 版には
  なし）。判定・発見（§6–§10）は `ppo_p3_stage2_result.md` に追記済み、集計スクリプト
  `freeparlor/scripts/aggregate_stage2_judgment.py` /
  `aggregate_stage2_secondary.py` を commit（判定集計の再現性確保）。
  事前登録に従い **Stage3（構造化探索）解封の議論を解禁**（次アクションは下記）
- **Stage3 実装完了**（2026-07-11、実装・検定のみ・**未発進**）: anneal 付き
  per-decision 鳴きボーナス（`stage3_design.md` §2/§7 準拠）。`mortal/ppo.py` に
  純関数 `call_bonus_coeff`（0–full_until 一定 → zero_at まで線形 anneal → 0、
  b=0.0 で恒等 0）と `apply_call_bonus`（sel = 鳴き可能∧赤保持∧鳴き実行、定義は
  既存計装と同一。b_now=0.0 では入力テンソルをそのまま返す = OFF 時ビット不変）を
  追加し、鳴き系定数（CALL_ACTION_MIN/MAX・RIICHI_ACTION・AKA_OBS_ROWS）を
  `train_ppo.py` から `ppo.py` へ移設（値不変・定義の単一ソース化）。
  `train_ppo.py` は `rewards` 転送直後（GAE 計算前）にボーナスを加算
  （GAE/returns/advantage はボーナス込み — 仕様）し、毎バッチ `call_bonus`
  イベント（b / n_applied / bonus_total）を diag へ分離ログ。正典3ストリーム
  （reward_sotensu/grp/chip）は非汚染のまま。config キー不在時のデフォルト
  0.0/0/0 は「設計された OFF」。`dump_engine_config` に `call_bonus_b`
  （getattr フォールバック、エンジンには属性を設定しない = 常時 0.0）を追加。
  検定 (18) `check_call_bonus` を追加（(a) OFF 不変性 + キー不在デフォルト恒等 0、
  (b) 4クラス合成バッチでの配置の正確性 + schedule 境界値、(c) eval 構成 dump の
  call_bonus_b=0 assert は検定 17d に同居）。`freeparlor/configs/ppo_stage3.toml`
  （stage1 config との diff は run パス / p_enrich=0.0 / call_bonus_* 3キーのみ、
  run パスは `stage3_PENDING_LAUNCH` プレースホルダ）+ `run_ppo_stage3.sh`
  （`run_ppo_stage2.sh` 踏襲）作成。libriichi・client.py rollout 経路・Stage1/2
  成果物は無変更。**ローカル WSL で `verify_ppo_p1.py` 全 18 検定 PASS 確認済み
  （2026-07-11）**。発進・drain 清掃・preflight ディスク検査（`stage3_design.md`
  §8）はそれぞれ別タスクで未着手
- **Stage3 設計確定・commit 済み（2026-07-11、設計セッション完了・コード変更なし）**:
  `freeparlor/docs/design/stage3_design.md` を commit（判定条件・発進ゲート・
  スケジュールは事前登録済み・変更禁止）。骨子: anneal 付き per-decision 鳴きボーナス
  （鳴き可能∧赤保持で鳴き実行した decision step に b=5.0 を加算、千点単位=1チップ相当）、
  init から自然分布（p_enrich=0）で 16k steps、b は 0–4k 一定 → 4k–8k 線形 anneal → 0、
  **判定窓 8k–16k（正典報酬のみの区間）**。切り分け対象は競技力ギャップ仮説 vs
  本質的機会費用仮説（`stage3_design.md` §0）。実装（ボーナス配線 + 検定3本追加 +
  config/launcher — 同 §7）、drain 清掃（発進の必須前提 — 同 §8）、発進は
  それぞれ別タスクで未着手
- **Stage3 発進前提の drain 清掃 + preflight ディスク検査 完了**（2026-07-11、
  ops作業・学習コード変更なし）: `stage3_design.md` §8 に従い、判定完了済み4run
  （`stage1_20260705_053301`・`stage1_20260706_020120_resume`・
  `stage2_20260709_092541`・`stage2_20260709_194510_resume`）の `drain/` を
  Gamba 承認の下で削除（削除前 du: 267GB/129GB/4KB/217GB、計 ~613GB）。
  削除前後で `df -h /` は 251GB → 863GB 空きに回復。各 run の `checkpoints/`・
  `logs/ppo_diag.jsonl`・config は削除後も実在確認済み（許可リスト外のパスは
  一切削除せず）。`run_ppo_stage3.sh` に発進 preflight のディスク検査を追加
  （プレースホルダ置換の前に `/home/gamba/mahjong/runs` の空き容量を検査し、
  `DISK_MIN_GB`（env、デフォルト 450GB）未満なら FATAL exit 1、config は無変更のまま
  終了 — 2026-07-09 ディスク枯渇インシデントの再発防止）。
  `DISK_MIN_GB=999999` で FATAL 発火（exit 1）・config プレースホルダ
  未消費（`grep` で確認）を実演済み。バックログ 1b 消化
- **verify スイートの発進前安定化**（2026-07-11、`verify_ppo_p1.py` のみ変更・
  学習コード/config/launcher/検定(1)–(14)(16)–(18)ロジック無変更）: 検定(15)
  `check_daiminkan_direct_path` が torch RNG 未シードにより約1/5で
  「no game with trainee daiminkan」失敗する flake（commit cfcb015 で観測記録）を、
  上限付き決定論的シード走査（`range(20)`、各シードで `torch.manual_seed(s)` →
  既存 self-play → trainee daiminkan がヒットした最初のシードで既存の検証本体を実施、
  ヒットしたシードをログ出力、20 シード全滅時は従来どおり大声で FAIL）で解消。
  検定の意味（daiminkan 直接経路の報酬配置検証）は不変。単体連続10回実行で
  10/10 PASS 確認済み（ヒットシードは 0/0/0/0/0/0/0/2/1/0 と分散、flake 解消を実証）。
  加えて `main()` 末尾のログ出力先を docs 再編（97a0a45）前の陳腐化パス
  `freeparlor/docs/ppo_p1_verify_log.txt` から `freeparlor/docs/reports/
  ppo_p1_verify_log.txt` に修正、旧パスの untracked ファイルを削除。
  **ローカル WSL で `verify_ppo_p1.py` 全 18 検定 PASS 確認済み（2026-07-11）**
- **Stage3 初回発進 → 発進ゲート未達 → 停止・裁定・ゲート v2 amendment**
  （2026-07-11）: 初回発進 run はゲート v1（[401,500] 窓 ≥2.0×）未達（実測 1.322×）で
  事前登録プロトコルどおり停止、`aborted1_stage3_20260711_141705` として保全
  （preflight 18 検定 PASS・開始報告・機械的な bonus 適用確認は全て正常だった）。
  設計監督側の一次ソース解析（同 run ppo_diag.jsonl）で**実装は正常**と裁定:
  call_bonus 全1340バッチ b=5.0・計5248適用、call_taken 正規化 advantage が
  step 100 以降一貫して正（Stage1 の負から符号反転 = 勾配到達）、π(鳴き|可能∧赤保持)
  は 0.27→0.47（step 900–999）へ単調上昇（2.0× 到達外挿 step ~1300–1500）。
  原因は Stage2 の機械的介入ゲートの「step 500」を学習応答に移植した較正ミス
  （配管検査と方策応答検査の混同）。Gamba 裁定の下、`stage3_design.md` §3 を
  **二段ゲート v2**（機械ゲート @step200: b=5.0 + n_applied>0 バッチ割合 ≥50% /
  学習応答ゲート @step2000: [1801,2000] n加重 ≥2.0×baseline / 較正バンド
  [0.20,0.31] 不変）へ明示的 amendment として改訂（§4 判定プロトコル・b・
  schedule は不変）。**再発進は新 run dir で別タスク（未実施）**。
  `check_stage3_launch_gate.py` の v2 窓対応も再発進タスクに含める
- **Stage3 再発進・二段ゲート v2 両方通過・凍結中（2026-07-12 03:34:03 JST 発進）**:
  run dir `/home/gamba/mahjong/runs/ppo/stage3_20260712_033403`、tmux セッション
  `ppo_stage3_20260712_033403`。init = beta1_huber_192x40（v1 aborted run と同一、
  Stage1/2 checkpoint 混入なし）。config は前回同様 `stage3_PENDING_LAUNCH` プレース
  ホルダを launcher が run パスへ in-place 解決（diff は run パスのみ、b/schedule/
  p_enrich は不変）。発進前 preflight（残党チェック・libriichi rebuild・
  `verify_ppo_p1.py` 全18検定 PASS）をローカル WSL で実施済み。
  `check_stage3_launch_gate.py` を v2（`--gate mechanical` / `--gate learning_response`
  の二段）に書き換え、v1 aborted run の diag で事前サニティ確認済み。
  **機械ゲート（@step200）通過**: 全バッチ b=5.0、trainer_step∈[0,200] で
  n_applied>0 のバッチ割合 85.6%（≥50%閾値）。**学習応答ゲート（@step2000）通過**:
  baseline（step0–200）0.2714（較正バンド[0.20,0.31]内）、gate値（step1801–2000）
  0.6719、倍率 **2.475×**（≥2.0×閾値）。両ゲート判定時点で監視4項目
  （mismatch/fallback/chip_err/trainer NaN）全て0、alive clients 3/3（server/trainer
  含め全プロセス稼働）。**凍結宣言済み: step 16000 完走までコード・config 変更禁止**
  （例外はクラッシュとデータ整合性の破れのみ）。**判定窓は step 8000–16000**
  （`stage3_design.md` §4）。完走・判定集計は別タスク
- **DRCA プローブ設計ドラフト commit 済み**（2026-07-12、docs-only・コード変更なし・
  Stage3 凍結非抵触）: `freeparlor/docs/design/drca_probe_design.md`。
  duplicate rollout による鳴き反実仮想アドバンテージ Q(s,鳴く)−Q(s,鳴かない) の
  直接測定（診断計測器、訓練介入ではない）。実装方式は fork-by-replay
  （seed 決定論 + 台本再生 prefix、libriichi 変更ゼロが設計目標）。
  **ステータスは DRAFT**: 解釈条件は未凍結、実行タスク発進前に同書 §5 チェックリストで
  確定・commit して事前登録とする。着手条件は Stage3 凍結解除
  （step 16000 完走 + 判定確定）後。バックログ 8
- **Stage3 完走確認 + eval バッテリー(3レンズ)完了**（2026-07-13、コード変更なし・
  eval実行のみ）: run `stage3_20260712_033403` は step 16000 で正常終了（exit=0、
  mismatch/fallback/chip errors/trainer NaN 全て0）、checkpoints 9個・ppo_diag.jsonl
  （144,008行）・tb 保全確認済み。anneal schedule 違反 0。step_016000.pth は
  steps=16000・actor_critic 込みでロード可能（sha256 確認済み）。**発見: watchdog×
  Cleanup 競合による良性の二重起動 1 回**（trainer が正常終了(exit 0)後も server
  生存中は watchdog が無条件再起動、2回目は checkpoint ロード直後に max_steps 到達を
  検知し即終了・追加学習ステップなし。学習内容への影響なし、バックログ#4
  「launcher Cleanup ステップの修正」に統合する新規証拠）。標準argmax eval
  （6 checkpoint、init/2000/4000/8000/12000/16000）、grp_baseline 1v3
  （init/step16000、配備税チェック含む）、メタ対決probe（Stage3-16000 vs
  Stage1-16000 ×3）を実行。全 eval 経路で p_enrich=0・call_bonus_b=0 を assert 確認
  （`eval_ppo_smoke_sanity.py`/`eval_grp_baseline_1v3.py`/`eval_meta_stage1_vs_stage2.py`
  に call_bonus_b assert を追加、学習コード・config・訓練launcher・検定ロジック無変更）。
  レンズ1: b=5.0固定期（step0-4000）の副露率スパイク（62.86%→77.84%）は事前登録済み
  想定挙動、anneal完了後step16000で14.66%まで沈静化。レンズ2: 配備税チェック
  （2SE基準）は全ストリームで n.s.（Stage2の明確な有意差と対照的）。レンズ3:
  Stage3-16000 vs Stage1-16000 ×3 は全ストリームでミラーマッチ理論値から明確に逸脱
  （素点−7.42、チップ−0.675±0.236≈−2.9SE、Stage2の同レンズがパリティだったのと対照的）。
  init行はレンズ1/2ともStage1/2結果と完全一致（整合性チェック）。結果は
  `ppo_p3_stage3_result.md` に eval 節のみ commit（判定窓 step 8000–16000 の判定
  1/2/3 は未実施、設計監督側の別タスク）。新規eval スクリプト3本追加
  （`run_eval_battery_stage3.sh`/`run_eval_grp_baseline_1v3_stage3.sh`/
  `run_eval_meta_stage1_vs_stage3.sh`）
- **Stage3 判定完了（2026-07-13、設計監督側起草・学習コード変更なし）**: 事前固定条件
  （`stage3_design.md` §4）への照合により**分岐2 成立**（窓平均 0.2894 = 1.066×、
  slope/SE=**−21.03** の有意下降で減衰中。四半期 1.445×→0.690×、最終 500-step
  バケット 0.154 で Stage1 平衡方向へ収束中）→ **本質的機会費用仮説を支持、
  Stage3（anneal 付き報酬足場）は失敗。探索ラダー（Stage1 純探索 → Stage2 分布介入
  → Stage3 報酬介入）は全段不成立で閉幕**。事前登録の副次確認（鳴き局正典収支の
  改善方向）は成立（チップ/局 −0.95→−0.63→−0.33）だが損益分岐に届かず、機会費用
  ギャップ（~5.5–5.9 チップ）が支配項のまま（7b）。判定窓の raw advantage
  （call_taken +0.018 vs declined +0.522）は Stage1（+0.012 vs +0.524）をほぼ定量再現
  — 正典反鳴き勾配は経路によらない頑健な均衡性質と確定（7a）。anneal 内蔵の報酬介入は
  配備税ゼロ（レンズ2 全 n.s.、Stage2 の有意な税と対照）で「介入は anneal とセット」
  制約を対照実験的に検証（7c、方法論的収穫）。メタ対決は残留鳴きが立直メタで負債で
  あることを示唆（7d、−2.9SE）。判定・発見（§6–§10）は `ppo_p3_stage3_result.md` に
  追記済み、集計スクリプト `aggregate_stage3_judgment.py` /
  `aggregate_stage3_secondary.py` を commit（再現性確保）。**次アクションは
  DRCA プローブ解封（着手条件充足）と探索ラダー閉幕後の方針設計セッション**（下記）
- **DRCA プローブ事前登録凍結（2026-07-13、設計監督側・docs-only・コード変更なし）**:
  Gamba 裁定（前セッション末尾の「次は CFR」= DRCA プローブの意と確認済み）を受け、
  `drca_probe_design.md` に §5a を追加し解釈条件を凍結・commit（当該 commit が事前登録。
  以後の変更禁止、唯一の許容 amendment は §5a-1 の規模確定規則の機械的適用のみ）。
  確定内容: checkpoint 4点（init / Stage1-16000 / Stage3-8000 / Stage3-16000、
  Stage2 系は配備税による解釈汚染のため除外）、実効測定枠 6（セット(a)4 + セット(b)2、
  セット(b) 参照方策 = Stage1-16000）、有意基準 2SE（cluster-robust）+ 主 contrast
  事前列挙（多重比較補正なし・層別は全て exploratory）、暫定 K=8 / N=500 と
  パイロット 50 分岐点からの規模確定規則、抽出プロトコル（1局最大1分岐点、
  seed 基点 20260713 / 抽出 seed 713）。**実装（採取・並走・集計ハーネス3本、
  学習コード・config・libriichi 無変更が設計目標）→ パイロット → 規模確定
  amendment → 本測定は別タスクで未着手**
- **DRCA プローブ実装（2026-07-13 初版 → 同日差し戻し修正、実装エージェント=Sonnet・
  実装+smoke test のみ・パイロット/本測定は未着手）**: 初版は採取・並走・集計の
  3スクリプトを `freeparlor/scripts/` に追加（`drca_collect_branchpoints.py` /
  `drca_run_probe.py` / `drca_aggregate.py`）+ 共有ヘルパ `drca_common.py` +
  静的サポート設定 `freeparlor/configs/ppo_drca_probe.toml`（76cd237）。初版の
  台本再生機構（採取済み json.gz ログを `GameplayLoader` で事後的に全4席分
  再デコードし、(obs,mask) の全席探索で問い合わせ元の席を解決）は、監督側の
  独立再実行で **実 react_batch クエリ列と 1:1 対応しないことが判明**
  （at_kyoku>=1 の分岐点で "query matched 0 candidate seat(s)" — 訓練監視の
  loader size delta と同種の既知クラスの divergence）。**同日中に差し戻し修正**:
  1) 採取時（`drca_collect_branchpoints.py`）に両 engine へ記録専用パススルー
  ラッパー（`RecordingPassthroughEngine`、新規・`drca_common.py`）を噛ませ、
  実 react_batch クエリ列そのものを game_key 別 (mask_obs_digest, action) の
  順序付きリストとして sidecar jsonl (`<game_key>.script.jsonl`) に記録。
  GameplayLoader 由来の候補点は digest 一致（厳密1件を assert、0/複数件は
  loud FAIL）でこの記録列内 index に紐付ける。`drca_run_probe.py` の
  `ScriptedForkEngine`/`ForkState` は席解決を全廃し、この記録列を game_key 単位の
  単一 FIFO キューとして消費（毎クエリで digest 一致を assert = r2 検定）。
  GameplayLoader は分岐点のメタデータ（at_kyoku/shanten/at_turn）と最終報酬計算
  にのみ用途限定。2) `--challenger-seat-only`（採取側）+ mode=b 起動時の
  全分岐点 seat assert（`drca_run_probe.py`、one_vs_three.rs の固定 split
  写像に一致しない分岐点は起動直後に loud FAIL）を追加。3) engine 構築を
  ロールアウトループの外（`main()` で1回）に移し、ForkState/wrapper のみ
  ロールアウトごとに新規生成するよう再構成。4) `--torch-seed`（デフォルト
  20260713）を採取側に追加。5) 分岐時点の点況順位（分岐局開始時点の
  持ち点由来、`kyoku_start_scores`/`score_rank`、`drca_common.py`）を採取
  レコードに追加し、`drca_aggregate.py` の exploratory 層別に4項目目として
  追加（§5a-5 は解釈条件ではなく列挙の追加のみのため凍結節の変更に当たらない）。
  6) `drca_aggregate.py` に `--expect-branch-points` 安全弁（不足時 FATAL exit 1）
  を追加。libriichi・`mortal/train_ppo.py`・`mortal/client.py`・`mortal/ppo.py`・
  検定(1)–(18)ロジックは無変更。**ローカル WSL で end-to-end smoke test 完了
  (2026-07-13、init checkpoint=beta1_huber_192x40)**: (a) 採取 n=6 ×2セット
  （`--torch-seed` 違い、各セット at_kyoku>=1 が5/6）× mode a K=2 が両セットとも
  24/24 ロールアウト divergence ゼロで完走（`n_scripted_served` が採取時の
  `script_index` と一致することを確認）。(b) `--challenger-seat-only` 採取 +
  mode b（参照=Stage1 step_016000）K=2 が 12/12 完走、および seat 不一致の
  分岐点（採取時 `--challenger-seat-only` 未指定のセット）を mode b に食わせると
  起動直後に assert で loud FAIL することを実演。(c) `--inject-fault-for-test` は
  引き続き libriichi 自身の合法性検査への loud RuntimeError
  (`invalid action ... Caused by: 1m is not in hand`) で止まることを実演。
  (d) ローカル WSL で `verify_ppo_p1.py` 全18検定 PASS 確認済み（学習コード
  無変更につき検定内容は不変）。**副次的な観測事実（停止条件には該当せず、
  実装バグではなく既知クラスの構造的非決定論）**: `--torch-seed` 固定でも
  同一引数2回実行で採取される分岐点集合は一致しなかった（同一 seed の
  同一山でも、並列 obs エンコード（rayon）のスレッド完了順序が
  react_batch へのバッチ到着順を左右し、単一 global torch RNG ストリームの
  消費順がずれる — `verify_ppo_p1.py` 検定(15) の daiminkan flake と同種、
  libriichi/mortal 側の並行処理設計に起因し本タスクのスコープ外）。台本
  記録・消費という fork-by-replay の中核メカニズム自体はこの非決定論に
  依存しない（採取時に記録された実クエリ列をそのまま強制再生するため）ので
  r2 の健全性には影響しない。パイロット (50分岐点)・規模確定 amendment・
  本測定はいずれも別タスクで未着手
- **DRCA プローブ実装 2度目の差し戻し修正（2026-07-13、実装エージェント=Sonnet・
  実装+smoke test のみ・パイロット/本測定は未着手）**: 上記1度目の差し戻し
  （f59137b）は「台本記録・消費の中核メカニズムは到着順非決定論に依存しない」と
  評価していたが誤りだった。監督側の独立再実行で、同一 checkpoint・同一引数の
  probe 全数を複数回走らせると digest divergence が実行のたびに異なる queue
  index（観測: 27 / 101 / 174 / 183）で発生することが判明し、単一 FIFO 前提の
  破れを特定: `libriichi/src/agent/mortal.rs` の `get_reaction` は
  `rayon::spawn` 内で `step_metas.push` を行うため、react_batch へのバッチ
  到着順はスレッド完了順に左右され実行ごとに変動する（採取時に記録した順序と
  再生時に問い合わせが来る順序が一致する保証がない）。**原因特定を受け、
  到着順に依存しない鍵へ置換**: step_meta の第2要素 `seq`（mortal.rs の
  `step_seqs` — スロット単位の決定論的連番、`start_game` で0リセット、
  クエリごとに+1。到着順ではなくゲーム内の論理的発生順そのものなので
  非決定論に影響されない）を使い、`RecordingPassthroughEngine`
  （`drca_common.py`）の記録先を `store[game_key][role][seq] = [(digest,
  action), ...]` に変更（role は 'challenger'/'champion'、コンストラクタ引数
  として新規追加）。challenger は1 game 1 slot なので (game_key, role, seq)
  は unique、champion は1 game 3 slot を持ち各 slot の seq が独立に0起算のため
  同一 (game_key, 'champion', seq) に最大3エントリ（異なる digest）が集まり
  得る — その場合は digest が実際に問い合わせられた1件を特定する（0件・
  複数件一致は loud FAIL、r2 assert の新形態）。sidecar 形式を
  `{"role","seq","digest","action"}`/行に変更（`write_script_sidecar`/
  `load_script_sidecar`）。`drca_collect_branchpoints.py` は分岐点 digest を
  当該 game_key の全 role・全 seq エントリから検索し一致1件を assert する
  新規ヘルパ `find_unique_role_seq_match`（`drca_common.py`）を使い、
  `script_index` の代わりに `branch_role`/`branch_seq` を記録。
  `drca_run_probe.py` の `ForkState.next_action_for` は
  `consumed`（FIFO カーソル）を全廃し `(role, seq)` バケット + digest 照合に
  置換、`run_one_rollout` は sidecar 読み込み直後に
  `find_unique_role_seq_match` で再検証し記録済み `branch_role`/`branch_seq`
  と一致することを assert（sidecar 破損検出）。libriichi・
  `mortal/train_ppo.py`・`mortal/client.py`・`mortal/ppo.py`・検定(1)–(18)
  ロジックは無変更（Rust 表面積ゼロを維持）。**ローカル WSL で end-to-end
  smoke test 完了（2026-07-13、init checkpoint=beta1_huber_192x40 /
  S1FINAL=Stage1 step_016000）**: (a) 採取 n=6（at_kyoku>=1=5）→ mode a K=2
  全数 probe（24 rollout/回）を**連続3回**実行し、3回とも divergence ゼロ・
  24/24完走・exit=0（監督側はこの反復実行方法で f59137b の破れを検出していた
  ため、単発成功ではなく反復で確認）。(b) `--challenger-seat-only` 採取 n=3
  → mode b（参照=Stage1 step_016000）K=2 が 12/12 完走・
  `challenger-seat assert passed for all 3 branch points`、および seat 不一致
  分岐点（採取時 `--challenger-seat-only` 未指定のセット）を mode b に食わせる
  と起動直後に `mode=b requires branch point seat...` assert で loud FAIL
  することを実演（exit=1）。(c) `--inject-fault-for-test` は引き続き
  libriichi 自身の合法性検査への loud RuntimeError（`invalid action:
  ... Dahai { actor: 0, pai: 1m ... }`）で止まることを実演（exit=1）。
  (d) ローカル WSL で `verify_ppo_p1.py` 全18検定 PASS 確認済み（学習コード
  無変更につき検定内容は不変）。パイロット (50分岐点)・規模確定 amendment・
  本測定はいずれも別タスクで未着手
- **DRCA 実装 9fea7a7 の監督側3段検証 合格 + §1 no-call 腕 amendment
  （2026-07-13、設計監督側）**: (role, seq) バケット照合の独立再実行検証で
  mode a probe 連続3回 divergence ゼロ（深い分岐点・champion 役・seq 92 まで）、
  mode b 6/6 + 負テスト発火、fault 注入 loud FAIL、`--expect-branch-points`
  安全弁発火、検定18本 PASS — **再生機構（fork-by-replay）は検証合格**。
  ただし監督側スモークで**凍結仕様の穴を1件発見**: 自分の手番の暗槓/加槓機会
  （分岐点定義 sel が正当に含むクラス、実測 `call_types_available=[42]`・
  元の行動=打牌）では パス(45) が mask 上存在せず、§1「no-call 腕 = パスのみ強制」
  が定義不能（no-call 腕の assert で停止。実装バグではなく仕様の未考慮ケース。
  Sonnet のスモークが通ったのは槓×赤保持機会が稀なため）。Gamba 裁定の下、
  `drca_probe_design.md` §1 を**非鳴き制限サンプリング**（no-call 腕 = π を
  非鳴き行動群 mask ∧ ¬[38..42] に制限して再正規化 — call 腕と対称、通常の
  リアクションでは従来のパス強制に帰着、ロン可能時はロン許容）へ測定開始前
  amendment として改訂・commit。分岐点定義（§5a-4）・解釈条件（§5a）は不変。
  **no-call 腕の実装変更（`drca_run_probe.py` `_sample_branch_action` のみ）+
  own-turn kan 分岐点を含むスモークは別タスクで未実施**
- **DRCA no-call 腕の非鳴き制限サンプリング化 実装完了**（2026-07-14、実装エージェント
  =Sonnet・変更対象は `freeparlor/scripts/drca_run_probe.py` の `_sample_branch_action`
  のみ・学習コード/config/libriichi/検定(1)–(18)ロジック無変更）: §1 amendment
  （6a439c2）対応。no-call 腕を「`DECLINE_ACTION`(45) 強制」から call 腕と対称の
  実装へ置換 — legal mask ∧ ¬[`CALL_ACTION_MIN`..`CALL_ACTION_MAX`]（38–42）に
  制限した上で π を再正規化サンプリング（非鳴き合法行動が0件なら assert で loud
  FAIL）。call 腕のロジック・`DECLINE_ACTION`（`drca_common.py` 定義、値不変）は
  無変更。**ローカル WSL で検証完了**: (a) 監督側の検証済み採取物
  （`/tmp/drca_supervisor_verify3/bp1.jsonl`、own-turn 暗槓/加槓分岐点
  `call_types_available=[42]`・元の行動=打牌 を含む 6 分岐点）で mode a K=2
  全数 probe（24 rollout/回）を連続3回実行し、3回とも 24/24 完走・exit=0・
  divergence ゼロ。当該 kan 分岐点（branch 5/6）の no_call 腕は3回とも
  forced_action∈{1,28}（いずれも非鳴き discard action、call 範囲外）で、
  amendment 前なら assert 停止していたケースが正常に非鳴き行動へ解決されることを
  確認。(b) 残り5分岐点（通常のリアクション、パスが唯一の非鳴き選択肢）の no_call
  腕は3回とも一貫して forced_action=45 — 従来挙動への帰着を確認。(c)
  `--inject-fault-for-test` は引き続き libriichi 自身の合法性検査への loud
  RuntimeError（`1m is not in hand`）で停止することを実演（exit=1）。(d)
  ローカル WSL で `verify_ppo_p1.py` 全18検定 PASS 確認済み（学習コード無変更に
  つき検定内容は不変）。**残りは規模確定 amendment（設計監督側）→ 本測定 → 判定
  （別タスク、バックログ0）**
- **DRCA パイロット測定完了**（2026-07-14、測定のみ・コード変更なし）: 成果物
  `/home/gamba/mahjong/runs/drca/pilot_20260714_020133`（tmux `drca_pilot_20260714_020133`、
  exit=0）。セット(a)×Stage1-16000×主レンズ、§5a-4 凍結 seed 類は全デフォルト。
  採取 137s（50/50、at_kyoku>=1=44/50、own-turn kan=0件 — 当初 branch_index=21 を
  own-turn kan と誤記したが実体は call_types_available=[41,42] のポン/大明槓
  リアクション（no_call 腕は全て 45）。監督検証 2026-07-15 で訂正）、
  並走 52906s（800 rollout、54.44 rollouts/h）、集計 <1s。構成 dump で
  p_enrich=0.0・call_bonus_b=0.0 を確認。divergence/assert/エラー 0 件。
  aggregate（50 分岐点・両腕揃い）: ΔQ̂=−0.4575 千点、cluster SE=1.5882、
  |ΔQ̂|/SE=0.288、符号検定 25+/24−/1ゼロ（p=1.0）。exploratory 層別 —
  shanten: 1向け−2.04/2+向け+0.10/聴牌−3.13、turn: early+0.60/mid−2.95/late+2.43、
  call_type: chi−1.52/pon+3.67、score_rank: rank3−4.13 が最大絶対値。
  本測定素朴外挿（6枠×N=500×2K=16 → 48000 rollout @54.44/h）: 並走 ~882h
  （~36.7日）+ 採取 ~2.3h ≈ 合計 ~884h。規模確定 amendment は設計監督側（§5a-1）。
  パイロット 50 分岐点は本測定に合算予定のため成果物保全済み
- **パイロット監督検証合格 + 規模確定（§5a-1a）+ 並列化方針裁定**（2026-07-15、
  設計監督側・docs-only）: パイロットの3段検証合格（aggregate 独立再実行が commit 値と
  完全一致、divergence 0、構成 assert 確認。own-turn kan 1件は誤記で実体 0 件 — 上記
  エントリに訂正済み）。§5a-1 規則の機械適用で **K=8（K=4 間引きの SE 劣化 +11.7% ≥
  10%）/ N=485** を確定し `drca_probe_design.md` §5a-1a として commit。素朴 projection
  855h の律速が GPU/CPU いずれでもない（実測 GPU util 9% / CPU 11%、10コア）ことを
  確認し、Gamba 裁定で **(1) ロールアウト並列化（ドライバ変更のみ・解釈条件に無関係）
  → 再実測 projection に対し 48h 条項を適用、(2) 実施順 = セット(a)×Stage1-16000 先行、
  (3) 物理的中断（2026-07-17 PC 移設）と残り分岐点からの再開を許容** を §5a-1a 運用
  amendment として登録。次: 並列化実装（Composer）→ 監督3段検証 → 本測定発進
  （17日の PC 移設で強制停止 → 移設後に --resume で再開）
- **DRCA ロールアウト並列化 実装完了**（2026-07-15、実装エージェント=Composer・
  変更対象 `drca_run_probe.py` / `drca_aggregate.py` のみ・学習コード/config/libriichi/
  検定(1)–(18)ロジック無変更）: `--parallel N`（デフォルト1）で N ワーカースレッドが
  分岐点×腕×k を分担。live engine 2体は main で1回構築・全スレッド共有、
  ForkState/wrapper/arena/tempdir はロールアウトごとに新規。共有 engine への全 forward
  （`react_batch` / `_forward_logits`）は単一 `threading.Lock` で直列化、jsonl 書き込みも
  lock+flush。両 engine 構築直後に `record_trajectory=False`（probe は pending_by_game を
  drain しないため本測定規模で RAM 無制限成長を防止）、終了時
  `illegal_action_fallback_count==0` assert。`--resume` は既存 `--out` から 2×K 揃い
  分岐点のみ保持・書き直し・残りのみ処理（§5a-1a 運用3）。`drca_aggregate.py` の
  グループキーを `branch_index` → `(game_key, branch_role, branch_seq)` に変更
  （`branch_index` は参考列として残す）。**ローカル WSL 検証完了**: (a) bp1.jsonl
  mode a K=2 --parallel 8 を2回実行、いずれも 24/24 完走・divergence/assert 0；
  (b) --limit 4 K=2 実測: parallel=1 993.1s/58.00 rollouts/h、parallel=8 1025.0s/
  56.20 rollouts/h（speedup 0.96×、律速は Rust シミュレーション側。nvidia-smi
  memory peak 1742 MiB）→ N=485 枠（7760 rollout）projection ≈131h（@~59 rollouts/h）；
  (c) --resume 実演（6 件部分記録 kill → 2 不完全分岐点破棄・0 保持 → 16/16 完走）；
  (d) --inject-fault-for-test loud FAIL（parallel=8、`1m is not in hand`）；
  (e) `verify_ppo_p1.py` 全18検定 PASS
- **DRCA 本測定・第1枠 発進中**（2026-07-15 08:20:21 JST 並走開始、測定のみ・
  コード変更なし）: run dir
  `/home/gamba/mahjong/runs/drca/main_s1final_20260715_075837`、tmux セッション
  `drca_main_s1final_20260715_075837`（3ペイン）。枠 = セット(a)×Stage1-16000
  （§5a-1a: N=485/K=8 = パイロット50 + 追加435）。発進前 preflight（残党なし・
  port5000空・libriichi rebuild・`verify_ppo_p1.py` 全18検定 PASS）実施済み。
  追加採取: `--seed-base=20260715`（pilot 最終 seed 20260714+1）`--n 435`、
  wall 1061s、at_kyoku>=1=388/435。shard 分割 145/145/145 → 3×`drca_run_probe.py`
  （各 `--mode a --k 8 --parallel 1`、checkpoint=Stage1 step_016000）。
  構成 dump（3プロセス共通）: p_enrich=0.0・call_bonus_b=0.0・
  record_trajectory=False（live engine 上）。30分時点合算: 47 rollout /
  94.0 rollouts/h（監督実測 112/h より初速やや低い・ウォームアップ中）、
  GPU util 59%・mem 2472MiB。残6913 rollout 予測 ~73.5h（~3.1日）。
  divergence/assert/エラー 0。**17日 PC 移設停止手順**:
  `stop_for_relocation.sh`（3プロセス kill → pilot/main を
  `/home/gamba/mahjong/backups/drca/` へ tar 退避 → shard 別完了分岐点報告）→
  移設後 `run_resume.sh`（同一 tmux 構成・`--resume`）で再開。
  判定・集計は別タスク
- **mjai 牌譜 HTML ビューア生成スクリプト 実装完了**（2026-07-15、新規スクリプト1本のみ・
  学習コード/libriichi/DRCA スクリプト/検定無変更）: `freeparlor/scripts/mjai_log_to_html.py`。
  入力 json.gz/jsonl（複数・ディレクトリ可）を `log-viewer/index.example.html` テンプレの
  `allActions` 埋め込みで HTML 化。`log-viewer/files/` を `--out-dir`（デフォルト
  `/home/gamba/mahjong/runs/viewer_out/`）へ1回コピー、一覧 `index.html` も生成。
  イベント行に `` ` `` / `${` が含まれる場合は loud FAIL。同名ログは `__2` 等で自動回避。
  実演4ログ（DRCA pilot×2 / Stage1 argmax eval `test_play`×1 /
  Stage3 vs Stage1 meta `eval_meta`×1）で元ログ行数と HTML 埋め込み行数が全件一致確認済み。
  生成 HTML はローカル成果物（commit 対象外）
- **mjai 牌譜 HTML ビューア 差し戻し修正**（2026-07-15、変更対象
  `mjai_log_to_html.py` のみ・学習コード/libriichi/DRCA/検定無変更）:
  `render_html` の `ALLACTIONS_RE.sub` が置換文字列のエスケープ解釈によりテンプレ末尾
  `.split('\n')` のバックスラッシュ+n を実改行へ変換し、生成 HTML のインライン JS が
  SyntaxError（allActions 未定義→盤面未描画）となるバグを修正。`re.sub` を廃止し
  match 位置での文字列スライス結合へ置換。`main` に生成 HTML へ
  `.split('\n')`（バックスラッシュ+n 2文字）が残存することを assert（loud FAIL）。
  既存4ログ再生成で行数一致（1478/1296/1048/1152）+ 新設 assert 全件 PASS 確認済み
- **DRCA 48h 条項の機械適用（§5a-1b）+ 残枠 launcher 準備完了**（2026-07-16、
  docs + ops スクリプト1本のみ・学習コード/DRCA ハーネス4本/検定(1)–(18)/進行中の
  第1枠 run 無変更）: §5a-1a 運用1 が指定する「並列化後の再実測 projection」への
  48h 条項適用を実施し `drca_probe_design.md` §5a-1b として commit。実測（第1枠
  有効稼働 ~28.6h で 3127 rollout ≈ 109 rollouts/h → ~71h/枠、全6枠 ≈ 427h ≫ 48h）
  → 削減(1) セット(b)×init 脚を脱落（実効測定枠 6→5、主 contrast 2 は
  「Stage3-16000 vs Stage1-16000」の1本に）、削減(2) 副レンズ絞りは字義どおり適用
  するが §5a-1a のコスト0化により壁時計効果 0（contrast 3 の評価対象を
  Stage1-16000 / Stage3-16000 の2枠に限定、記録は全枠継続・絞り外2枠のレンズ間比較は
  exploratory 降格）、削減リストはこれで尽きるため残 projection（~320h）で継続。
  新規 launcher `freeparlor/scripts/run_drca_main_frame.sh <frame>`（frame = a_init /
  a_s3final / a_s3mid / b_s3final、実施順もこの順 — §5a-1b）: 第1枠の実証済み
  テンプレ（run_main_s1final.sh / stop_for_relocation.sh / run_resume.sh、run dir 内）
  を1本に一般化。preflight（checkpoint/config 実在 → ディスク `DISK_MIN_GB` env・
  デフォルト100GB → 残党チェック `pgrep -f 'drca_run_probe|drca_collect_branchpoints|
  train_ppo'`）→ run dir `runs/drca/main_<frame>_<日時>` 新規作成（再利用禁止準拠）→
  採取（N=485・seed 基点 20260713 = §5a-4 凍結値へ回帰（枠間で同一配牌母集団から採取）・
  torch-seed 20260713・extract-seed 713、b_s3final のみ `--challenger-seat-only` +
  採取 checkpoint=Stage1-16000）→ shard 162/162/161 → tmux 3ペイン probe（K=8・
  `--parallel 1`×3、mode b は `--reference-checkpoint`=Stage1-16000）→ 停止・再開
  スクリプトを run dir へ自動生成（第1枠実証パターンのパラメタ化）。`--dry-run` は
  構成解決 + preflight のみで終了（run dir 非作成）。**検証（GPU 不要範囲、2026-07-16
  ローカル WSL）**: bash -n PASS、全4枠 dry-run で構成解決の正しさを確認（残党チェックが
  進行中の第1枠 probe 3プロセスを検出し exit 1 = GPU 1系統ガードの実発火を実証）、
  `DISK_MIN_GB=999999` で FATAL exit 1、不正 frame 名で FATAL exit 1、stop/resume
  生成ブロックの隔離実行（ダミー変数・mode b 構成）で生成物2本の bash -n PASS +
  resume 内 `\$?` エスケープ3箇所（二重展開防止）を確認。フル dry-run PASS（exit 0）と
  実発進は第1枠完走後の別タスク（発進時に従来どおり preflight +
  `verify_ppo_p1.py` 全検定を実施）
- **run 成果物の保持・清掃運用を明文化（バックログ7 消化）**（2026-07-16、docs-only・
  削除は未実施）: `freeparlor/docs/ops/run_artifact_retention.md` を制定。成果物クラス別
  保持ポリシー（checkpoints/logs/config/eval 牌譜/DRCA 採取物 = 恒久保全、drain =
  判定 commit 後に削除可、smoke = 検証結果 commit 後に削除可）、イベント駆動の清掃
  トリガ（判定後 + preflight FATAL 時。定期清掃はしない）、許可リスト方式の削除手順
  （毎回 Gamba 明示承認・du/df 前後記録・保全物実在確認）を規定。2026-07-16 棚卸しで
  **Stage3 本走の drain 368GB が判定完了（07-13）後も残置**されていたことを発見
  （明文化欠如の実害を実証）。清掃候補 3件 計 ~412GB（stage3 drain 368GB /
  aborted1_stage3 drain 31GB / stage1_20260705_014852 drain 13GB）を同書 §5 に列挙 —
  **削除の実施は Gamba 承認待ち**（GPU 非依存のため DRCA 測定中でも実施可）。
  ほかに `buffer/`（`.traj` ×1660、stage2_resume に 36GB）という未分類の蓄積クラスを
  発見、用途調査まで削除保留
- **0b 方針設計セッションの事前フレーム起草**（2026-07-16、docs-only・DRAFT・
  裁定非関与）: `freeparlor/docs/ops/policy_session_0b_frame.md`。3+1 議題
  （商用採否 / 経済定数変更 / 搾取者訓練 / Stage2b 再評価）× DRCA 判定帰結
  （シナリオ A/B/C/D/N、§4 凍結解釈に対応）の討議用分岐、確定済み事実ベースの
  棚卸し、議題別の不足材料リスト（商品性要件の Gamba ヒアリングが最大の未充足）を
  整理。DRCA の凍結解釈条件・判定には一切関与しない（セッション準備資料）
- **DRCA パイロット定性ドリルダウン + 牌譜 HTML ビューア生成**（2026-07-16、
  exploratory・判定非関与・進行中 run 無変更・CPU のみ）:
  `freeparlor/docs/reports/drca_pilot_qualitative_notes.md` + 再現スクリプト
  `freeparlor/scripts/drca_pilot_qualitative_drilldown.py`（read-only・全体平均の
  commit 値 −0.4575 一致を assert）。パイロット 50 分岐点の分岐点別 ΔQ̂ を計算し
  |ΔQ̂| 極値 8 件を牌譜から局面近似再構成。観察: 負の極値は「no_call 側の未来に
  自分の和了がある」型（オーラス2着目の逆転トップ手を序盤チーが壊す branch 44 等）、
  正の極値は (a) 役牌+赤2の明白な好機（branch 24 — 方策は実際に鳴いて親ツモ+12700、
  教科書的好機は既に取れている）と (b) 親ラッシュへの防御的速度鳴き（branch 15 —
  d_chip +16.25 で全項目中最大、方策はパス。**取りこぼし候補の作業仮説**）の2類型。
  ΔQ̂ 恒等 0 の終盤デッドポジション（branch 12）も確認。牌譜 HTML は pilot 5 半荘 +
  Stage3-16000 argmax eval 2 半荘（`test_play/` の中身が step_016000 のものであることを
  clear_log_dir 仕様 + ORDER + mtime で確定）を `runs/viewer_out/` に生成
  （ローカル成果物）。Stage3-16000 の人手定性レビュー（Stage1 版
  `qualitative_expert_review_20260715.md` の対）は未実施 — 0b 議題1 の材料候補として
  Gamba/専門家に委ねる
- **drain 清掃実施（run_artifact_retention.md §2 手順の初回適用）**（2026-07-16、
  ops 作業・コード変更なし・Gamba 承認済み）: 許可リスト3パスのみ削除 —
  `stage3_20260712_033403/drain`（368GB、判定完了済み 07-13）・
  `aborted1_stage3_20260711_141705/drain`（31GB、裁定完了済み 07-11）・
  `stage1_20260705_014852/drain`（13GB、判定非関与）。`df`: 495GB 使用/462GB 空き →
  84GB 使用/**872GB 空き**（411GB 回収）。削除後の保全物実在確認済み: stage3 の
  checkpoints ×9・`logs/ppo_diag.jsonl`（38MB）・config.toml、aborted1 の
  `ppo_diag.jsonl`（3.2MB）、stage1_014852 の checkpoints/logs/config。
  `buffer/` は許可リスト外につき未削除（用途調査まで保留）。進行中の DRCA 第1枠は
  削除前後で 3 プロセス稼働継続を確認（削除は GPU/プロセス非干渉の I/O のみ）
- **Stage3-16000 argmax の専門家定性レビューを記録（初回1半荘、判定非関与）**
  （2026-07-16、docs-only）: `qualitative_expert_review_stage3_20260716.md`
  （Stage1 版 07-15 の対）。Gamba 所見: (1) 鳴き頻度増加 + 新レパートリー
  （喰いタン・役バック仕掛け）だが役バック後に当の役牌を切って役消し、(2) 鳴いた後は
  下手なまま（ポンテンスルー複数回・鳴き手での不要なシャンテン戻し）、(3) カン判断が
  顕著に異常（役牌暗刻テンパイで4枚目にポンしてテンパイ崩し・良型立直中の暗槓拒否 —
  カン局面の希少性による未学習仮説）、(4) 降りの規律が崩壊気味（現物→突然の危険牌
  放銃が頻出、放銃率上昇と整合）。監督注記: 「入口だけ増えて鳴き後サブゲームが未熟」
  という state-coverage 仮説（未検証）、DRCA セット(b) の解釈への注意材料
  （Stage3 の鳴きスキルは実行品質では高くない可能性）、0b 議題1 では「鳴きが増えたが
  下手」は商品性最悪象限という暫定所見

## 残タスク（バックログ、2026-07-16 時点）

Stage3 判定完了（`ppo_p3_stage3_result.md` §9、探索ラダー閉幕）に伴う
次アクション・バックログ:

0. **DRCA プローブ（診断、事前登録済み）**: ~~解釈条件の凍結~~ ~~実装（採取・並走・
   集計ハーネス3本）~~ ~~no-call 腕の非鳴き制限サンプリング化（§1 amendment 対応の
   小修正、`drca_run_probe.py` のみ）~~ ~~パイロット 50 分岐点~~ **パイロット消化済み
   （2026-07-14、「現在の状態」節参照。成果物
   `pilot_20260714_020133`、異常 0 件）**。
   ~~規模確定 amendment~~ **消化済み（2026-07-15、§5a-1a: K=8/N=485 + 並列化・
   実施順・中断再開の運用 amendment — 「現在の状態」節参照）**。
   ~~ロールアウト並列化~~ **消化済み（2026-07-15、「現在の状態」節参照）**。
   ~~監督3段検証~~ **消化済み（2026-07-15、並列化実装検証 — 「現在の状態」節参照）**。
   **本測定第1枠 発進中**（2026-07-15、`main_s1final_20260715_075837`、
   3プロセス並走・17日 PC 移設で中断 → `--resume` 再開 — 「現在の状態」節参照）。
   **48h 条項適用済み（2026-07-16、§5a-1b: セット(b)×init 脱落で実効5枠）+
   残4枠 launcher 準備済み（`run_drca_main_frame.sh` — 「現在の状態」節参照）**。
   残り: 第1枠完走 → 残4枠（a_init → a_s3final → a_s3mid → b_s3final）→ 判定。
   7b「ギャップの支配項は立直ペイロード」読みの反実仮想による直接検証
0b. **探索ラダー閉幕後の方針設計セッション**（設計監督側）: 立直マキシマリズムの
   商用採否 / 経済定数変更（新実験系）/ 敵対的搾取者訓練の要否を DRCA 結果と併せて裁定。
   **事前フレーム起草済み（2026-07-16、`freeparlor/docs/ops/policy_session_0b_frame.md`
   — 議題×DRCA 帰結の分岐シナリオ A/B/C/D/N と不足材料リスト。DRAFT・裁定非関与）**。
   残り: 商品性要件の Gamba ヒアリング → DRCA 判定後にセッション実施

1. ~~**Stage3 実装**~~ **消化済み（2026-07-11、実装・検定のみ・未発進 —
   「現在の状態」節参照）**。仕様は `stage3_design.md` §2/§7 が正
1b. ~~**Stage3 発進前提の drain 清掃**（`stage3_design.md` §8）~~ **消化済み
   （2026-07-11、「現在の状態」節参照）**
2. anneal 実験（`stage2_design.md` §6）は対象外のまま（判定1成立時のみの規定、今回不成立）
3. **Stage2b（解凍実験）の再評価**: 配備税の発見により「収束済み方策の分布シフト適応」の
   商用価値が上がった。実施判断は Stage3 議論と併せて検討
4. launcher Cleanup ステップの修正（server/client 道連れ終了、2026-07-10 インシデント。
   2026-07-13 に新規証拠追加: trainer_watchdog が正常終了(exit 0)後も無条件再起動する
   watchdog×Cleanup 競合による良性の二重起動、Stage3 run で観測 — 「現在の状態」節参照）
5. メタ系ハーネス（レンズ3 等）へのミラー較正脚追加
6. verify（`verify_ppo_p1.py`）冒頭への .so 鮮度チェック追加
7. ~~drain 清掃の恒久運用~~ **明文化済み（2026-07-16、
   `freeparlor/docs/ops/run_artifact_retention.md`）+ 清掃候補3件の削除実施済み
   （同日 Gamba 承認、計411GB回収 — 「現在の状態」節参照）**。残り:
   `buffer/`（.traj、stage2_resume に 36GB）の用途調査のみ
8. ~~**DRCA プローブ（診断）**~~ **解封済み → 項目 0 へ統合（2026-07-13、
   Stage3 判定確定により着手条件充足）**。設計ドラフトは `drca_probe_design.md`
   （2026-07-12 commit）、実装は採取・並走・集計ハーネス3本・学習コード無変更が目標

## 役割分担

設計判断・レビュー・仮説の裁定は Claude（chat 側・監督）が担当。このリポジトリでの
あなた（実装エージェント = Cursor Composer / Claude Code 等。現在どれかは
「現在の状態」節を参照）の役割は実装・検証・commit・push。
設計変更が必要だと感じたら、実装せずに提案として報告する。
