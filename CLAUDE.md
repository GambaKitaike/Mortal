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

## 現在の状態（2026-07-09 時点）

> **この節は陳腐化する前提。** 正は `stage2_design.md`・`ppo_p3_stage1_result.md`・
> git log。状態を変えるタスクを完了したら、この節の更新も同一 commit に含めること。

- **Stage1 判定完了**（2026-07-06）: 事前固定条件（赤保持鳴き試行率 2 倍未満 かつ
  上昇トレンド無し）が両方成立（0.236×、slope/SE=−6.14）→ Stage2 移行確定
- **Stage2 設計 commit 済み**（`stage2_design.md`、判定条件は事前登録済み・変更禁止）
- **実装エージェント**: Claude Code CLI（ローカル WSL 内。GPU・conda 環境・tmux に直接アクセス可）
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

## 残タスク（バックログ、2026-07-11 時点）

Stage2 判定完了（`ppo_p3_stage2_result.md` §9）・Stage3 設計確定
（`stage3_design.md`）に伴う次アクション・バックログ:

1. ~~**Stage3 実装**~~ **消化済み（2026-07-11、実装・検定のみ・未発進 —
   「現在の状態」節参照）**。仕様は `stage3_design.md` §2/§7 が正
1b. ~~**Stage3 発進前提の drain 清掃**（`stage3_design.md` §8）~~ **消化済み
   （2026-07-11、「現在の状態」節参照）**
2. anneal 実験（`stage2_design.md` §6）は対象外のまま（判定1成立時のみの規定、今回不成立）
3. **Stage2b（解凍実験）の再評価**: 配備税の発見により「収束済み方策の分布シフト適応」の
   商用価値が上がった。実施判断は Stage3 議論と併せて検討
4. launcher Cleanup ステップの修正（server/client 道連れ終了、2026-07-10 インシデント）
5. メタ系ハーネス（レンズ3 等）へのミラー較正脚追加
6. verify（`verify_ppo_p1.py`）冒頭への .so 鮮度チェック追加
7. drain 清掃の恒久運用（判定集計後の削除運用の明文化、または古い aborted/smoke run の
   定期清掃 — ディスク枯渇インシデント再発防止、2026-07-09）

## 役割分担

設計判断・レビュー・仮説の裁定は Claude（chat 側・監督）が担当。このリポジトリでの
あなた（実装エージェント = Cursor Composer / Claude Code 等。現在どれかは
「現在の状態」節を参照）の役割は実装・検証・commit・push。
設計変更が必要だと感じたら、実装せずに提案として報告する。
