# CLAUDE.md — Mortal フリー雀荘 PPO プロジェクト

## プロジェクト概要

Mortal をフリー雀荘ルール（素点+ウマオカ+チップ、β=1）向けに再設計する調査基盤。
教師データ非依存・自己対戦 PPO。探索ラダー（Stage1〜3）は全段不成立で閉幕し、
DRCA プローブで反鳴き均衡が経済の性質であることを確認。現在は **anchor 系列
（アンカー付き PPO、基礎技能劣化への対策）が優先軸**で、Arm C 本走が凍結中。

**最初に読む文書（この順で）:**
1. `freeparlor/docs/design/ppo_migration_design.md` — PPO 移行の設計正典
2. `freeparlor/docs/design/anchored_ppo_design.md` — **現行 run（anchor 系列）の設計・
   事前登録済み判定条件**（凍結済み）
3. `freeparlor/docs/reports/fundamentals_degradation_diagnosis_20260725.md` +
   `fundamentals_significance_pass_20260725.md` — anchor 系列の動機（基礎技能劣化の診断）
4. `freeparlor/docs/reports/ppo_p3_stage1_result.md` — Stage1 判定結果（立直マキシマリズム）
5. `freeparlor/docs/design/reward_design_teacherfree.md` — 報酬設計の確定事項
6. `freeparlor/docs/ops/project_history.md` — 2026-07-06 以降の時系列経緯（旧「現在の状態」節）
7. `freeparlor/docs/ops/next_steps_2.md` — プロジェクト全体史（初期）
8. `freeparlor/docs/INDEX.md` — docs 全体の索引

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
  本書に本数をハードコードしない（2026-07-25 時点 20 本。19/20 本目が anchor 系列の
  Arm C / Arm K assert — `anchored_ppo_design.md` §7）
- 発進前 preflight は**実装完了報告を受け取った後にも**残党チェックを行う
  （報告に現れない孤児プロセスが GPU を掴んでいた事例あり — `project_history.md` 2026-07-25）
- 発進後: 開始報告（config 全文 + 監視項目 + step 100 到達 + alive clients 3/3）
- **凍結ルール**: 事前登録した run は判定窓が閉じるまでコード・config 変更禁止。
  例外はクラッシュとデータ整合性の破れのみ。「気になる挙動」は記録して続行

### 監視期待値（1件でも非ゼロなら報告）

**run 中の監視は `run_ppo_p3_stage1_inner.sh` の monitor ループが正**（本節はその写し。
片側だけ直すと乖離するので、実装を変えたら同一 commit で本節も直す）。

FATAL（検出即停止・run dir は証拠として保全）:

| シグナル | 意味 | exit |
|---|---|---|
| `trajectory game key missing` | **その局を丸ごと捨てている**（`client.py:108`） | 9 |
| `trajectory orphan steps` | 記録済み step に対応する牌譜が無い（`client.py:184`） | 10 |
| `illegal_action_fallback_count` 非ゼロ | 不正行動のフォールバック | 5 |
| `online chip resolution failed` | チップ解決の失敗 | 2 |
| trainer NaN / 非有限 | — | 3 |

非致命: `loader size delta` = INFO（報告のみ。完走 run で 8,759〜12,137 件出るのが平常）。
その他: alive clients = 3/3、step 到達性（停滞は異常）、monitor 期限切れ = exit 8
（`MONITOR_HOURS` 既定 48h。**期限切れは正常完走と別経路** — 2026-07-26 インシデント）。

⚠ **`trajectory step count mismatch` はデッドの遺物**（2026-07-28 確認）。この文字列を
**出力するコードは repo に存在しない**（`verify_ppo_p1.py` の counter と docs にのみ残存。
P2 期の改修で emitter が消えた）ため構造的に常に 0 で、**「0 だから健全」の根拠にならない**。
監視 grep は emitter 復活時に拾えるよう残してあるが、状態表示は
`mismatch=0(legacy:no-emitter)` と明示する。trajectory 結合の健全性を見るのは上表の
最初の2項目（発進前検定 `verify_ppo_p1.py` check(13) は3種とも 0 を assert 済み）

### 実装の禁則
- **サイレント修正・サイレントフォールバック禁止**。解決不能は例外で大声で落とすか、
  カウンタ+WARNING で可視化する（zeros 埋め・黙って skip は過去に run を3本殺した）
- **訓練 rollout への行動上書き禁止**（rule-based guard 等）。eval は本家準拠で guard ON
- 訓練 client は必ず π からの純サンプリング（greedy/top_p 混合禁止）
- 新規 Rust 表面積は最小に。本家挙動に戻せるならリバートを優先
- 検定・診断の self-play client は**本番 client と同一構成**（構成 dump diff==空）
- **訓練側の介入は eval 経路に漏らさない**。介入パラメータは engine の構成 dump に含め、
  eval 側で 0（無効）であることを検定で assert する。現行の対象は
  `p_enrich`（Stage2）/ `call_bonus_b`（Stage3）/ `anchor_prob`・`kl_beta`（anchor 系列）
  で、いずれも検定(17d) に同居（`stage2_design.md` §2 / `anchored_ppo_design.md` §7）

### 実験の規律
- 単一変数アブレーション優先。GPU を焼く前に設計文書を commit
- 判定条件は run 前に固定し、結果を見てから変更しない（post-hoc goalpost 禁止）
- **seed の使い方（2026-07-29 Gamba 裁定・全実験に掛かる）**: 同一 config の run 間
  ばらつきが効果量と同程度であることが判明した（`early_damage_probe_result_20260729.md`
  §9）。よって逐次スクリーニング設計を採る —
  **新規アイデアは全て 1 seed → 事前登録の判定を満たしたものだけ 2 seed 目 →
  2 seed とも同じ方向なら採用候補 / 食い違えば「効果不確実」として保留**。
  「同じ方向」の操作的定義は **2 seed 目を走らせる前に**登録すること
  （結果を見てから決めると post-hoc goalpost）。正は
  `freeparlor/docs/ops/policy_session_0b_decisions_20260729.md` §1
- 挙動の評価は2レンズ併記: argmax eval（配備挙動）と sampled action_mass（学習方向）。
  Stage2 以降は分布にも注意: 訓練測定は濃縮分布上、eval は常に自然分布
  （絶対値の run 跨ぎ比較は不可、倍率同士で比較 — `stage2_design.md` §4）
- 400 step 級スモークで挙動の結論を出さない（分散が支配する。配管検証のみ）
- **定性レビュー（レンズ4）を判定より前に通す**: run 完走 → eval バッテリー（レンズ1–3）→
  **Gamba による牌譜の人手レビュー** → 判定の起草、の順を守る。判定非関与だが、所見は必ず
  定量指標に突き合わせ、乖離があれば診断タスクを起票する（手順は
  `freeparlor/docs/ops/qualitative_review_protocol.md`）。GPU 不要のため後回しにしない
- **基本指標を「表に載せる」で終わらせない**: eval の各レンズで基本指標を **init 基準との
  差分**として評価し、半荘クラスタ SE を付けて有意性まで出す
  （`analyze_fundamentals_1v3.py`。基本 = agari / houjuu / avg_rank、拡張 = fuuro / riichi /
  ryukyoku / 平均和了打点 / 平均放銃打点 / 順位分布 / 順位点·半荘 / 素点·半荘 / チップ·局）。
  劣化が有意なら判定文書の**結論部**に明示する。目的指標（チップ・鳴き）だけで損益を語らない。
  ※ Stage1–3 では基本指標は毎回 eval で出力され結果 md の表にも載っていたが、
  init との差分の有意性を取らなかったため、放銃劣化（z +3.9〜+4.4、全 stage 有意）の
  検出が最大2週間遅れた（データは 2026-07-08 時点で揃っていた）

## 現在の状態（2026-07-25 時点）

> **この節は現在進行中・未決の事項だけを載せる。** 完了済みの経緯（探索ラダー Stage1〜3、
> DRCA プローブ、過去のインシデント史）は `freeparlor/docs/ops/project_history.md` へ
> 不改変で移設した。正は各設計書・判定レポートと git log。
> **状態を変えるタスクを完了したら、この節の更新も同一 commit に含めること**
> （current でなくなった項目は削除ではなく `project_history.md` の末尾へ移す）。

### 現在地（要約）

- **完了**: anchor 系列は **C（象限 IV・不成立）/ K（象限 III・買ったが払った）で決着**。
  引き戻しは相手分布（pool）より**損失側（KL）に置くほうが効く**ことが単一変数で示された。
  ただし基礎劣化はどちらでも完全には止まらず、ダマ和了の消滅（立直マキシマリズム）も不変
- **走行中（2026-07-29 15:06 発進・凍結中）**: **`anchor_k_b04_20260729_150631`** —
  §6a の機械的適用による **kl_beta=0.4 の再走**（Arm K は判定1✗ → β×4。**最大1回・
  これが最後**）。config diff は `ppo_anchor_k.toml` との run パス + `kl_beta` の1行のみ。
  判定条件は §6 と同一。完走後の手順（完走確認 → eval → **レンズ4 → 判定**）と
  留保は `session_handover_20260729.md` §3
- **確定（2026-07-29 Gamba 裁定、正は
  `freeparlor/docs/ops/policy_session_0b_decisions_20260729.md`）**: 次の軸は
  **L1（最適化衛生）の O1（submit_every 50→10）→ O3（ppo_epochs 4→1）**。
  その後 **候補2（敵対的搾取者訓練）→ 候補1（oracle 蒸留）** の順。
  議題1（立直マキシマリズムの商用採否）は**否決**（条件つき — 鳴き判断と牌理が
  init 水準に戻るまで）、議題2（経済定数変更）は**現行ルールでは行わない**、
  議題4/5 は保留。**申し送りは `session_handover_20260729.md`**
- **新規（2026-07-28、最大の発見）**: 中間 checkpoint の軌跡測定により
  **損傷もチップ獲得も最初の 2000 step（全体の 12.5%）でほぼ完了**していることが判明
  （`anchor_checkpoint_trajectory_20260728.md`）。残り 14000 step は「維持（K）か
  喪失（C）か」の期間。**KL アンカーの効能は初期劣化の防止ではなくドリフトの停止**。
  判定1・判定2 を両立する中間 checkpoint は存在しない（全 ckpt で放銃 z ≥ 2.5）
- **運用（2026-07-28）**: ディスク清掃を Gamba 承認の下で実施。判定 commit 済み run の
  `drain`/`buffer` と検証済み smoke run のみ削除し **727GB 回収（184GB → 909GB）**。
  checkpoints / logs / tb / config / game_logs / DRCA tar は全て保全・実在確認済み
- **新規（2026-07-28、レンズ4 起票タスクの消化）**: 指標3本を実装・測定
  （`policy_quality_metrics_20260728.md`、判定非関与）。**鳴き判断は両腕とも双方向に劣化**
  （テンパイ機会の見送り K z=+7.21 / C z=+13.17、かつ**取った鳴きのテンパイ率も低下**
  K z=−2.01 / C z=−3.96 — 「少なく鳴くが良く鳴く」にはなっていない）。一方
  **牌効率と方策の一貫性は Arm C だけが壊れ Arm K は init 水準を保存**
  （向聴を外した率 C z=+22.27 / K n.s.、孤立牌の切り順逆転率 C z=+5.69 / K n.s.）。
  = **判定指標と独立な牌理側の測定でも「pool より損失側（KL）」が再現した**。
  土台の向聴計算器 `shanten.py` は libriichi と **2,259,173 決定点で不一致ゼロ**
- **新規（2026-07-29、最重要）**: 初期損傷プローブを **replicate 2本**完走させ、
  Arm K step2000 を n=400 で測り直して**交絡を全部消した3点比較**を得た
  （`early_damage_probe_result_20260729.md` §9、判定非関与）。
  同一条件（step2000 / n=400 / 同一 seed）で放銃劣化は
  **+2.04pp(z=+2.70) / +0.12pp(+0.16) / +0.96pp(+1.32)** と**連続的に散らばり、
  どの run も外れ値ではない** = **同一 config の run 間ばらつきが効果量と同程度**。
  一方 **立直シフト（両 run とも step200 で 63–67%、終端 75–77%。init 54.29%）と
  降りの中断率の悪化は再現**した。
  → 「損傷は最初の 2000 step で完了」のうち確実なのは**行動シフトと降りの規律**で、
  **放銃率という結果指標の水準は run 間ばらつきに埋もれる**。
  **含意は「1変数 = 1 run で z≈2 級を判定する」設計そのものの検出力**（0b の資源配分）
- **新規（2026-07-29、判定の解釈に効く）**: eval の SE がクラスタ単位の取り違えで
  歪んでいないかを検査（`check_cluster_se.py`）。eval は **1 seed = 1つの山を4回、
  challenger の座席だけローテーション**する設計で、同一 seed の4半荘は**負に相関**
  （ICC −0.04〜−0.32）。**現行の半荘クラスタ SE は過小評価ではなく保守的**で、
  seed クラスタ + 両脚の対応を使うと SE は 0.77–0.81 倍、
  Arm K 判定1 は z=+2.51 → **+3.15**、判定2 は +2.18SE → **+3.35SE** と
  **結論はすべて強まる方向**。判定値は凍結のまま変更しない
- **新規（2026-07-29、B3 の部品）**: `shanten.py` に**形別の向聴分解**
  （通常手/七対子/国士）と**受け入れの形別分解 + 1手先の受け入れ**を追加、
  `yaku.py`（副露手の役判定）と `hand_value.py`（**確定打点** — libriichi の
  `agari.rs`/`point.rs` の移植）を新設。検証は実データで
  向聴 2,259,173 点・受け入れ ⊇ waits 全件・**和了 17,152 件で打点完全一致**。
  副産物: **`AgariDetail.num_aka` は暗槓内の赤を数えない**ため
  **暗槓の赤がチップから漏れている**（実測 0.005枚/半荘 = 判定の効果量の約1/100 で無害）
- **旧（2026-07-28、事前登録・完走済み）**: `early_damage_probe_design.md` —
  step 0–2000 の内部形状を測る短 run（2000 step ≈ 3.1h + eval 数時間）。
  単一変数は観測専用の `diag_save_every`（**`save_every` を下げる素朴案は
  `OpponentPool` の glob 対象を変えて2変数になるので不可** — 同書 §2a）。
  実装・検定(21)・config・launcher まで完了、**発進可否は Gamba 裁定**
- **運用（2026-07-28、負債返済）**: バックログ **11 消化**（run 中監視に実在シグナル2種を
  FATAL 追加、死んだ mismatch は legacy 表示へ降格）、**4/5 の消化を記録**。
  追跡調査で **train_ppo.py の完走後 trainer 再起動**（本家由来の監督ループ）を発見し、
  launcher 側の孤児化・偽トレースバックまで対処（根治は Gamba 裁定待ち = バックログ4）
- **中断**: DRCA プローブ本測定（実効5枠のうち2枠のみ完了、§5a-1c で打ち切り裁定）
- **閉幕**: 探索ラダー Stage1〜3 は全段不成立（本質的機会費用仮説を支持）
- **新規（2026-07-25、GPU 不要で並行実施）**: 「壊れにくい自己学習 PPO」の設計整理。
  壊れにくさを4層に分解した結果、anchor 系列が L2（参照点）/ L3（相手分布）をカバーする一方
  **L1（最適化衛生）と L4（運用・早期検知）が設計・実装ともに空白**であることが確定。
  L1 は既存 8 run の横断実測で3つの構造的事実を得た（下記）。設計ノートは
  `robust_selfplay_ppo_design.md`（DRAFT）、実装はバックログ12（0b / Arm K 判定後）
- **実装エージェント**: Cursor Composer / Claude Code
  （いずれもローカル WSL の GPU・conda 環境・tmux に直接アクセス可）

### 新規: L1（最適化衛生）の横断診断結果（2026-07-25・判定非関与）

正は `freeparlor/docs/reports/ppo_optimization_health_20260725.md`。
**8 run 全てに共通する構造的事実**（anchor_c 固有ではない）:

1. **`minibatch_size = 512` は全 run・全バッチで一度も効いていない**（batch median 168–179、
   512 超 0.00%）。1 optimizer step = 1 半荘の full-batch × 4 epochs。
   `collate_trajectory_batches` はデッドインポート
2. **バッチ到着時点（epoch 1）で既に clip_fraction ≈ 0.20–0.33**。陽性対照
   `trainer_step=0`（client と trainer が同一パラメータ）では clip = 0.0000。
   定常 staleness は lag 1–2 = **50–100 optimizer step**（`submit_every=50`）
3. **4 epochs は trust region 占有をほぼ動かさない**（epoch1→epoch4 の差 −0.0008〜−0.0032）
   → clip は「この更新の行き過ぎを抑える」機能を果たしておらず、
   **収集経験の約2割が恒常的に勾配に寄与していない**

**因果は未検証**（L1 が基礎劣化の原因である証拠はない）。効くのは**判定の解釈**の側で、
「C も K も効かなかった」場合に L1 由来の可能性を排除できないという留保が付く。
**L1 を直すのは C/K 判定後**（凍結を壊さない・過去 run と比較不能になるため）。
なお副産物として `stage1_20260705_053301` の step 10000–10239 に
`lag < 0`（版採番の不整合・resume 境界）が 240/11990 バッチ見つかったが、
Stage1 判定に用いた run ではなく他 7 run では 0 件

### 完了: anchor Arm C 本走 + eval（2026-07-26 完走・判定は牌譜レビュー待ち）

- **完走**: 元 run `anchor_c_20260725_164756`（step 0–14100）→ launcher 事故で切断 →
  `anchor_c_20260726_171144_resume`（step_014000 から 16000 へ、21:27 完走）。
  完走は `reached step 400`… ではなく `reached step 16000` の COMPLETED 経路、
  diag max=16000、`step_016000.pth` は steps=16000（sha256 bcd6e6ee…）で照合済み。
  監視4項目は両 run とも全てゼロ。anchor 採択率は**全期間 0.2459**（16280 draw）で
  anchor が返した checkpoint は 4003 件すべて init
- **インシデント（本日の最重要）**: `run_ppo_p3_stage1_inner.sh` の monitor ループが
  ハードコード 24h DEADLINE を持ち、**期限切れが正常完走と同じ shutdown 経路**
  （Done → Cleanup → exit 0）へ落ちて step 14100/16000 で SIGTERM。「成功を装った失敗」。
  Arm C は 24h を超えた最初の run（anchor は draw の 25% で別 checkpoint をロードする
  ぶん遅い）。修正済み: `MONITOR_HOURS` env（既定 48h）+ `COMPLETED` フラグで
  正常 break のみ通し、期限切れは exit 8。隔離レプリカで両経路実演
- **eval バッテリー完了**（`run_eval_anchor_c.sh`、5レンズ直列、23:27 完了）。
  レンズ2 は判定条件どおり **n=800 両脚**（seed [10000,10200)、牌譜 800×2 実在確認）。
  ミラー較正脚（バックログ5 初適用）は **overall PASS** でレンズ3 のゼロ点を実測裏付け
- **数値（判定は未起草 — レンズ4 のレビューが先）**:
  判定1 の測定器 = 放銃 11.85%→15.45%（**z=+6.57**）、和了 20.60%→17.69%（z=−4.78）、
  avg_rank 2.4863→2.6575（z=+3.02）。判定2 = チップ +0.035→+0.140（差 +0.105、
  **+0.45SE**）。拡張: 平均和了打点 **+1137.8点（z=+7.72）**、平均放銃打点 +43（n.s.）、
  ラス率 +7.50pp（z=+3.33）。メタ対決（vs init×3）素点 −6.913±0.796。
  **機械的には象限 IV（判定1✗ 判定2✗）だが、正式判定は
  `qualitative_review_protocol.md` に従い牌譜レビュー後に起草する**
- **牌譜 HTML 生成済み**: `runs/viewer_out/`（自己対戦 step16000 ×3 + 1v3 ×3）
- **レンズ4 定性レビュー実施済み（2026-07-27、Gamba・1半荘）**:
  `freeparlor/docs/reports/qualitative_review_anchor_c_20260727.md`。
  **(a) 対象の訂正**: レビューされた自己対戦牌譜は `self_play=true` で
  4席すべてが anchor_c step_016000（`mortal_clone` は同一重みのクローン）。
  init の弱さを示すものではない（Gamba 仮説は未検証のまま。直接検証用に全席 init の
  牌譜を `runs/viewer_out/INIT_*` に生成済み）。
  **(b) 最重要の発見**: 所見「打点を作らない」と測定「平均和了打点 +1137.8点(z=+7.72)」の
  矛盾を層別で解消 — **どの層でも打点は有意に上がっていない**
  （立直和了 −147.5 n.s. / ダマ +708 n.s. / 副露 +288 n.s.）。上昇の全量は**構成シフト**
  （立直和了 54.3%→**92.7%**、ダマ 21.3%→**0.8%**、副露 24.4%→**6.5%**）による
  Simpson 型の合成効果。**「平均和了打点の上昇」を Arm C の成果として引用してはならない**。
  (c) 加カンが実質消滅（0.14%→0.01%、z=−3.07）。放銃は打点不変で頻度だけ増加。
  診断は `diagnose_agari_composition.py`（新規・read-only）

### 完了: anchor Arm K 本走 + eval + 判定（2026-07-28、象限 III）

- **完走**: `anchor_k_20260727_000805`（07-27 00:20 発進 → 07-28 00:59、24.6h）。
  `reached step 16000` の COMPLETED 経路 / diag max=16000 / step_016000.pth steps=16000
  （sha256 3c1a993d…）/ 監視4項目ゼロ。**経過 24.6h は旧 24h 期限を超えており、
  MONITOR_HOURS 修正が無ければ step 15,600 付近で「成功を装って」切られていた**
- KL 項は全期間健全: kl_anchor 16,000 件・kl_beta 全て 0.1・**step0=0.0 ちょうど**
  （陽性対照）・非有限 0 件・全期間平均 0.13048
- **判定（`anchor_arm_k_result.md`）: 象限 III（買ったが払った）— C の IV と分かれた**
  - 判定1（基礎維持）**不成立**: 放銃 11.85%→13.16%（**z=+2.51**、閾値2 の境界だが
    緩めない）。ただし C の +3.60pp の**約1/3**で、和了率 −1.03pp・avg_rank −0.018 は
    **いずれも n.s.**（avg_rank は方向としては改善）
  - 判定2（経済適応）**成立**: チップ +0.035→+0.543（差 **+0.508 = +2.18SE**）。
    素点・順位点・合算も全て + 方向（C は素点/順位点が有意に負）。
    メタ対決も好転（チップ +0.680±0.243）
- **全劣化指標で K は C と init の中間**: 降りの中断率 C 49.29%→**K 33.75%**（init 23.78%）、
  副露和了 C 6.53%→**K 15.84%**、役牌絡み C 11.48%→**K 21.39%**、
  **加カンは K で保存**（0.10%、n.s.。C は 0.01% で消滅）
- **両腕に共通する未解決**: ダマ和了が消滅（init 21.30% → C 0.82% / K 0.80%）＝
  **立直マキシマリズムは KL アンカーでも止まらない**。打点上昇（+1017.2点 z=+6.84）は
  C 同様に**構成シフト由来で全層 n.s.** — 成果として引用してはならない
- **レンズ4（Gamba・1半荘、`qualitative_review_anchor_k_20260728.md`）**: 
  「かなり良い。**論外な打ち方が全く無かった**」— C の「見るに堪えない」と対照的で
  定量と同方向。残る所見は七対子決め打ち・打点構築の見送り・鳴き機会の取りこぼし
- **進路は未確定（意図的）**: §6a の再走規定は「判定1✗ → kl_beta×4 で再走（最大1回）」
  を**許容**する（β=0.4、実施可否は資源配分の裁定）。また
  `anchor_c_route_decision_20260726.md` は **C の象限**を軸に書かれており
  「C=IV かつ K=III」を明示的に扱っていないため、**次の軸の選択は 0b の裁定事項**
  （post-hoc に進路を作らない）

### 中断中: DRCA プローブ本測定（§5a-1c 打ち切り裁定・2026-07-25）

- 完了した実効枠は2つ: **第1枠**（セット(a)×Stage1-16000）ΔQ̄ = **−3.2635** 千点 /
  cluster SE 0.7208 / **4.53SE**、**第2枠**（セット(a)×init）ΔQ̄ = **−2.1353** /
  SE 0.4788 / **4.46SE**。中核の問い（反鳴き均衡は基礎劣化アーティファクトか経済の性質か）は
  この挟み撃ちで決着（= 経済の性質）
- **第3枠**（セット(a)×Stage3-16000）は進捗 4.6%（359/7760 rollout）で**中断**。
  成果物は `backups/drca/main_a_s3final_20260725_120218_20260725_163836.tar.gz` に退避、
  run dir 内 `run_resume.sh` で再開可能。残枠（同枠の残り・a_s3mid・b_s3final）は**未測定**
- 放棄した内容は `drca_probe_design.md` §5a-1c に明記（主 contrast 2 全損 → 解釈 C は
  判定せず、contrast 3 測定不能、§4 の A/B/C/D/N 割り当てに到達しない）。
  **最終報告では未測定枠の存在を必ず明示する**（選択的報告の回避）
- 再開可否は改めて裁定。現時点の優先度は anchor 系列より下

### 新規: 教師データ非依存の訓練方式 候補棚卸し（2026-07-26 取り込み・DRAFT・非事前登録）

`freeparlor/docs/design/teacherfree_training_candidates.md`（ブランチ
`claude/ai-training-unlabeled-5gkaoa` から取り込み）。ブレスト成果を
**問題 I（cold start = 天鳳教師フェーズの代替）と 問題 II（均衡脱出）の分離**で整理し、
候補5本を棚卸し。(1) シミュレータ由来の自己教師あり補助タスク（問題 I の最有力・
実体は oracle 蒸留）、(2) リーグ訓練・敵対的搾取者（最も安い・0b 議題3 と同一物）、
(3) KL 正則化ダイナミクス R-NaD 系（**不採用寄り** — 売りの理論保証が4人戦で消える。
裾感度の論点 §4b は再利用のため保存）、(4) 信念状態つき探索（問題 II の本丸・最も高い。
DRCA はその手動1点版）、(5) HITL（当初案は不成立、修正版は「人間は answer ではなく
attention」）。**一次ソース確認済みの事実**（監督側で再検証・全一致）: obs `version=4` は
既に単騎計算機の出力を**入力**に持つ（`obs_repr.rs` の `encode_sp_table`）ため候補1 の
marginal value は隠れ情報 oracle 層にある、観測用 SP 計算は `agent_helper.rs:715-716` で
`calc_tegawari:false`/`calc_shanten_down:false` と意図的に無効（候補1 より安い単一変数
実験候補）、oracle テンソル（217×34）と `AuxNet` 前例は既存で Rust 変更ゼロ・全 call site
`is_oracle=False` で休眠。`policy_session_0b_frame.md` に**議題6（次世代訓練方式の選択）**
を追加（起草時は議題5 だったが同日先行の議題5「ライセンス分界」と衝突し 6 へ繰り下げ。
内容無変更）。**注意: 同書 §7「DRCA 結果条件つきの進路」は DRCA が A/B/C/D/N の判定に
到達する前提で書かれているが、§5a-1c の打ち切り裁定によりその前提は成立しない**
（同書冒頭に監督注記を追記済み）

### 判断の土台（確定済み・詳細は `project_history.md` と各判定レポート）

- **探索ラダー全段不成立**: Stage1 純探索（立直マキシマリズムを発見、赤鳴きは沈む）→
  Stage2 分布介入（配備税を新たに発見）→ Stage3 報酬介入（anneal 内蔵で配備税ゼロ）の
  いずれでも赤鳴きの経済的合理性は覆らず、**本質的機会費用仮説を支持**して閉幕
- **反鳴き均衡は基礎劣化アーティファクトではない**（上記 DRCA 2枠）
- **基礎技能の劣化は有意**: 放銃劣化 z +3.9〜+4.4・和了劣化 −2.1〜−3.6（全 stage）、
  avg_rank 悪化が有意なのは Stage2 のみ（`fundamentals_significance_pass_20260725.md`）。
  これが anchor 系列の動機
- **運用の教訓**: 発進前の残党チェックは**実装完了報告の後にも必須**
  （2026-07-25、報告に現れない孤児 verify プロセスが GPU 1系統ルールを破っていた）。
  run 成果物の保持・清掃は `run_artifact_retention.md` の許可リスト方式で、毎回 Gamba 承認

## 残タスク（バックログ）

消化済み項目（1 / 1b / 2 / 6 / 7 / 8、および 0・9 の完了分）は
`freeparlor/docs/ops/project_history.md` の付録に原文で保全。以下は未決のみ。

0. **DRCA プローブ（診断、事前登録済み）**: 本測定は §5a-1c で**打ち切り・中断**
   （上記「現在の状態」節）。残枠 a_s3final の残り / a_s3mid / b_s3final は未測定。
   再開可否は anchor 系列の帰結と併せて改めて裁定する
0b. **探索ラダー閉幕後の方針設計セッション**（設計監督側）: 議題は **4+1** —
   ①立直マキシマリズムの商用採否 / ②経済定数変更（新実験系）/ ③敵対的搾取者訓練の要否 /
   ④（付随）Stage2b / **⑤ライセンス・データ権利の分界（2026-07-25 追加）** を裁定。
   事前フレームは `freeparlor/docs/ops/policy_session_0b_frame.md`（DRAFT・裁定非関与）、
   設計ギャップの技術検討は `freeparlor/docs/design/product_gaps_design_notes.md`（DRAFT）。
   商品性要件のヒアリングは **2026-07-24 に完了**（同書 §5）。
   **議題5 の要点**: anchor 系列は牌譜由来 init（天鳳2009 = 商用不可、Mortal 本体は AGPL）を
   **学習ループに常駐させる**介入であり、「牌譜依存を減らす」方向と論理的に衝突する。
   「anchor は研究計測器か製品アーキテクチャの一部か」「アンカーを牌譜非依存な参照点に
   置換できることを設計要件にするか」を裁定する枠（材料は同書 §6、技術候補は
   `robust_selfplay_ppo_design.md` §5a）。診断 §6 B（天鳳 BC/蒸留の再導入）も同枠。
   **⚠ 資源配分の未計上（2026-07-28 追記・`robust_selfplay_ppo_design.md` §5c）**:
   Arm K が得たのは**部分的な**保護（放銃劣化を C の約 1/3 に圧縮、ただし z=+2.51 で
   **判定1 は不成立**）で、それは**強い人間譜由来の方策**を参照点にした結果。牌譜非依存
   アンカー（A1 解析的 auxiliary loss / A2 ルールベース蒸留 / A3 scratch 中間 checkpoint）は
   参照点として格段に弱く、**同等以上の保護が出るかは論理的に導けない = 別の実験。
   K が既に判定1 を落としている以上マージンは無い**。検証には **16k run 1本の追加**
   （≈2日 GPU + drain ~230–370GB）が要り、**現行ロードマップに入っていない**
   （議題2 のプリセット run・§6a の kl_beta=0.4 再走と GPU 直列枠を食い合う）。
   なお anchor 系列自体は無駄にならない（「引き戻しは pool より損失側が効く」という
   順序関係と配管は持ち越せる。アンカーは config のパス指定で差し替え可能）。
   **`anchor_checkpoint_trajectory_20260728.md`（損傷は最初の 2000 step でほぼ完了）を
   踏まえると、置換アンカーへの要求は「16000 step 引き戻す」ではなく「最初の 2000 step で
   損傷を防ぐ」に変わる可能性**（A1 に有利な方向・未検証）— 詳細は同書 §5b/§5c、
   0b frame §6.4。
   残り: **セッション実施**（anchor C/K 判定後）
3. **Stage2b（解凍実験）の再評価**: 配備税の発見により「収束済み方策の分布シフト適応」の
   商用価値が上がった。実施判断は 0b と併せて検討
4. ~~**launcher Cleanup 修正の run-validation**~~ **消化済み（2026-07-28 確認）**:
   Arm K（`anchor_k_20260727_000805`）が発進 preflight → 24.6h 完走 → 残党ゼロで
   end-to-end を通した。**ただし追跡調査で別口の欠陥が出た（2026-07-28）**:
   `mortal/train_ppo.py` の `main()` は本家 online DQN 由来の
   `while True: Popen(child); wait()` 監督ループで、**完走の約3秒後にもう1つ子を spawn する**。
   その子は checkpoint を読み直して `steps >= max_steps` で即抜けるので**追加学習は 0 step**
   だが、(a) cmdline が `<python> .../mortal/train_ppo.py` で cleanup の
   `run_train_ppo.py` パターンに掛からず**孤児化して GPU を掴み得る**、
   (b) cleanup が server を先に落とすため `submit_param` が `ConnectionRefusedError` で
   落ち、**完走 run の trainer.log が例外で終わる**（実例: Arm K。判定値・checkpoint は無傷）。
   launcher 側で「trainer 系を server より先に reap + 孫パターン追加」まで対処済み。
   **根治（train_ppo.py 側で完走時に再起動しない）は学習コード変更につき Gamba 裁定待ち**
5. ~~**メタ系ハーネスのミラー較正 RUN**~~ **消化済み（2026-07-27）**: Arm C の eval
   バッテリーで初適用し **overall PASS**（理論ミラー値 素点 −5 / 順位点 0 / チップ 0 と
   SE 圏内で一致）。レンズ3 のゼロ点が実測で裏付けられた（`anchor_arm_c_result.md` §1）
9. **anchor 系列（現行の優先軸）**: 設計凍結済み（`anchored_ppo_design.md`）。実装・
   Arm C 発進まで消化済み。残り: **C 完走（step16000）→ C の eval バッテリー**
   （argmax 6ckpt + 1v3 n=800 + ミラー較正脚 + メタ対決）**→ レンズ4 定性レビュー**
   （`qualitative_review_protocol.md`、判定より前）**→ C 判定 → K の発進 preflight**
   （verify 全20本 + 400-step 配管スモーク + バックログ4 + バックログ11）
   **→ K 発進 → K 判定 → 0b 接続**。
   判定の解釈時に L1 交絡の留保を添える（`ppo_optimization_health_20260725.md` §5）
10. **DRCA の山運ノイズ処理（DRCA-v2 候補・着手時期未定）**:
   **Gamba 裁定（2026-07-25）**: 現行 `drca_probe_design.md` の「鳴きによるツモ順の乖離は
   ノイズではなく鳴きの帰結の一部であり、ペア差分に正しく含まれる」は**言い過ぎ**。山は伏せ
   られており、プレイヤーはずれの中身を意図して選べない。**ΔQ の推定においてはノイズとして
   扱う**（当初は N を増やせば平均消えすると見て放置したが、frame1 極値レビューで裾を支配する
   と判明し立場が変わった — `qualitative_expert_review_drca_frame1_20260722.md` §4b の分類①③）。
   実装方針の含意（DRCA-v2 の事前登録時に確定させる）:
   - **「腕をまたいで自摸列を揃える」実装はバイアスを入れる**ので不可。鳴けば自分のツモ機会が
     減るのは平均としても実在する損であり、それを消すと鳴きを過大評価する
   - **正しい形は「k ごとに未知の山を引き直す」**。Q(s,a) の定義自体が未知情報についての
     期待値なので、これは推定量を定義に近づける操作でバイアスを入れない。ツモ機会が減る損は
     保存され、山運だけが分岐点内で平均消えする。K=8 のコストも変わらない（ほぼ無料）
   - 残る制約: 台本再生 prefix は相手の配牌を固定するため、引き直せるのは**残り山のみ**。
     相手手牌由来の共通ショックは残る（完全な事後分布ではない）。libriichi の盤面介入が要り、
     現行 DRCA の「Rust 表面積ゼロ」方針は破る
   - **既存枠への amendment は不可**（推定量の分散構造が変わる）。測定済みの第1/2枠の
     集計値は無傷（山運は N=485 の分岐点間で平均消えする）。壊れているのは**ケース単位の解釈**
   - **着手条件**: DRCA を再開し、かつケース単位の問い（取りこぼし仮説＝選択的な鳴きの価値）を
     立てるとき。anchor 系列の帰結が出るまで保留
11. ~~**run 中監視の穴埋め（defect）**~~ **消化済み（2026-07-28）**:
   `run_ppo_p3_stage1_inner.sh` の monitor に実在シグナル2種を **FATAL** で追加 —
   `trajectory game key missing`（exit 9、**その局を丸ごと捨てる**）/
   `trajectory orphan steps`（exit 10）。FATAL 判断の根拠は完走3 run
   （stage3 / anchor_c / anchor_k）の実測で**両方とも 0 件**だったこと
   （同期間に非致命の `loader size delta` は 8,759〜12,137 件）＝ 非ゼロは
   ノイズではなく実の整合性破れであり、鉄則「データ整合性シグナルで即停止」に載る。
   死んだ `trajectory step count mismatch` は **grep は残すが「legacy:no-emitter」と
   明示表示**に降格（emitter 復活時に拾えるようにするため撤去はしない）。
   本書「監視期待値」節も同一 commit で実装に合わせた。
   検証: 隔離レプリカ 10 ケース（healthy / loader_delta のみ / 各 FATAL の発火 /
   legacy が 0 のまま / emitter 復活時に exit 4）+ 完走3 run の実ログ回帰
12. **壊れにくい自己学習 PPO の実装（0b / Arm K 判定後・別ブランチ・1変数ずつ）**:
   設計ノートは `freeparlor/docs/design/robust_selfplay_ppo_design.md`（DRAFT・裁定非関与）。
   壊れにくさを4層に分解し、anchor 系列が L2（参照点）/ L3（相手分布）をカバーする一方
   **L1（最適化衛生）と L4（運用・早期検知）が空白**であることを確定した。
   L1 の一次証拠は `freeparlor/docs/reports/ppo_optimization_health_20260725.md`。
   実装候補は L1: O1（`submit_every` 引き下げ）→ O3（`ppo_epochs=1` の対照）→
   O2（複数半荘の batch 束ね。**判定窓を step ではなく消費半荘数で定義し直す必要あり**）、
   L2: D2（param group 分離）→ D3（残差方策 `logits = ref_logits + Δ(s)`。**Arm K の
   ref forward 配線を再利用できるので K の後なら配線のみ**）、L4: 訓練 rollout からの
   基礎指標トリップワイヤ（**観測のみ・自動停止なし**）。
   **凍結中の run（Arm C / Arm K）には一切入れない**

## 役割分担

設計判断・レビュー・仮説の裁定は Claude（chat 側・監督）が担当。このリポジトリでの
あなた（実装エージェント = Cursor Composer / Claude Code 等。現在どれかは
「現在の状態」節を参照）の役割は実装・検証・commit・push。
設計変更が必要だと感じたら、実装せずに提案として報告する。
